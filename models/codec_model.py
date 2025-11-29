import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models.base import CompressionModel
from compressai.models.utils import conv, deconv
from compressai.layers import GDN
import torch.nn.functional as F


class ScaleHyperprior(CompressionModel):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.

    .. code-block:: none

                  ┌───┐    y     ┌───┐  z  ┌───┐ z_hat      z_hat ┌───┐
            x ──►─┤g_a├──►─┬──►──┤h_a├──►──┤ Q ├───►───·⋯⋯·───►───┤h_s├─┐
                  └───┘    │     └───┘     └───┘        EB        └───┘ │
                           ▼                                            │
                         ┌─┴─┐                                          │
                         │ Q │                                          ▼
                         └─┬─┘                                          │
                           │                                            │
                     y_hat ▼                                            │
                           │                                            │
                           ·                                            │
                        GC : ◄─────────────────────◄────────────────────┘
                           ·                 scales_hat
                           │
                     y_hat ▼
                           │
                  ┌───┐    │
        x_hat ──◄─┤g_s├────┘
                  └───┘

        EB = Entropy bottleneck
        GC = Gaussian conditional

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(**kwargs)

        self.entropy_bottleneck = EntropyBottleneck(N)

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

class ProjectionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, target_size, mode='bilinear', use_transpose=False):
        """
        A projection layer that maps a tensor to a specific HxW and channel size.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            target_size (tuple): Desired spatial size (H, W).
            mode (str): Interpolation mode ('bilinear', 'nearest', etc.).
            use_transpose (bool): If True, use a transposed convolution instead of interpolation.
        """
        super(ProjectionLayer, self).__init__()
        self.target_size = target_size
        self.mode = mode
        self.use_transpose = use_transpose
        self.out_channels = out_channels

        # 1x1 convolution to adjust channels
        self.channel_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        if use_transpose:
            # Use transposed convolution to upsample instead of interpolation
            self.upsample = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
        else:
            self.upsample = None

    def forward(self, x):
        """
        Forward pass of the projection layer.
        """
        # Adjust channels first
        x = self.channel_proj(x)

        # Resize spatial dimensions
        if self.use_transpose and x.shape[2] < self.target_size[0]:  # Only upsample if needed
            x = self.upsample(x)
        else:
            x = F.interpolate(x, size=self.target_size, mode=self.mode, align_corners=False)

        return x

class ScaleHyperpriorWithObjectFeatures(CompressionModel):
    r"""Modification of the Scale Hyperprior model from J. Balle, D. Minnen,
    S. Singh, S.J. Hwang, N. Johnston:
    `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018. We add the ability to also read object features and therefore
    to condition the compression on these features.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(**kwargs)

        self.entropy_bottleneck_z = EntropyBottleneck(N)
        self.entropy_bottleneck_w = EntropyBottleneck(N)

        ######################### g_a ###############################
        self.fp_g_a_0 = ProjectionLayer(in_channels=64,
                                        out_channels=32,
                                        target_size=(256,256))
        self.g_a_0 = nn.Sequential(
            conv(3 + self.fp_g_a_0.out_channels, N),
            GDN(N)
        )

        self.fp_g_a_1 = ProjectionLayer(in_channels=128,
                                        out_channels=64,
                                        target_size=(128,128))
        self.g_a_1 = nn.Sequential(
            conv(N + self.fp_g_a_1.out_channels, N),
            GDN(N)
        )

        self.fp_g_a_2 = ProjectionLayer(in_channels=256,
                                        out_channels=256,
                                        target_size=(64,64))
        self.g_a_2 = nn.Sequential(
            conv(N + self.fp_g_a_2.out_channels, N),
            GDN(N),
            conv(N, 2*M),
        )

        ######################### g_s ###############################
        self.g_s = nn.Sequential(
            deconv(2*M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

        ######################### h_a_image #########################
        self.h_a_image = nn.Sequential(
            conv(2*M, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        ######################### h_a_obj ###########################
        self.fp_h_a_obj_hook_0 = ProjectionLayer(in_channels=64,
                                                 out_channels=128,
                                                 target_size=(16,16))
        self.fp_h_a_obj_hook_1 = ProjectionLayer(in_channels=128,
                                                 out_channels=256,
                                                 target_size=(16,16))
        self.fp_h_a_obj_hook_2 = ProjectionLayer(in_channels=256,
                                                 out_channels=512,
                                                 target_size=(16,16))
        self.h_a_obj = nn.Sequential(
            conv(2*M + self.fp_h_a_obj_hook_0.out_channels + self.fp_h_a_obj_hook_1.out_channels + self.fp_h_a_obj_hook_2.out_channels, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        ######################### h_s_image #########################
        self.h_s_image = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        ######################### h_s_obj ###########################
        self.h_s_obj = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x, object_detector_hooks):
        ######################### g_a ###############################

        # Projection of hook 0 to x
        p_hook0_g_a_0 = self.fp_g_a_0(object_detector_hooks[0])
        g_a_0_input = torch.cat([x, p_hook0_g_a_0], dim=1)

        g_a_0 = self.g_a_0(g_a_0_input)

        # Projection of hook 1 to y
        p_hook1_g_a_1 = self.fp_g_a_1(object_detector_hooks[1])
        g_a_1_input = torch.cat([g_a_0, p_hook1_g_a_1], dim=1)

        g_a_1 = self.g_a_1(g_a_1_input)

        # Projection of hook 2 to y
        p_hook2_g_a_2 = self.fp_g_a_2(object_detector_hooks[2])
        g_a_2_input = torch.cat([g_a_1, p_hook2_g_a_2], dim=1)

        y = self.g_a_2(g_a_2_input)

        ##################### image hyperprior ######################

        z = self.h_a_image(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck_z(z)
        scales_z_hat = self.h_s_image(z_hat)

        ################## obj_features hyperprior ##################

        p_hook0_h_a = self.fp_h_a_obj_hook_0(object_detector_hooks[0])
        p_hook1_h_a = self.fp_h_a_obj_hook_1(object_detector_hooks[1])
        p_hook2_h_a = self.fp_h_a_obj_hook_2(object_detector_hooks[2])

        v = torch.cat([y, p_hook0_h_a, p_hook1_h_a, p_hook2_h_a], dim=1)
        w = self.h_a_obj(torch.abs(v))
        w_hat, w_likelihoods = self.entropy_bottleneck_w(w)
        scales_w_hat = self.h_s_obj(w_hat)

        ################## gaussian conditional #####################

        scales_hat = torch.cat([scales_z_hat, scales_w_hat], dim=1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        x_hat = self.g_s(y_hat)

        # print("N", self.N)
        # print("M", self.M)
        # print("x", x.shape)
        # print("hook 0", object_detector_hooks[0].shape)
        # print("hook 1", object_detector_hooks[1].shape)
        # print("hook 2", object_detector_hooks[2].shape)
        # print("p_hook0_g_a_0", p_hook0_g_a_0.shape)
        # print("cat p_hook0_g_a_0", g_a_0_input.shape)
        # print("g_a_0", g_a_0.shape)
        # print("p_hook1_g_a_1", p_hook1_g_a_1.shape)
        # print("cat p_hook1_g_a_1", g_a_1_input.shape)
        # print("g_a_1", g_a_1.shape)
        # print("p_hook2_g_a_2", p_hook2_g_a_2.shape)
        # print("cat p_hook2_g_a_2", g_a_2_input.shape)
        # print("y", y.shape)
        # print("scales_z_hat", scales_z_hat.shape)
        # print("p_hook0_h_a", p_hook0_h_a.shape)
        # print("p_hook1_h_a", p_hook1_h_a.shape)
        # print("p_hook2_h_a", p_hook2_h_a.shape)
        # print("scales_w_hat", scales_w_hat.shape)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods, "w": w_likelihoods},
        }
    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a_0.0.weight"].size(0)
        M = state_dict["g_a_2.2.weight"].size(0)//2
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x, object_detector_hooks):
        ######################### g_a ###############################

        # Projection of hook 0 to x
        p_hook0_g_a_0 = self.fp_g_a_0(object_detector_hooks[0])
        y = torch.cat([x, p_hook0_g_a_0], dim=1)

        y = self.g_a_0(y)

        # Projection of hook 1 to y
        p_hook1_g_a_1 = self.fp_g_a_1(object_detector_hooks[1])
        y = torch.cat([y, p_hook1_g_a_1], dim=1)

        y = self.g_a_1(y)

        # Projection of hook 2 to y
        p_hook2_g_a_2 = self.fp_g_a_2(object_detector_hooks[2])
        y = torch.cat([y, p_hook2_g_a_2], dim=1)

        y = self.g_a_2(y)

        ##################### image hyperprior ######################

        z = self.h_a_image(torch.abs(y))
        z_strings = self.entropy_bottleneck_z.compress(z)
        z_hat = self.entropy_bottleneck_z.decompress(z_strings, z.size()[-2:])
        scales_z_hat = self.h_s_image(z_hat)

        ################## obj_features hyperprior ##################

        p_hook0_h_a = self.fp_h_a_obj_hook_0(object_detector_hooks[0])
        p_hook1_h_a = self.fp_h_a_obj_hook_1(object_detector_hooks[1])
        p_hook2_h_a = self.fp_h_a_obj_hook_2(object_detector_hooks[2])

        v = torch.cat([y, p_hook0_h_a, p_hook1_h_a, p_hook2_h_a], dim=1)
        w = self.h_a_obj(torch.abs(v))
        w_strings = self.entropy_bottleneck_w.compress(w)
        w_hat = self.entropy_bottleneck_w.decompress(w_strings, w.size()[-2:])
        scales_w_hat = self.h_s_obj(w_hat)

        ################## gaussian conditional #####################

        scales_hat = torch.cat([scales_z_hat, scales_w_hat], dim=1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)

        return {"strings": [y_strings, z_strings, w_strings], "shapes": [z.size()[-2:], w.size()[-2:]]}

    def decompress(self, strings, shapes):
        assert isinstance(strings, list) and len(strings) == 3
        assert isinstance(shapes, list) and len(shapes) == 2
        z_hat = self.entropy_bottleneck_z.decompress(strings[1], shapes[0])
        scales_hat_z = self.h_s_image(z_hat)
        
        w_hat = self.entropy_bottleneck_w.decompress(strings[2], shapes[1])
        scales_hat_w = self.h_s_obj(w_hat)
        
        scales_hat = torch.cat([scales_hat_z, scales_hat_w], dim=1)

        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}