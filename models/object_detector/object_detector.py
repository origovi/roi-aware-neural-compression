import ultralytics
import os
import torch


class YOLOv8Kitti2dDetection:
    class _SaveIO:
        """Simple PyTorch hook to save the output of a nn.module."""
        def __init__(self):
            self.input = None
            self.output = None
            
        def __call__(self, module, module_in, module_out):
            self.input = module_in
            self.output = module_out

    def __init__(self, device):
        pretrained_yolo_path = os.path.join(os.path.dirname(__file__), "yolov8n_kitti.pt")
        self.model = ultralytics.YOLO(pretrained_yolo_path).to(device)
        self._hooks = [YOLOv8Kitti2dDetection._SaveIO() for _ in range(len(self.model.model.model))]
        print(f"{len(self._hooks)} hooks have been set for each module in YOLOv8's Sequential.")
        for i, module in enumerate(self.model.model.model):
            module.requires_grad = False  # NOTE this is to freeze weights
            module.register_forward_hook(self._hooks[i])
        self.hook_sizes = self._get_hooks_sizes(device)

    def __call__(self, *args, **kwds):
        return {"result": self.model.__call__(*args, **kwds),
                "hooks": [hook.output for hook in self._hooks]
                }

    def _get_hooks_sizes(self, device):
        """
        Returns the model's hooks with random content. Should be used to check sizes.
        """
        input_zeros = torch.zeros((1, 3, 480, 480), device=device)
        call_res = self.__call__(input_zeros, verbose=False)
        return [hook.shape[1:] for hook in call_res['hooks'][:-1]]  # -1 is to avoid including a tuple that is at the end


    
