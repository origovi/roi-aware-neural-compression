from torch.utils.tensorboard import SummaryWriter
from utils.logger_utils import *
import torch
from datetime import datetime
import os
import json
from visualization.images import plot_sample_img
from utils.interface import *

class Logger(SummaryWriter):
    def __init__(self, log_dir, hparams, which="hyperprior", weights_save_rate=1, comment="", purge_step=None, max_queue=10, flush_secs=120, filename_suffix=""):
        self.log_dir = os.path.join(log_dir, f"{which}_{datetime.now().strftime('%Y%m%d_%H:%M')}_lambda-{hparams['lambda']}_alpha-{hparams['alpha']}_comprImSize-{hparams['compr_img_size']}{'_zeroHooks' if hparams['zero_hooks'] else ''}")
        self.weights_save_rate = weights_save_rate
        super().__init__(self.log_dir, comment, purge_step, max_queue, flush_secs, filename_suffix)
        self.reset()
        self._log_hparams(hparams)
    
    def epoch_reset(self):
        """
        Resets internal cumulative values for the losses.
        Should be called at the beginning/end of each epoch.
        """
        self.train_epoch_loss = {}
        self.test_epoch_loss = {}

    def reset(self):
        """
        Resets internal cumulative values for everything.
        Should be called at start / end of training.
        """
        self.epoch_reset()
        self.train_batch_i = 0
        self.test_batch_i = 0
        self.train_epoch = 0
        self.test_epoch = 0

    def _log_hparams(self, hparams: dict):
        """
        Saves in a file all hparams passed as argument and saves them in tensorboard.
        """
        hparams_filepath = os.path.join(self.log_dir, "hparams.json")
        dump_dict_to_file(hparams, hparams_filepath)
        self.add_hparams(hparams, {})

    def checkpoint(self,
                   codec_model: torch.nn.Module,
                   test_sample: torch.Tensor,
                   test_sample_reconstr: torch.Tensor,
                   loss: dict,
                   object_set: ROISet = None):
        """
        This function saves a checkpoint of the training state:
            - Saves the codec model's weights
            - Saves an example of the model's performance (image)
        """
        checkp_dir = os.path.join(self.log_dir, f"checkpoints")
        os.makedirs(checkp_dir, exist_ok=True)
        if (self.test_epoch+1) % self.weights_save_rate == 0:
            torch.save(codec_model.state_dict(), os.path.join(checkp_dir, f"epoch_{self.test_epoch}.pt"))

        title = f"Performance sample on epoch {self.test_epoch} with {loss['bpp_loss']:.3e} bpp"
        file_route = os.path.join(checkp_dir, f"epoch_{self.test_epoch}.png")
        plot_sample_img(test_sample, test_sample_reconstr.clip(0, 1), title=title, save_file_route=file_route, object_set=object_set)
    
    def log_train_batch(self, lr: float, loss: dict[torch.Tensor]):
        # Log LR
        self.add_scalar("Train Batch LR", lr, self.train_batch_i)

        # Log training loss
        for key, value in loss.items():
            if key not in self.train_epoch_loss:
                self.train_epoch_loss[key] = 0.0
            self.train_epoch_loss[key] += value.item()
            self.add_scalar(f"Train Batch Loss/{key}", value.item(), self.train_batch_i)
        self.train_batch_i += 1

    def log_train_epoch(self, train_dataloader_len):
        for key, value in self.train_epoch_loss.items():
            self.add_scalar(f"Train Epoch Loss/{key}", value/train_dataloader_len, self.train_epoch)
        self.train_epoch += 1
    
    def log_test_batch(self, loss: dict[torch.Tensor]):
        for key, value in loss.items():
            if key not in self.test_epoch_loss:
                self.test_epoch_loss[key] = 0.0
            self.test_epoch_loss[key] += value.item()
            self.add_scalar(f"Test Batch Loss/{key}", value.item(), self.test_batch_i)
        self.test_batch_i += 1

    def log_test_epoch(self, test_dataloader_len):
        for key, value in self.test_epoch_loss.items():
            self.add_scalar(f"Test Epoch Loss/{key}", value/test_dataloader_len, self.test_epoch)
        self.test_epoch += 1