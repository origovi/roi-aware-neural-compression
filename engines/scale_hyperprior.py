import torch
from models.codec_model import ScaleHyperprior
from data.kitti_detection_dataset import KittiDetectionDataset
from data.coco_dataset import COCODataset
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
from losses.rate_distortion import RateDistortionLoss
from visualization.images import plot_sample_img
from utils.logger import Logger
from typing import Callable
from math import floor
from compressai.zoo.pretrained import load_pretrained


class EngineScaleHyperprior:
    def __init__(self, args):
        # torch.autograd.set_detect_anomaly(True, check_nan=True)
        torch.set_float32_matmul_precision("high")
        self.device = args.device

        self.hparams = self._get_hparams(args)

        # Codec model allows to compress / decompress an image
        self.codec_model = ScaleHyperprior(N=128, M=192).to(self.device)

        # Datasets & Dataloaders
        self.dataset = KittiDetectionDataset("/workspace/kitti_2d/training", obj_det_image_size=args.obj_det_img_size, compr_image_size=args.compr_img_size, read_all_images=False, load_objects=False)
        # self.dataset = COCODataset("/workspace/unlabeled2017", compr_image_size=args.compr_img_size)
        train_size = floor(len(self.dataset)*self.hparams['train_size_p1'])
        self.train_dataloader = DataLoader(
            Subset(self.dataset, range(0, train_size)),
            self.hparams["batch_size"],
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=self.dataset.collate_fn,
            pin_memory=True,
            persistent_workers=True,
        )
        self.test_dataloader = DataLoader(
            Subset(self.dataset, range(train_size, len(self.dataset))),
            self.hparams["batch_size"],
            num_workers=args.num_workers,
            collate_fn=self.dataset.collate_fn,
            pin_memory=True,
        )

        # Logger (important stuff)
        self.logger = Logger(log_dir=args.logdir, hparams=self.hparams, which=args.which, weights_save_rate=50)

    def _get_hparams(self, args):
        """
        Provides a dict with all necessary hyper parameters for the training to
        happen from the provided arguments.
        """
        return {"batch_size": args.batch_size,
                "train_size_p1": args.train_size_p1,
                "max_epochs": args.max_epochs,
                "max_lr": args.max_lr,
                "lambda": args.lmbda,
                "alpha": args.alpha,
                "obj_det_img_size": args.obj_det_img_size,
                "compr_img_size": args.compr_img_size,
                "which": args.which,
                "zero_hooks": args.zero_hooks,
                }

    def train(self):
        print(f"Starting training of ScaleHyperprior on {self.device} device.")
        optimizer = torch.optim.AdamW(self.codec_model.parameters())
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams["max_lr"],
                                                           epochs=self.hparams['max_epochs'],
                                                           steps_per_epoch=len(self.train_dataloader))
        loss_fn = RateDistortionLoss(lmbda=self.hparams["lambda"], metric="mse")

        for epoch in range(self.hparams['max_epochs']):
            print(f"Starting epoch {epoch}...")
            self._train_one_epoch(optimizer, lr_scheduler, loss_fn)
            self._test_one_epoch(loss_fn)
    
    def test(self, model_weights_path):
        print(f"Starting testing of ScaleHyperprior on {self.device} device.")
        if model_weights_path is not None:
            print("Loading model weights... ", end="")
            self.codec_model.load_state_dict(load_pretrained(torch.load(model_weights_path, weights_only=True, map_location=self.device)))
            print("OK")
        loss_fn = RateDistortionLoss(lmbda=self.hparams["lambda"], metric="mse")
        self._test_one_epoch(loss_fn)

    def _train_one_epoch(self, optimizer: torch.optim.Optimizer,
                         lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
                         loss_fn: Callable):
        self.codec_model.train()
        with tqdm(self.train_dataloader, unit="b", desc="Training") as pbar:
            for batch_i, batch in enumerate(pbar):
                compr_batch = batch

                compr_batch = compr_batch.to(self.device)
                optimizer.zero_grad()

                codec_output = self.codec_model.forward(compr_batch)

                loss = loss_fn(codec_output, compr_batch)
                loss['rd_loss'].backward()
                torch.nn.utils.clip_grad_norm_(self.codec_model.parameters(), 1.0)
                optimizer.step()
                last_lr = lr_scheduler.get_last_lr()[0]
                lr_scheduler.step()

                self.logger.log_train_batch(last_lr, loss)
                pbar.set_postfix({'loss': f"{loss['rd_loss'].item():.3f}", "lr": f"{last_lr:.3e}"})

        self.logger.log_train_epoch(len(self.train_dataloader))

    def _test_one_epoch(self, loss_fn: Callable):
        self.codec_model.eval()
        with torch.no_grad():
            for batch_i, batch in enumerate(tqdm(self.test_dataloader, unit="b", desc="Testing")):
                compr_batch = batch

                compr_batch = compr_batch.to(self.device)

                codec_output = self.codec_model.forward(compr_batch)

                loss = loss_fn(codec_output, compr_batch)
                if batch_i == 0:
                    self.logger.checkpoint(self.codec_model, compr_batch[0], codec_output['x_hat'][0], loss)
                self.logger.log_test_batch(loss)

        self.logger.log_test_epoch(len(self.test_dataloader))
        self.logger.epoch_reset()


