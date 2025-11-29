from utils.utils import *
import argparse
from engines.scale_hyperprior import EngineScaleHyperprior
from engines.scale_hyperprior_with_obj_features import (
    EngineScaleHyperpriorWithObjFeatures,
)
from engines.scale_hyperprior_with_obj_eval import EngineScaleHyperpriorWithObjEval


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments of my testing script.")
    parser.add_argument(
        "--which",
        type=str,
        default="hyperprior",
        choices=["hyperprior", "hyperprior_with_obj_features", "hyperprior_eval_with_objs"],
        help="What hyperprior model to train. ['hyperprior', 'hyperprior_with_obj_features', 'hyperprior_eval_with_objs']"
    )
    parser.add_argument(
        "--what",
        type=str,
        default="train",
        choices=["train", "test"],
        help="What to do. ['train', 'test']"
    )
    parser.add_argument(
        "--device", type=str, default=default_device(), help="PyTorch device to use."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size of a forward pass. (default: 256)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of worker threads being used in the DataLoaders. (default: 4)",
    )
    parser.add_argument(
        "--max_lr",
        type=float,
        default=6.0e-4,
        help="Max lr of the scheduler. (default: 2.0e-3)",
    )
    parser.add_argument(
        "--lmbda",
        type=int,
        default=240,
        help="Lambda value of the RD loss. (default: 240)",
    )
    parser.add_argument(
        "--obj_det_img_size",
        type=int,
        default=640,
        help="Size of the images given to the obejct detector (assumed square). (default: 640)",
    )
    parser.add_argument(
        "--compr_img_size",
        type=int,
        default=128,
        help="Size of the images given to the compressor (assumed square). (default: 128)",
    )
    parser.add_argument(
        "--train_size_p1",
        type=float,
        default=0.95,
        help="Proportion of train dataset w.r.t. test. (default: 0.95)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Proportion of only_bb to no_bb distortion loss. (default: 0.8)",
    )
    parser.add_argument(
        "--zero_hooks",
        action="store_true",
        help="Replaces object features hooks with zeros.",
    )
    parser.add_argument(
        "--model_load_file",
        type=str,
        help="A path to the model weights file.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=300,
        help="Number of epochs to train for.",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="./runs",
        help="Logging directory. (default: ./runs)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.which == "hyperprior":
        engine = EngineScaleHyperprior(args)
    elif args.which == "hyperprior_with_obj_features":
        engine = EngineScaleHyperpriorWithObjFeatures(args)
    else:
        engine = EngineScaleHyperpriorWithObjEval(args)

    if args.what == 'train':
        engine.train()
    else:
        engine.test(args.model_load_file)

    print("Bye bye!")
