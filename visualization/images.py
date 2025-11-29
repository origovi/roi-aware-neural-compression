import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from utils.interface import *

def plot_sample_img(gt_image: torch.Tensor, model_image: torch.Tensor, object_set: ROISet=None, title=None, save_file_route=None):
    C, H, W = gt_image.shape
    gt_image_np = gt_image.detach().cpu().permute(1,2,0).numpy()
    model_image_np = model_image.detach().cpu().permute(1,2,0).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    if title is not None:
        fig.suptitle(title)
    axes[0].imshow(gt_image_np)
    axes[0].axis('off')
    axes[1].imshow(model_image_np)
    axes[1].axis('off')
    
    if object_set is not None:
        for object in object_set.objects:
            bb = object.bounding_box_2d
            axes[0].add_patch(Rectangle((bb.xmin*W, bb.ymin*H), (bb.xmax-bb.xmin)*W, (bb.ymax-bb.ymin)*H, edgecolor='r', facecolor='none'))
            # axes[1].add_patch(Rectangle((bb.xmin*W, bb.ymin*H), (bb.xmax-bb.xmin)*W, (bb.ymax-bb.ymin)*H, edgecolor='r', facecolor='none'))
            axes[0].text(bb.xmin*W, bb.ymin*H - 5, object.type, fontsize=12, color='white', bbox=dict(facecolor='red', alpha=0.5, edgecolor='none'))

    plt.tight_layout()
    if save_file_route is None:
        plt.show()
    else:
        plt.savefig(save_file_route, pad_inches=0)
        plt.close()
