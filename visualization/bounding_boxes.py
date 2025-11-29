from utils.interface import *
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def plot_bbs(image: torch.Tensor, object_set: ROISet, title=None):
    C, H, W = image.shape
    image = image.permute(1,2,0).numpy()
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    ax.axis("off")  # Hide axes
    for object in object_set.objects:
        bb = object.bounding_box_2d
        ax.add_patch(Rectangle((bb.xmin*W, bb.ymin*H), (bb.xmax-bb.xmin)*W, (bb.ymax-bb.ymin)*H, edgecolor='r', facecolor='none'))
        ax.text(bb.xmin*W, bb.ymin*H - 5, object.type, fontsize=12, color='white', bbox=dict(facecolor='red', alpha=0.5, edgecolor='none'))

    # Add a title with the number of objects detected
    full_title = f"Objects: {len(object_set.objects)}"
    if title:
        full_title = title + "\n" + full_title
    ax.set_title(full_title, fontsize=14, fontweight="bold")
    plt.show()