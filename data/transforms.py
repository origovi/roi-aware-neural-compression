import torch
import torchvision.transforms.functional as F
from PIL import Image
from typing import Optional, Union
import random

class RandomSquareCrop:
    def __init__(self, output_size: Optional[int] = None, return_crop_offset=False):
        self.output_size = output_size
        self.return_crop_offset = return_crop_offset

    def __call__(self, image: Union[Image.Image, torch.Tensor]):
        if isinstance(image, Image.Image):
            W, H = image.size
        elif isinstance(image, torch.Tensor):
            _, H, W = image.shape
        else:
            raise TypeError(f"Unsupported input type: {type(image)}")
        
        # Determine the size of the largest square that can fit in the image
        square_size = min(H, W)

        # Pick a random top-left corner for the square crop
        x_offset = random.randint(0, W - square_size) if W > square_size else 0
        y_offset = random.randint(0, H - square_size) if H > square_size else 0

        # Crop the square region
        if isinstance(image, Image.Image):
            cropped_image = image.crop((x_offset, y_offset, x_offset + square_size, y_offset + square_size))
            if self.output_size:
                cropped_image = cropped_image.resize((self.output_size, self.output_size))
        else:
            cropped_image = F.crop(image, y_offset, x_offset, square_size, square_size)
            if self.output_size:
                cropped_image = F.resize(cropped_image, (self.output_size, self.output_size))
        
        if self.return_crop_offset:
            return cropped_image, (x_offset, y_offset)
        else:
            return cropped_image


# def random_square_crop(image: Image.Image, output_size: int = None):
#     width, height = image.size
    
#     # Determine the size of the largest square that can fit in the image
#     square_size = min(width, height)
    
#     # Pick a random top-left corner for the square crop
#     x_offset = random.randint(0, width - square_size) if width > square_size else 0
#     y_offset = random.randint(0, height - square_size) if height > square_size else 0
    
#     # Crop the square region
#     cropped_image = image.crop((x_offset, y_offset, x_offset + square_size, y_offset + square_size))
    
#     # Resize if an output size is specified
#     if output_size:
#         cropped_image = cropped_image.resize((output_size, output_size))
    
#     return cropped_image, (x_offset, y_offset)

# def random_square_crop_T(image: torch.Tensor, output_size: int = None):
#     C, H, W = image.shape
    
#     # Determine the size of the largest square that can fit in the image
#     square_size = min(H, W)
    
#     # Pick a random top-left corner for the square crop
#     x_offset = random.randint(0, W - square_size) if W > square_size else 0
#     y_offset = random.randint(0, H - square_size) if H > square_size else 0
    
#     # Crop the square region
#     cropped_image = F.crop(image, y_offset, x_offset, square_size, square_size)
    
#     # Resize if an output size is specified
#     if output_size:
#         cropped_image = F.resize(cropped_image, (output_size, output_size))
    
#     return cropped_image, (x_offset, y_offset)