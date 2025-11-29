import torch
from torch.utils.data import Dataset
import os
import warnings
from torchvision import transforms
from PIL import Image
from utils.interface import ROISet
from tqdm import tqdm
import torchvision.io as io
from data.transforms import RandomSquareCrop


class KittiDetectionDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        obj_det_image_size=640,  # Assume square image
        compr_image_size=128,    # Assume square image
        read_all_images=False,
        load_objects=False
    ):
        self.dataset_path = dataset_path
        self.obj_det_image_size = obj_det_image_size
        self.compr_image_size = compr_image_size
        self.read_all_images = read_all_images
        self.load_objects=load_objects
        self.obj_det_transform = RandomSquareCrop(output_size=obj_det_image_size, return_crop_offset=True)
        self.compr_transform = RandomSquareCrop(output_size=compr_image_size, return_crop_offset=True)

        images_path = os.path.join(self.dataset_path, "image_2")
        labels_path = os.path.join(self.dataset_path, "label_2")
        self.images_files = []
        self.images = []
        self.labels_files = []
        for file in tqdm(os.listdir(images_path), desc="Reading images"):
            full_file = os.path.join(images_path, file)
            self.images_files.append(full_file)
            if self.read_all_images:
                self.images.append(io.read_image(full_file, io.ImageReadMode.RGB))
            self.labels_files.append(
                os.path.join(labels_path, file.replace(".png", ".txt"))
            )
        if len(self.images_files) != len(self.labels_files):
            warnings.warn(
                "Different number of images vs labels. Labels and images could mismatch!"
            )

    def __len__(self):
        return len(self.images_files)


    def __getitem__(self, index):
        # Load & preprocess images
        image_filepath = self.images_files[index]
        # image = Image.open(image_filepath).convert("RGB")

        if self.read_all_images:
            image = self.images[index]
        else:
            image = io.read_image(image_filepath, io.ImageReadMode.RGB)
        _, image_height, image_width = image.shape

        if not self.load_objects:
            compr_image, _ = self.compr_transform(image)
            return compr_image

        else:
            obj_det_image, crop_offset = self.obj_det_transform(image)
            # Load & preprocess labels (objects)
            object_set = None
            label_filepath = self.labels_files[index]
            with open(label_filepath, "r") as label_file:
                object_set = ROISet.from_kitti_txt(label_file.readlines(), image_size=(image_width, image_height))
                object_set.transform_square_crop(crop_offset, orig_size=(image_width, image_height))
            return obj_det_image, object_set
    
    def collate_fn(self, data):
        if self.load_objects:
            obj_images, objects = zip(*data)
            obj_images = torch.stack(obj_images).float()/255
            compr_images = torch.nn.functional.interpolate(obj_images, size=(self.compr_image_size, self.compr_image_size))
            return obj_images, compr_images, objects
        else:
            return torch.stack(data).float()/255


# -------- EXAMPLE --------
# dataset = KittiDetectionDataset("/Users/origovi/Downloads/kitti_2d_objects_original/training")
