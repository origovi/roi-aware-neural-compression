import torch
from torch.utils.data import Dataset
import os
import torchvision.io as io
from data.transforms import RandomSquareCrop
from pycocotools.coco import COCO
from utils.interface import ROISet


class COCODataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        annotation_path: str,
        obj_det_image_size=640,  # Assume square image
        compr_image_size=128,    # Assume square image
        load_objects=False
    ):
        self.dataset_path = dataset_path
        self.obj_det_image_size = obj_det_image_size
        self.compr_image_size = compr_image_size
        self.load_objects=load_objects
        self.obj_det_transform = RandomSquareCrop(output_size=obj_det_image_size, return_crop_offset=True)
        self.compr_transform = RandomSquareCrop(output_size=compr_image_size, return_crop_offset=True)
        self.coco = COCO(annotation_path)
        self.img_ids = self.coco.getImgIds()

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.dataset_path, img_info['file_name'])

        # Load image
        image = io.read_image(img_path, io.ImageReadMode.RGB)
        _, H, W = image.shape

        if not self.load_objects:
            compr_image, _ = self.compr_transform(image)
            return compr_image

        else:
            obj_det_image, crop_offset = self.obj_det_transform(image)

            # Load annotations
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            # Extract bounding boxes and labels
            boxes_xyxyn = []
            labels = []
            for ann in anns:
                x, y, w, h = ann['bbox']
                boxes_xyxyn.append([x/W, y/H, (x + w)/W, (y + h)/H])  # Convert to [x1, y1, x2, y2]
                labels.append(self.coco.loadCats(ann['category_id'])[0]['name'])

            object_set = ROISet.from_coco(labels, boxes_xyxyn)
            object_set.transform_square_crop(crop_offset, orig_size=(W, H))

            return obj_det_image, object_set
    
    def collate_fn(self, data):
        if self.load_objects:
            obj_det_images, objects = zip(*data)
            obj_det_images = torch.stack(obj_det_images).float()/255
            compr_images = torch.nn.functional.interpolate(obj_det_images, size=(self.compr_image_size, self.compr_image_size))
            return obj_det_images, compr_images, objects
        else:
            return torch.stack(data).float()/255



# -------- EXAMPLE --------
# dataset = KittiDetectionDataset("/Users/origovi/Downloads/kitti_2d_objects_original/training")
