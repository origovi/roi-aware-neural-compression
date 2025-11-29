from enum import Enum
from ultralytics.engine.results import Boxes
import torch
from typing import Optional

class KITTIObjType(Enum):
    Car = 0
    Van = 1
    Truck = 2
    Pedestrian = 3
    Person_sitting = 4
    Cyclist = 5
    Tram = 6
    Misc = 7
    DontCare = 8

class ObjectBoundingBox:
    def __init__(self, xmin: float, ymin: float, xmax: float, ymax: float):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
    
    def transform_square_crop(self, crop_offset: tuple[int,int], orig_size: tuple[int,int]) -> int:
        """
        This function transforms the bounding box assuming the image has been
        cropped and returns:
            -> 0 if the bounding box is completely outside the image.
            -> 1 if the bounding box is partially outside the image.
            -> 2 if the bounging box is completely inside the image.
        """
        square_side = min(orig_size[0] - crop_offset[0], orig_size[1] - crop_offset[1])

        # Convert relative coordinates to absolute
        self.xmin *= orig_size[0]
        self.xmax *= orig_size[0]
        self.ymin *= orig_size[1]
        self.ymax *= orig_size[1]
        
        # Shift based on crop offset
        self.xmin -= crop_offset[0]
        self.xmax -= crop_offset[0]
        self.ymin -= crop_offset[1]
        self.ymax -= crop_offset[1]

        # Normalize within cropped image
        self.xmin /= square_side
        self.xmax /= square_side
        self.ymin /= square_side
        self.ymax /= square_side
        
        inside_min = self.xmin >= 0.0 and self.ymin >= 0.0 and self.xmin < 1.0 and self.ymin < 1.0
        inside_max = self.xmax >= 0.0 and self.ymax >= 0.0 and self.xmax < 1.0 and self.ymax < 1.0

        # Clamp to valid range
        self.xmin = max(0, min(1.0, self.xmin))
        self.xmax = max(0, min(1.0, self.xmax))
        self.ymin = max(0, min(1.0, self.ymin))
        self.ymax = max(0, min(1.0, self.ymax))

        return int(inside_min) + int(inside_max)
    
    def scale_to_shape(self, size: tuple[int,int]):
        # Convert relative coordinates to absolute
        xmin = self.xmin * size[0]
        xmax = self.xmax * size[0]
        ymin = self.ymin * size[1]
        ymax = self.ymax * size[1]
        return self.__class__(xmin, ymin, xmax, ymax)



class ROI:
    def __init__(self, type: str, bounding_box_2d: ObjectBoundingBox):
        self.type: str = type
        # Coordinates of the object's bounding box in pixels
        self.bounding_box_2d: ObjectBoundingBox = bounding_box_2d
    
    @classmethod
    def from_kitti_txt(cls, kitti_txt_line: str, image_size: tuple[int,int]):
        # <Type> <Truncated> <Occluded> <Alpha> <Xmin> <Ymin> <Xmax> <Ymax> <3D dims> <3D location> <Rotation>
        split_txt = kitti_txt_line.split(' ')
        type: str = KITTIObjType[split_txt[0]].name
        # Coordinates of the object's bounding box in pixels
        bounding_box_2d = ObjectBoundingBox(float(split_txt[4])/image_size[0], float(split_txt[5])/image_size[1], float(split_txt[6])/image_size[0], float(split_txt[7])/image_size[1])
        return cls(type, bounding_box_2d)
    
    @classmethod
    def from_kitti_ultralytics(cls, type: torch.Tensor, conf: torch.Tensor, xyxyn: torch.Tensor):
        type = KITTIObjType(int(type.item()))
        xyxyn_numpy = xyxyn.numpy()
        bounding_box_2d = ObjectBoundingBox(xyxyn_numpy[0], xyxyn_numpy[1], xyxyn_numpy[2], xyxyn_numpy[3])
        return cls(type, bounding_box_2d)
    
    @classmethod
    def from_coco(cls, type: str, xyxyn: list[float]):
        bounding_box_2d = ObjectBoundingBox(xyxyn[0], xyxyn[1], xyxyn[2], xyxyn[3])
        return cls(type, bounding_box_2d)

    
class ROISet:
    def __init__(self, objects: list[ROI]):
        self.objects = objects

    @classmethod
    def from_kitti_txt(cls, kitti_txt_lines: list[str], image_size: tuple[int,int]):
        objects = [ROI.from_kitti_txt(line.rstrip(), image_size) for line in kitti_txt_lines]
        return cls(objects)

    @classmethod
    def from_kitti_ultralytics(cls, boxes: Boxes):
        objects = [ROI.from_kitti_ultralytics(boxes.cls[i], boxes.conf[i], boxes.xyxyn[i]) for i in range(len(boxes.cls))]
        return cls(objects)
    
    @classmethod
    def from_coco(cls, types: list[str], boxes_xyxyn: list[list[float]]):
        objects = [ROI.from_coco(types[i], boxes_xyxyn[i]) for i in range(len(boxes_xyxyn))]
        return cls(objects)

    def transform_square_crop(self, crop_offset: tuple[int,int], orig_size: tuple[int,int]):
        """
        This function pretends to transform and filter all objects such that:
          1. Objects that lie fully outside the cropped image are discarded from the set.
          2. Objects that lie partially inside the cropped image have updated coordinates.
        """
        new_objects = []
        for object in self.objects:
            result: int = object.bounding_box_2d.transform_square_crop(crop_offset, orig_size)
            if result != 0:
                new_objects.append(object)
        self.objects = new_objects