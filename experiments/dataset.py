import os
import glob
import json
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

# Unified class mappings
# Model outputs: 3 classes
# 0: person/ pedestrains
# 1: jcbs(excavators)
# 2: trucks

VISDRONE_MAPPING = {
    1: 0,  # pedestrian -> 0
    2: 0,  # people -> 0
    6: 2,  # truck -> 2
}

class UnifiedObjectDetectionDataset(Dataset):
    def __init__(self, img_size=224, grid_size=14, num_classes=3, augment=False):
        self.img_size = img_size
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.augment = augment
        
        # Core transform applied to all images
        self.normalize_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        
        # Augmentations for small datasets
        if augment:
            self.color_jitter = transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
            )
        else:
            self.color_jitter = None

    def apply_augmentations(self, img, bboxes):
        """
        Applies color jitter and random horizontal flips to image and bboxes
        """
        if not self.augment:
            return img, bboxes
            
        # 1. Color Jitter
        if self.color_jitter:
            img = self.color_jitter(img)
            
        # 2. Random Horizontal Flip (50% probability)
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            flipped_bboxes = []
            for box in bboxes:
                xc, yc, w, h = box
                # Flipped center xc is mirror of original xc
                flipped_bboxes.append([1.0 - xc, yc, w, h])
            bboxes = flipped_bboxes
            
        return img, bboxes

    def construct_grid_target(self, bboxes, classes):
        grid_obj = torch.zeros(1, self.grid_size, self.grid_size)
        grid_bbox = torch.zeros(4, self.grid_size, self.grid_size)
        grid_class = torch.zeros(self.grid_size, self.grid_size, dtype=torch.long)
        
        for box, cat in zip(bboxes, classes):
            xc, yc, wn, hn = box
            
            # Find grid cell indices
            col = int(xc * self.grid_size)
            row = int(yc * self.grid_size)
            col = max(0, min(self.grid_size - 1, col))
            row = max(0, min(self.grid_size - 1, row))
            
            # Assign (single object per cell assignment)
            grid_obj[0, row, col] = 1.0
            grid_bbox[:, row, col] = torch.tensor([xc, yc, wn, hn])
            grid_class[row, col] = cat
            
        return grid_obj, grid_bbox, grid_class

class VisDrone3ClassDataset(UnifiedObjectDetectionDataset):
    def __init__(self, root_dir, img_size=224, grid_size=14, num_classes=3, augment=False):
        super().__init__(img_size, grid_size, num_classes, augment)
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "images")
        self.ann_dir = os.path.join(root_dir, "annotations")
        
        # Get all JPG files
        self.img_paths = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        basename = os.path.basename(img_path).split(".")[0]
        ann_path = os.path.join(self.ann_dir, f"{basename}.txt")
        
        img = Image.open(img_path).convert("RGB")
        w_img, h_img = img.size
        
        bboxes = []
        classes = []
        
        if os.path.exists(ann_path):
            with open(ann_path, "r") as f:
                for line in f:
                    parts = [p.strip() for p in line.split(",") if p.strip()]
                    if len(parts) >= 6:
                        # xmin, ymin, w, h, score, category
                        xmin = float(parts[0])
                        ymin = float(parts[1])
                        w = float(parts[2])
                        h = float(parts[3])
                        category = int(parts[5])
                        
                        if category in VISDRONE_MAPPING:
                            model_cat = VISDRONE_MAPPING[category]
                            
                            # Normalize relative to image size [0-1]
                            xc = (xmin + w/2) / w_img
                            yc = (ymin + h/2) / h_img
                            wn = w / w_img
                            hn = h / h_img
                            
                            xc = max(0.0, min(1.0, xc))
                            yc = max(0.0, min(1.0, yc))
                            wn = max(0.0, min(1.0, wn))
                            hn = max(0.0, min(1.0, hn))
                            
                            bboxes.append([xc, yc, wn, hn])
                            classes.append(model_cat)
                            
        # Apply augmentations
        img, bboxes = self.apply_augmentations(img, bboxes)
        
        # Apply normalization
        img_tensor = self.normalize_transform(img)
        
        # Build targets
        grid_obj, grid_bbox, grid_class = self.construct_grid_target(bboxes, classes)
        
        return img_tensor, grid_obj, grid_bbox, grid_class, img_path

class ExcavatorsCOCODataset(UnifiedObjectDetectionDataset):
    def __init__(self, root_dir, split="train", img_size=224, grid_size=14, num_classes=3, augment=False):
        super().__init__(img_size, grid_size, num_classes, augment)
        self.split_dir = os.path.join(root_dir, split)
        self.coco_path = os.path.join(self.split_dir, "_annotations.coco.json")
        
        # Load COCO annotations
        with open(self.coco_path, "r") as f:
            self.coco_data = json.load(f)
            
        # Parse images and map ID to file path & properties
        self.images = {img["id"]: img for img in self.coco_data["images"]}
        self.img_ids = sorted(list(self.images.keys()))
        
        # Parse annotations and group by image_id
        self.img_annotations = {}
        for ann in self.coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.img_annotations:
                self.img_annotations[img_id] = []
            self.img_annotations[img_id].append(ann)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        
        img_filename = img_info["file_name"]
        img_path = os.path.join(self.split_dir, img_filename)
        
        img = Image.open(img_path).convert("RGB")
        w_img = img_info["width"]
        h_img = img_info["height"]
        
        bboxes = []
        classes = []
        
        anns = self.img_annotations.get(img_id, [])
        for ann in anns:
            category_id = ann["category_id"]
            # Category 1: JCB -> Model Class 1
            # Category 2: Trucks -> Model Class 2
            if category_id in [1, 2]:
                model_cat = 1 if category_id == 1 else 2
                
                # COCO bbox: [xmin, ymin, width, height]
                xmin, ymin, w, h = ann["bbox"]
                
                # Normalize relative to image size [0-1]
                xc = (xmin + w/2) / w_img
                yc = (ymin + h/2) / h_img
                wn = w / w_img
                hn = h / h_img
                
                xc = max(0.0, min(1.0, xc))
                yc = max(0.0, min(1.0, yc))
                wn = max(0.0, min(1.0, wn))
                hn = max(0.0, min(1.0, hn))
                
                bboxes.append([xc, yc, wn, hn])
                classes.append(model_cat)
                
        # Apply augmentations
        img, bboxes = self.apply_augmentations(img, bboxes)
        
        # Apply normalization
        img_tensor = self.normalize_transform(img)
        
        # Build targets
        grid_obj, grid_bbox, grid_class = self.construct_grid_target(bboxes, classes)
        
        return img_tensor, grid_obj, grid_bbox, grid_class, img_path
