import torch
import torch.nn as nn
from .cobra_block import CobraBlock, RMSNorm

class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=512):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # B, Ph*Pw, C
        x = self.norm(x)
        return x

class YOLOHead(nn.Module):
    """
    Lightweight YOLO-style Detection Head.
    Takes 2D grid features and predicts bounding boxes, objectness, and class logits.
    """
    def __init__(self, d_model, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.bbox_pred = nn.Conv2d(d_model, 4, kernel_size=1)
        self.obj_pred = nn.Conv2d(d_model, 1, kernel_size=1)
        self.cls_pred = nn.Conv2d(d_model, num_classes, kernel_size=1)

    def forward(self, x_grid):
        # x_grid: (B, C, H, W)
        bboxes = self.bbox_pred(x_grid)   # (B, 4, H, W)
        objness = self.obj_pred(x_grid)   # (B, 1, H, W)
        classes = self.cls_pred(x_grid)   # (B, num_classes, H, W)
        return bboxes, objness, classes

class CobraForObjectDetection(nn.Module):
    """
    Cobra-YOLO Alternative Wrapper.
    Unifies Patch Embeddings, Bidirectional CobraBlocks, and a YOLO Detection Head.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=80,
                 d_model=512, n_layers=8, d_state=16, num_scales=4, num_slots=64):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, d_model)
        
        self.blocks = nn.ModuleList([
            CobraBlock(d_model, d_state, num_scales, num_slots)
            for _ in range(n_layers)
        ])
        
        self.norm_f = RMSNorm(d_model)
        self.yolo_head = YOLOHead(d_model, num_classes)
        
    def forward(self, pixel_values):
        # 1. Patchify
        x = self.patch_embed(pixel_values) # (B, L, D)
        
        # 2. Unified Backbone (Bidirectional scan, no SEP tokens)
        for block in self.blocks:
            x, _, _ = block(x, input_ids=None, bidirectional=True)
            
        x = self.norm_f(x)
        
        # 3. Un-flatten back to 2D grid
        B, L, D = x.shape
        grid_size = self.patch_embed.grid_size
        x_grid = x.transpose(1, 2).reshape(B, D, grid_size, grid_size)
        
        # 4. YOLO Head
        bboxes, objness, classes = self.yolo_head(x_grid)
        
        return {
            "bboxes": bboxes,
            "objectness": objness,
            "classes": classes
        }
