import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from cobrassm.vision_model import CobraForObjectDetection

# Class Mapping for VisDrone to model (0-9)
VISDRONE_CLASSES = {
    1: 0,  # pedestrian -> 0
    2: 1,  # people -> 1
    3: 2,  # bicycle -> 2
    4: 3,  # car -> 3
    5: 4,  # van -> 4
    6: 5,  # truck -> 5
    7: 6,  # tricycle -> 6
    8: 7,  # awning-tricycle -> 7
    9: 8,  # bus -> 8
    10: 9  # motor -> 9
}

class LocalVisDroneDataset(Dataset):
    def __init__(self, root_dir, img_size=224, grid_size=14, num_classes=10):
        self.root_dir = root_dir
        self.img_size = img_size
        self.grid_size = grid_size
        self.num_classes = num_classes
        
        self.img_dir = os.path.join(root_dir, "images")
        self.ann_dir = os.path.join(root_dir, "annotations")
        
        # Get all JPG files
        self.img_paths = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")))
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        basename = os.path.basename(img_path).split(".")[0]
        ann_path = os.path.join(self.ann_dir, f"{basename}.txt")
        
        # Load image
        img = Image.open(img_path).convert("RGB")
        w_img, h_img = img.size
        
        # Parse annotations
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
                        
                        # Only keep valid categories (1 to 10)
                        if category in VISDRONE_CLASSES:
                            model_cat = VISDRONE_CLASSES[category]
                            
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
                            
        # Transform image
        img_tensor = self.transform(img)
        
        # Construct grid targets
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
            
            # Assign (single object per cell simple assignment)
            grid_obj[0, row, col] = 1.0
            grid_bbox[:, row, col] = torch.tensor([xc, yc, wn, hn])
            grid_class[row, col] = cat
            
        return img_tensor, grid_obj, grid_bbox, grid_class, img_path

def compute_yolo_loss(preds, grid_obj, grid_bbox, grid_class, num_classes=10):
    pred_bboxes = preds['bboxes']
    pred_obj = preds['objectness']
    pred_classes = preds['classes']
    
    # 1. Objectness Loss (BCE)
    bce_loss = nn.BCEWithLogitsLoss()
    loss_obj = bce_loss(pred_obj, grid_obj)
    
    # 2. Box Regression and Class Loss (only on active grid cells)
    loss_box = torch.tensor(0.0, device=pred_bboxes.device)
    loss_cls = torch.tensor(0.0, device=pred_bboxes.device)
    
    obj_mask = (grid_obj == 1.0).squeeze(1) # (B, H, W)
    if obj_mask.sum() > 0:
        loss_box = nn.functional.l1_loss(
            pred_bboxes.permute(0, 2, 3, 1)[obj_mask], 
            grid_bbox.permute(0, 2, 3, 1)[obj_mask]
        )
        loss_cls = nn.functional.cross_entropy(
            pred_classes.permute(0, 2, 3, 1)[obj_mask],
            grid_class[obj_mask]
        )
        
    total_loss = 2.0 * loss_obj + 5.0 * loss_box + 1.0 * loss_cls
    return total_loss, loss_obj.item(), loss_box.item(), loss_cls.item()

def draw_boxes(image, bboxes, categories, scores=None, color="red", label_prefix="", num_classes=10):
    from PIL import ImageDraw
    draw = ImageDraw.Draw(image)
    w_img, h_img = image.size
    
    class_map = {
        0: "pedestrian", 1: "people", 2: "bicycle", 3: "car", 
        4: "van", 5: "truck", 6: "tricycle", 7: "awning-tricycle",
        8: "bus", 9: "motor"
    }
    
    for i, (box, cat) in enumerate(zip(bboxes, categories)):
        cat_id = int(cat) % num_classes
        label = class_map.get(cat_id, f"class_{cat_id}")
        
        if scores is not None:
            score = scores[i]
            label = f"{label_prefix}{label}: {score:.2f}"
        else:
            label = f"{label_prefix}{label}"
            
        x_center, y_center, bw, bh = box
        
        # Convert back from center representation to pixel coordinates
        xmin = (x_center - bw / 2) * w_img
        ymin = (y_center - bh / 2) * h_img
        xmax = (x_center + bw / 2) * w_img
        ymax = (y_center + bh / 2) * h_img
        
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
        draw.text((xmin + 2, ymin + 2), label, fill=color)

def main():
    print("==================================================")
    print("  Training & Testing Cobra-YOLO on Local VisDrone ")
    print("==================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Paths
    train_dir = os.path.join("Data", "extracted", "VisDrone2019-DET-train")
    val_dir = os.path.join("Data", "extracted", "VisDrone2019-DET-val")
    
    # Initialize Datasets
    print("Loading datasets...")
    train_dataset = LocalVisDroneDataset(train_dir)
    val_dataset = LocalVisDroneDataset(val_dir)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    # Initialize Model
    print("Initializing CobraForObjectDetection Model...")
    model = CobraForObjectDetection(
        img_size=224,
        patch_size=16,
        d_model=256,
        n_layers=4,
        d_state=16,
        num_scales=4,
        num_slots=16,
        num_classes=10
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    
    # Micro training loop (50 steps)
    model.train()
    print("Starting local micro-training run (50 steps)...")
    
    step = 0
    num_steps = 50
    running = True
    
    while running:
        for img_tensor, grid_obj, grid_bbox, grid_class, _ in train_loader:
            if step >= num_steps:
                running = False
                break
                
            img_tensor = img_tensor.to(device)
            grid_obj = grid_obj.to(device)
            grid_bbox = grid_bbox.to(device)
            grid_class = grid_class.to(device)
            
            optimizer.zero_grad()
            outputs = model(img_tensor)
            
            loss, loss_obj, loss_box, loss_cls = compute_yolo_loss(
                outputs, grid_obj, grid_bbox, grid_class
            )
            
            loss.backward()
            optimizer.step()
            
            if (step + 1) % 5 == 0 or step == 0:
                print(f"Step {step+1}/{num_steps} | Loss: {loss.item():.4f} (Obj: {loss_obj:.4f}, Box: {loss_box:.4f}, Cls: {loss_cls:.4f})")
                
            step += 1
            
    # Save checkpoint
    os.makedirs("cobra_trained", exist_ok=True)
    save_path = "cobra_trained/cobra_yolo_local.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Saved local model weights to {save_path}")
    
    # --------------------------------------------------
    # Inference & Visualization Verification
    # --------------------------------------------------
    print("\nRunning model inference on validation split for verification...")
    model.eval()
    
    # Get a sample from validation
    img_tensor, grid_obj, grid_bbox, grid_class, img_path = val_dataset[0]
    
    # Save a comparison prediction vs ground truth
    orig_image = Image.open(img_path).convert("RGB")
    
    # Prepare batch matching visual input
    img_input = img_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_input)
        
    pred_bboxes = outputs['bboxes'].squeeze(0)      # (4, 14, 14)
    pred_obj = torch.sigmoid(outputs['objectness'].squeeze(0).squeeze(0)) # (14, 14)
    pred_classes = torch.softmax(outputs['classes'].squeeze(0), dim=0)    # (C, 14, 14)
    
    # Decode detected boxes
    grid_size = 14
    detected_boxes = []
    detected_cats = []
    detected_scores = []
    
    threshold = 0.15  # Low threshold for visualization of predictions
    for r in range(grid_size):
        for c in range(grid_size):
            score = pred_obj[r, c].item()
            if score > threshold:
                bbox = pred_bboxes[:, r, c].tolist()
                # Scale cell bbox coordinates back to relative grid coordinates
                xc = (c + max(0.0, min(1.0, bbox[0]))) / grid_size
                yc = (r + max(0.0, min(1.0, bbox[1]))) / grid_size
                wn = max(0.0, min(1.0, bbox[2]))
                hn = max(0.0, min(1.0, bbox[3]))
                
                cat = torch.argmax(pred_classes[:, r, c]).item()
                
                detected_boxes.append([xc, yc, wn, hn])
                detected_cats.append(cat)
                detected_scores.append(score)
                
    # Extract ground truth boxes
    gt_boxes = []
    gt_cats = []
    for r in range(grid_size):
        for c in range(grid_size):
            if grid_obj[0, r, c].item() == 1.0:
                bbox = grid_bbox[:, r, c].tolist()
                gt_boxes.append(bbox)
                gt_cats.append(grid_class[r, c].item())
                
    print(f"Found {len(gt_boxes)} Ground Truth objects.")
    print(f"Cobra-YOLO predicted {len(detected_boxes)} candidate objects above threshold={threshold}.")
    
    img_gt = orig_image.copy()
    img_pred = orig_image.copy()
    
    draw_boxes(img_gt, gt_boxes, gt_cats, color="green", label_prefix="GT: ")
    draw_boxes(img_pred, detected_boxes, detected_cats, scores=detected_scores, color="blue", label_prefix="Pred: ")
    
    # Combine side-by-side
    w, h = orig_image.size
    combined = Image.new('RGB', (w * 2, h))
    combined.paste(img_gt, (0, 0))
    combined.paste(img_pred, (w, 0))
    
    # Save the output image
    out_path = "experiments/local_detection_result.png"
    combined.save(out_path)
    print(f"Successfully saved local detection visualization to {out_path}!")
    
    # Copy to brain artifact directory if possible (using environment or generic check)
    # We will skip absolute hardcoded artifact directories as requested by the user
        
    print("\nTraining and validation run completed successfully!")

if __name__ == "__main__":
    main()
