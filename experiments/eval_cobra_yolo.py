import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw

from cobrassm.vision_model import CobraForObjectDetection
from dataset import VisDrone3ClassDataset, ExcavatorsCOCODataset

CLASS_MAP = {
    0: "person/ pedestrian",
    1: "jcb (excavator)",
    2: "truck"
}

def draw_boxes(image, bboxes, categories, scores=None, color="red", label_prefix=""):
    draw = ImageDraw.Draw(image)
    w_img, h_img = image.size
    
    for i, (box, cat) in enumerate(zip(bboxes, categories)):
        cat_id = int(cat)
        label = CLASS_MAP.get(cat_id, f"class_{cat_id}")
        
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

def evaluate_and_visualize(model, dataset, device, grid_size=14, threshold=0.15, out_vis_path="eval_result.png"):
    model.eval()
    
    # 1. Choose a random/first sample with objects for visualization
    sample_found = False
    for idx in range(len(dataset)):
        img_tensor, grid_obj, grid_bbox, grid_class, img_path = dataset[idx]
        if grid_obj.sum() > 0: # Found a sample with at least one object
            sample_found = True
            break
            
    if not sample_found:
        img_tensor, grid_obj, grid_bbox, grid_class, img_path = dataset[0]
        
    print(f"Generating detection visualization using image: {img_path}")
    orig_image = Image.open(img_path).convert("RGB")
    
    # Run inference on sample
    img_input = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_input)
        
    pred_bboxes = outputs['bboxes'].squeeze(0).cpu()      # (4, 14, 14)
    pred_obj = torch.sigmoid(outputs['objectness'].squeeze(0).squeeze(0)).cpu() # (14, 14)
    pred_classes = torch.softmax(outputs['classes'].squeeze(0), dim=0).cpu()    # (3, 14, 14)
    
    # Decode predicted boxes
    detected_boxes = []
    detected_cats = []
    detected_scores = []
    
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
                
    print(f"Ground Truth contains {len(gt_boxes)} objects.")
    print(f"Cobra-YOLO predicted {len(detected_boxes)} candidate objects above threshold={threshold}.")
    
    img_gt = orig_image.copy()
    img_pred = orig_image.copy()
    
    # Draw ground truth (green) and predictions (blue/red)
    draw_boxes(img_gt, gt_boxes, gt_cats, color="green", label_prefix="GT: ")
    draw_boxes(img_pred, detected_boxes, detected_cats, scores=detected_scores, color="blue", label_prefix="Pred: ")
    
    # Combine side-by-side
    w, h = orig_image.size
    combined = Image.new('RGB', (w * 2, h))
    combined.paste(img_gt, (0, 0))
    combined.paste(img_pred, (w, 0))
    
    combined.save(out_vis_path)
    print(f"Saved side-by-side detection result to: {out_vis_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Cobra-YOLO Model")
    parser.add_argument("--dataset", type=str, required=True, choices=["visdrone", "excavators"], help="Dataset to evaluate")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split to evaluate on")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint .pt")
    parser.add_argument("--img_size", type=int, default=224, help="Image resolution")
    parser.add_argument("--grid_size", type=int, default=14, help="YOLO Grid cell resolution")
    parser.add_argument("--model_dim", type=int, default=256, help="SSM d_model size")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of Cobra layers")
    parser.add_argument("--threshold", type=float, default=0.15, help="Detection threshold for visualization")
    parser.add_argument("--output_vis", type=str, default=None, help="Output visualization path")
    args = parser.parse_args()

    print("==================================================")
    print(f" Evaluating Cobra-YOLO on: {args.dataset.upper()} ({args.split.upper()}) ")
    print("==================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths & Datasets configuration
    if args.dataset == "visdrone":
        # Note: VisDrone doesn't have a separate test folder in standard local setup, let's map appropriately
        split_folder = "VisDrone2019-DET-val" if args.split == "val" else "VisDrone2019-DET-val"
        data_dir = os.path.join("Data", "extracted", split_folder)
        dataset = VisDrone3ClassDataset(data_dir, img_size=args.img_size, grid_size=args.grid_size, num_classes=3, augment=False)
    else:
        root_dir = os.path.join("Data", "Excavators")
        dataset = ExcavatorsCOCODataset(root_dir, split=args.split, img_size=args.img_size, grid_size=args.grid_size, num_classes=3, augment=False)

    print(f"Split size: {len(dataset)} samples")
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    # Initialize model
    print("Initializing CobraForObjectDetection Model...")
    model = CobraForObjectDetection(
        img_size=args.img_size,
        patch_size=16,
        d_model=args.model_dim,
        n_layers=args.n_layers,
        d_state=16,
        num_scales=4,
        num_slots=16,
        num_classes=3
    ).to(device)

    # Load weights
    print(f"Loading checkpoint weights from: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    
    # Compute Average Evaluation Loss
    model.eval()
    total_loss = 0.0
    total_obj = 0.0
    total_box = 0.0
    total_cls = 0.0
    
    # standard YOLO loss components
    bce_loss = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for img, grid_obj, grid_bbox, grid_class, _ in loader:
            img = img.to(device)
            grid_obj = grid_obj.to(device)
            grid_bbox = grid_bbox.to(device)
            grid_class = grid_class.to(device)
            
            outputs = model(img)
            
            pred_bboxes = outputs['bboxes']
            pred_obj = outputs['objectness']
            pred_classes = outputs['classes']
            
            # Loss calculations
            loss_obj = bce_loss(pred_obj, grid_obj)
            loss_box = torch.tensor(0.0, device=device)
            loss_cls = torch.tensor(0.0, device=device)
            
            obj_mask = (grid_obj == 1.0).squeeze(1)
            if obj_mask.sum() > 0:
                loss_box = nn.functional.l1_loss(
                    pred_bboxes.permute(0, 2, 3, 1)[obj_mask], 
                    grid_bbox.permute(0, 2, 3, 1)[obj_mask]
                )
                loss_cls = nn.functional.cross_entropy(
                    pred_classes.permute(0, 2, 3, 1)[obj_mask],
                    grid_class[obj_mask]
                )
                
            batch_loss = 2.0 * loss_obj + 5.0 * loss_box + 1.0 * loss_cls
            
            total_loss += batch_loss.item()
            total_obj += loss_obj.item()
            total_box += loss_box.item()
            total_cls += loss_cls.item()
            
    num_batches = len(loader)
    avg_loss = total_loss / num_batches
    avg_obj = total_obj / num_batches
    avg_box = total_box / num_batches
    avg_cls = total_cls / num_batches
    
    print("-" * 60)
    print("Performance Evaluation Metrics:")
    print(f"  Average Batch Loss: {avg_loss:.4f}")
    print(f"  Objectness Loss:    {avg_obj:.4f}")
    print(f"  Box L1 Loss:        {avg_box:.4f}")
    print(f"  Class CE Loss:      {avg_cls:.4f}")
    print("-" * 60)

    # 4. Generate visualization
    vis_path = args.output_vis if args.output_vis else f"experiments/vis_result_{args.dataset}_{args.split}.png"
    evaluate_and_visualize(model, dataset, device, grid_size=args.grid_size, threshold=args.threshold, out_vis_path=vis_path)

if __name__ == "__main__":
    main()
