import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from cobrassm.vision_model import CobraForObjectDetection
from dataset import VisDrone3ClassDataset, ExcavatorsCOCODataset, UAVDTDataset


def compute_yolo_loss(preds, grid_obj, grid_bbox, grid_class):
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

def evaluate(model, val_loader, device):
    model.eval()
    total_val_loss = 0.0
    total_obj = 0.0
    total_box = 0.0
    total_cls = 0.0
    
    with torch.no_grad():
        for img, grid_obj, grid_bbox, grid_class, _ in val_loader:
            img = img.to(device)
            grid_obj = grid_obj.to(device)
            grid_bbox = grid_bbox.to(device)
            grid_class = grid_class.to(device)
            
            outputs = model(img)
            loss, loss_obj, loss_box, loss_cls = compute_yolo_loss(
                outputs, grid_obj, grid_bbox, grid_class
            )
            
            total_val_loss += loss.item()
            total_obj += loss_obj
            total_box += loss_box
            total_cls += loss_cls
            
    num_batches = len(val_loader)
    if num_batches == 0:
        return 0, 0, 0, 0
    return (
        total_val_loss / num_batches,
        total_obj / num_batches,
        total_box / num_batches,
        total_cls / num_batches
    )

def main():
    parser = argparse.ArgumentParser(description="Train Cobra-YOLO Model")
    parser.add_argument("--dataset", type=str, required=True, choices=["visdrone", "excavators", "combined", "uavdt", "combined_all"], help="Dataset to train on")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--img_size", type=int, default=224, help="Image resolution")
    parser.add_argument("--grid_size", type=int, default=14, help="YOLO Grid cell resolution")
    parser.add_argument("--model_dim", type=int, default=256, help="SSM d_model size")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of Cobra layers")
    parser.add_argument("--save_dir", type=str, default="cobra_trained", help="Directory to save checkpoint")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt file to resume training from")
    args = parser.parse_args()

    print("==================================================")
    print(f" Training Cobra-YOLO on: {args.dataset.upper()} ")
    print("==================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths & Datasets configuration
    if args.dataset == "visdrone":
        train_dir = os.path.join("Data", "extracted", "VisDrone2019-DET-train")
        val_dir = os.path.join("Data", "extracted", "VisDrone2019-DET-val")
        
        print("Loading VisDrone dataset splits...")
        train_dataset = VisDrone3ClassDataset(train_dir, img_size=args.img_size, grid_size=args.grid_size, num_classes=3, augment=True)
        val_dataset = VisDrone3ClassDataset(val_dir, img_size=args.img_size, grid_size=args.grid_size, num_classes=3, augment=False)
    elif args.dataset == "excavators":
        root_dir = os.path.join("Data", "Excavators")
        
        print("Loading Excavators dataset splits...")
        train_dataset = ExcavatorsCOCODataset(root_dir, split="train", img_size=args.img_size, grid_size=args.grid_size, num_classes=3, augment=True)
        val_dataset = ExcavatorsCOCODataset(root_dir, split="val", img_size=args.img_size, grid_size=args.grid_size, num_classes=3, augment=False)
    elif args.dataset == "uavdt":
        root_dir = os.path.join("Data", "UAVDT")
        
        print("Loading UAVDT dataset splits...")
        train_dataset = UAVDTDataset(root_dir, split="train", img_size=args.img_size, grid_size=args.grid_size, num_classes=3, augment=True)
        val_dataset = UAVDTDataset(root_dir, split="val", img_size=args.img_size, grid_size=args.grid_size, num_classes=3, augment=False)
    elif args.dataset == "combined":
        visdrone_train_dir = os.path.join("Data", "extracted", "VisDrone2019-DET-train")
        visdrone_val_dir = os.path.join("Data", "extracted", "VisDrone2019-DET-val")
        excavators_root_dir = os.path.join("Data", "Excavators")
        
        print("Loading combined (VisDrone + Excavators) dataset splits...")
        visdrone_train = VisDrone3ClassDataset(visdrone_train_dir, img_size=args.img_size, grid_size=args.grid_size, num_classes=3, augment=True)
        visdrone_val = VisDrone3ClassDataset(visdrone_val_dir, img_size=args.img_size, grid_size=args.grid_size, num_classes=3, augment=False)
        
        excavators_train = ExcavatorsCOCODataset(excavators_root_dir, split="train", img_size=args.img_size, grid_size=args.grid_size, num_classes=3, augment=True)
        excavators_val = ExcavatorsCOCODataset(excavators_root_dir, split="val", img_size=args.img_size, grid_size=args.grid_size, num_classes=3, augment=False)
        
        from torch.utils.data import ConcatDataset
        train_dataset = ConcatDataset([visdrone_train, excavators_train])
        val_dataset = ConcatDataset([visdrone_val, excavators_val])
    else:
        visdrone_train_dir = os.path.join("Data", "extracted", "VisDrone2019-DET-train")
        visdrone_val_dir = os.path.join("Data", "extracted", "VisDrone2019-DET-val")
        excavators_root_dir = os.path.join("Data", "Excavators")
        uavdt_root_dir = os.path.join("Data", "UAVDT")
        
        print("Loading combined all (VisDrone + Excavators + UAVDT) dataset splits...")
        visdrone_train = VisDrone3ClassDataset(visdrone_train_dir, img_size=args.img_size, grid_size=args.grid_size, num_classes=3, augment=True)
        visdrone_val = VisDrone3ClassDataset(visdrone_val_dir, img_size=args.img_size, grid_size=args.grid_size, num_classes=3, augment=False)
        
        excavators_train = ExcavatorsCOCODataset(excavators_root_dir, split="train", img_size=args.img_size, grid_size=args.grid_size, num_classes=3, augment=True)
        excavators_val = ExcavatorsCOCODataset(excavators_root_dir, split="val", img_size=args.img_size, grid_size=args.grid_size, num_classes=3, augment=False)
        
        uavdt_train = UAVDTDataset(uavdt_root_dir, split="train", img_size=args.img_size, grid_size=args.grid_size, num_classes=3, augment=True)
        uavdt_val = UAVDTDataset(uavdt_root_dir, split="val", img_size=args.img_size, grid_size=args.grid_size, num_classes=3, augment=False)
        
        from torch.utils.data import ConcatDataset
        train_dataset = ConcatDataset([visdrone_train, excavators_train, uavdt_train])
        val_dataset = ConcatDataset([visdrone_val, excavators_val, uavdt_val])


    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize Cobra-YOLO model with 3 output classes
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

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.save_dir, exist_ok=True)
    start_epoch = 0
    best_val_loss = float("inf")
    save_path = os.path.join(args.save_dir, f"cobra_yolo_{args.dataset}.pt")

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming training from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint["best_val_loss"]
            print(f"Loaded checkpoint at epoch {checkpoint['epoch']} with best val loss {best_val_loss:.4f}")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded model state_dict (no optimizer/scheduler states found). Starting from epoch 0.")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_obj = 0.0
        epoch_box = 0.0
        epoch_cls = 0.0
        
        for step, (img, grid_obj, grid_bbox, grid_class, _) in enumerate(train_loader):
            img = img.to(device)
            grid_obj = grid_obj.to(device)
            grid_bbox = grid_bbox.to(device)
            grid_class = grid_class.to(device)
            
            optimizer.zero_grad()
            outputs = model(img)
            
            loss, loss_obj, loss_box, loss_cls = compute_yolo_loss(
                outputs, grid_obj, grid_bbox, grid_class
            )
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_obj += loss_obj
            epoch_box += loss_box
            epoch_cls += loss_cls
            
            if (step + 1) % 10 == 0 or step == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] Step [{step+1}/{len(train_loader)}] | "
                      f"Loss: {loss.item():.4f} (Obj: {loss_obj:.4f}, Box: {loss_box:.4f}, Cls: {loss_cls:.4f})")
        
        # Adjust learning rate
        scheduler.step()

        # Validation phase
        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss, val_obj, val_box, val_cls = evaluate(model, val_loader, device)
        
        print("-" * 60)
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f} (Obj: {val_obj:.4f}, Box: {val_box:.4f}, Cls: {val_cls:.4f})")
        print("-" * 60)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss
            }
            torch.save(checkpoint_state, save_path)
            print(f"[*] New best val loss! Saved model checkpoint to {save_path}")

    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
