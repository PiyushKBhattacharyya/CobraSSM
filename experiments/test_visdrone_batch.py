import torch
# pyrefly: ignore [missing-import]
from datasets import load_dataset
from torchvision import transforms
from cobrassm.vision_model import CobraForObjectDetection

def main():
    print("Loading VisDrone2019-DET dataset sample...")
    # We use streaming to avoid downloading the massive full dataset
    try:
        dataset = load_dataset("Voxel51/VisDrone2019-DET", split="val", streaming=True)
        iterator = iter(dataset)
    except Exception as e:
        print(f"Warning, failed to load Voxel51/VisDrone2019-DET: {e}. Trying alternative...")
        dataset = load_dataset("dgural/Data-Curation-for-Visual-AI-Module-4-VisDrone", split="train", streaming=True)
        iterator = iter(dataset)

    batch_images = []
    for i in range(2):
        print(f"Loading image {i+1}...")
        sample = next(iterator)
        img = sample['image'].convert('RGB')  # Ensure 3 channels
        
        # Resize and transform to tensor
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        img_tensor = transform(img)
        batch_images.append(img_tensor)
        
    pixel_values = torch.stack(batch_images) # (2, 3, 224, 224)
    print(f"Prepared batch of shape: {pixel_values.shape}")
    
    print("Initializing CobraForObjectDetection...")
    model = CobraForObjectDetection(
        img_size=224, 
        patch_size=16, 
        d_model=256, 
        n_layers=2, 
        d_state=16, 
        num_scales=4, 
        num_slots=16,
        num_classes=10 # VisDrone classes
    )
    
    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(pixel_values)
        
    print("\n--- Output Shapes ---")
    print(f"BBoxes: {outputs['bboxes'].shape}")
    print(f"Objectness: {outputs['objectness'].shape}")
    print(f"Classes: {outputs['classes'].shape}")
    print("\nVisDrone Batch Test completed successfully!")

if __name__ == "__main__":
    main()
