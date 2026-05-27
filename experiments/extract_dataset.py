import zipfile
import os

def extract_zip(zip_path, extract_dir, max_files=None):
    print(f"Extracting {zip_path} to {extract_dir}...")
    os.makedirs(extract_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        namelist = zip_ref.namelist()
        
        # Sort files to extract systematically
        files_to_extract = []
        
        # We need both annotations and images
        annotations = [f for f in namelist if "annotations/" in f and f.endswith(".txt")]
        images = [f for f in namelist if "images/" in f and f.endswith(".jpg")]
        
        # Keep them aligned
        img_basenames = {os.path.basename(f).split(".")[0]: f for f in images}
        ann_basenames = {os.path.basename(f).split(".")[0]: f for f in annotations}
        
        common = sorted(list(set(img_basenames.keys()) & set(ann_basenames.keys())))
        
        if max_files:
            common = common[:max_files]
            print(f"Limiting to first {max_files} matching image/annotation pairs to save disk space and speed up CPU execution.")
            
        for name in common:
            files_to_extract.append(img_basenames[name])
            files_to_extract.append(ann_basenames[name])
            
        total = len(files_to_extract)
        for i, file in enumerate(files_to_extract):
            zip_ref.extract(file, extract_dir)
            if (i + 1) % 100 == 0 or i == total - 1:
                print(f"Extracted {i + 1}/{total} files...")
                
    print(f"Finished extracting {zip_path}!\n")

def main():
    data_dir = "Data"
    extract_base = os.path.join(data_dir, "extracted")
    
    # 1. Extract Validation set fully
    val_zip = os.path.join(data_dir, "VisDrone2019-DET-val.zip")
    if os.path.exists(val_zip):
        extract_zip(val_zip, extract_base)
    else:
        print(f"Error: {val_zip} not found!")
        
    # 2. Extract Training set (subset of 500 image-annotation pairs for fast local CPU training)
    train_zip = os.path.join(data_dir, "VisDrone2019-DET-train.zip")
    if os.path.exists(train_zip):
        extract_zip(train_zip, extract_base, max_files=500)
    else:
        print(f"Error: {train_zip} not found!")

if __name__ == "__main__":
    main()
