# Cobra-YOLO: Training, Testing, & Ablation Studies

This directory contains modular scripts to train, evaluate, and benchmark **Cobra-YOLO**—a real-time, sequence-based alternative to convolutional YOLO models using a Bidirectional Multi-Scale Selective State Space Model (SSM) backbone.

---

## 🚀 Unified 3-Class Detection Paradigm
To support training a single model architecture across both drone surveillance and industrial machine datasets, we utilize a unified **3-Class System**:
- **Class 0**: `person/ pedestrains`
- **Class 1**: `jcbs(excavators)`
- **Class 2**: `trucks`

### Dataset Filtering & Mappings:
1. **VisDrone**:
   - `pedestrian` (1) & `people` (2) $\rightarrow$ **Class 0**
   - `truck` (6) $\rightarrow$ **Class 2**
   - All other classes are automatically ignored during loading.
2. **Excavators**:
   - `JCB` (1) $\rightarrow$ **Class 1**
   - `Trucks` (2) $\rightarrow$ **Class 2**

---

## 🛠️ Data Scarcity & Robust Augmentations
To prevent overfitting on small splits (e.g., the Excavators dataset), our [dataset.py](dataset.py) implementation incorporates **box-aware data augmentations** during the training phase:
* **Random Horizontal Flipping (50% probability)**: Mirrors the PIL image and mathematically scales bounding box center points ($x_c \leftarrow 1.0 - x_c$) while preserving target dimensions.
* **Color Jitter**: Randomly perturbs brightness, contrast, saturation, and hue to improve visual feature robustness under diverse lighting conditions.

---

## 💻 Running Training (`train_cobra_yolo.py`)

Train the unified model using configurable hyperparameters and automated class adaptation. Best checkpoints are saved to the `--save_dir` directory.

### 1. Train a Single Joint Model (VisDrone + Excavators)
To train **one single joint model file** classifying all 3 classes (`person/ pedestrian`, `jcb (excavator)`, and `truck`) jointly across both datasets:
```powershell
python experiments/train_cobra_yolo.py --dataset combined --epochs 15 --batch_size 4 --lr 1e-4
```

### 2. Train on Excavators only
```powershell
python experiments/train_cobra_yolo.py --dataset excavators --epochs 15 --batch_size 4 --lr 1e-4
```

### 3. Train on VisDrone only
```powershell
python experiments/train_cobra_yolo.py --dataset visdrone --epochs 10 --batch_size 4 --lr 1e-4
```

---

## 📊 Running Evaluation & Visualization (`eval_cobra_yolo.py`)

Run quantitative validation or test metrics on a checkpoint, and generate a side-by-side visualization comparing **Ground Truth vs. Model Predictions**.

### 1. Evaluate on Excavators (Test Split)
```powershell
python experiments/eval_cobra_yolo.py --dataset excavators --split test --checkpoint cobra_trained/cobra_yolo_excavators.pt
```

### 2. Evaluate on VisDrone (Val Split)
```powershell
python experiments/eval_cobra_yolo.py --dataset visdrone --split val --checkpoint cobra_trained/cobra_yolo_visdrone.pt
```

*Note: Visualizations are saved directly to `experiments/vis_result_<dataset>_<split>.png`.*

---

## 🧬 Architectural Ablation Studies (`benchmark_ablation_unified.py`)

Conduct structural comparisons of sequence-based Cobra-YOLO with standard convolutional YOLO baselines (**YOLOv8** and **YOLOv5**) measuring parameter counts, sequence complexities, average latencies (ms), and frame throughput (FPS).

### 1. Run Ablation on Excavators
```powershell
python experiments/benchmark_ablation_unified.py --dataset excavators --split val --checkpoint cobra_trained/cobra_yolo_excavators.pt
```

### 2. Run Ablation on VisDrone
```powershell
python experiments/benchmark_ablation_unified.py --dataset visdrone --split val --checkpoint cobra_trained/cobra_yolo_visdrone.pt
```

*Note: Summaries are saved automatically to a markdown report `experiments/ablation_results_<dataset>_val.md`.*
