# Cobra-YOLO: Training, Testing, & Ablation Studies

This directory contains modular scripts to train, evaluate, and benchmark **Cobra-YOLO**—a real-time, sequence-based alternative to convolutional YOLO models using a Bidirectional Multi-Scale Selective State Space Model (SSM) backbone.

---

## 🚀 Unified 3-Class Detection Paradigm
To support training a single model architecture across surveillance and industrial machine datasets, we utilize a unified **3-Class System**:
- **Class 0**: `person/ pedestrians` (from VisDrone)
- **Class 1**: `jcbs (excavators)` (from Excavators)
- **Class 2**: `trucks` (from VisDrone, Excavators, and UAVDT)

### Dataset Filtering & Mappings:
1. **VisDrone**:
   - `pedestrian` (1) & `people` (2) $\rightarrow$ **Class 0**
   - `truck` (6) $\rightarrow$ **Class 2**
   - All other classes are automatically ignored during loading.
2. **Excavators**:
   - `JCB` (1) $\rightarrow$ **Class 1**
   - `Trucks` (2) $\rightarrow$ **Class 2**
3. **UAVDT**:
   - `truck` (2) $\rightarrow$ **Class 2** (only truck is loaded; cars and buses are ignored)

---

## 🛠️ Data Scarcity & Robust Augmentations
To prevent overfitting on small splits (e.g., the Excavators dataset), our [dataset.py](dataset.py) implementation incorporates **box-aware data augmentations** during the training phase:
* **Random Horizontal Flipping (50% probability)**: Mirrors the PIL image and mathematically scales bounding box center points ($x_c \leftarrow 1.0 - x_c$) while preserving target dimensions.
* **Color Jitter**: Randomly perturbs brightness, contrast, saturation, and hue to improve visual feature robustness under diverse lighting conditions.

---

## ⚡ CUDA-Enabled GPU Acceleration
All training and evaluation pipelines automatically detect and use a CUDA-enabled GPU if available.

### Verifying CUDA Availability:
Run the following python line to verify your CUDA setup:
```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

If CUDA is available, PyTorch will automatically allocate model parameters, activations, and labels to the GPU (`cuda`), enabling massive speedups.

---

## 💻 Running Training (`train_cobra_yolo.py`)

Train the unified model using configurable hyperparameters and automated class adaptation. Best checkpoints are saved to the `--save_dir` directory.

### 1. Train on Combined All (VisDrone + Excavators + UAVDT)
To train **one single joint model file** classifying all 3 classes jointly across all three datasets on a CUDA GPU:
```powershell
python experiments/train_cobra_yolo.py --dataset combined_all --epochs 15 --batch_size 8 --lr 1e-4
```

### 2. Train on UAVDT only
To train only on the UAVDT dataset:
```powershell
python experiments/train_cobra_yolo.py --dataset uavdt --epochs 10 --batch_size 8 --lr 1e-4
```

### 3. Train on VisDrone + Excavators
```powershell
python experiments/train_cobra_yolo.py --dataset combined --epochs 15 --batch_size 4 --lr 1e-4
```

---

## 📊 Running Evaluation & Visualization (`eval_cobra_yolo.py`)

Run quantitative validation or test metrics on a checkpoint, and generate a side-by-side visualization comparing **Ground Truth vs. Model Predictions**.

### 1. Evaluate on UAVDT (Val Split)
```powershell
python experiments/eval_cobra_yolo.py --dataset uavdt --split val --checkpoint cobra_trained/cobra_yolo_uavdt.pt
```

### 2. Evaluate on Excavators (Test Split)
```powershell
python experiments/eval_cobra_yolo.py --dataset excavators --split test --checkpoint cobra_trained/cobra_yolo_excavators.pt
```

*Note: Visualizations are saved directly to `experiments/vis_result_<dataset>_<split>.png`.*

---

## 🧬 Architectural Ablation Studies & Benchmarks

Conduct structural comparisons of sequence-based Cobra-YOLO with standard convolutional YOLO baselines (**YOLOv5**, **YOLOv6**, **YOLOv8**, **YOLOv10**, and **YOLOv11**) measuring parameter counts, sequence complexities, average latencies (ms), and frame throughput (FPS).

### 1. Run Local Validation Ablation Benchmark
Verify local performance compared against YOLOv6, YOLOv8, YOLOv10, and YOLOv11 on local validation subsets:
```powershell
python experiments/local_benchmark_ablation.py
```
*Note: Results are printed to the console and automatically saved to `experiments/local_ablation_results.md`.*

### 2. Run Unified Ablation on specific datasets
To benchmark Cobra-YOLO against YOLOv5, YOLOv6, YOLOv8, YOLOv10, and YOLOv11 on UAVDT/VisDrone:
```powershell
python experiments/benchmark_ablation_unified.py --dataset uavdt --split val --checkpoint cobra_trained/cobra_yolo_uavdt.pt
```
*Note: Summaries are saved automatically to a markdown report `experiments/ablation_results_<dataset>_<split>.md`.*

---

## 🔄 Resuming Training from Checkpoints
You can pause and resume training of the Cobra-YOLO models from any saved checkpoint `.pt` file using the `--resume` argument. This automatically loads the model weights, optimizer state, Cosine Annealing learning rate scheduler, and restarts from the correct epoch:

```powershell
python experiments/train_cobra_yolo.py --dataset combined_all --epochs 15 --batch_size 8 --lr 1e-4 --resume cobra_trained/cobra_yolo_combined_all.pt
```
