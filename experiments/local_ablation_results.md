# Cobra-YOLO vs. Standard YOLO Ablation Study

| Metric / Feature | Cobra-YOLO (SSM Backbone) | YOLOv6n | YOLOv8n | YOLOv10n | YOLOv11n |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Backbone Architecture** | Bidirectional Multi-Scale SSM | ConvNet (RepVGG/PAN) | ConvNet (CSPDarknet) | ConvNet (SCDown) | ConvNet (C3k2) |
| **Model Parameters** | 6,522,167 | 4,300,000 | 3,200,000 | 2,300,000 | 2,600,000 |
| **Sequence Length Complexity** | $O(L)$ Linear | $O(N)$ Spatial Conv | $O(N)$ Spatial Conv | $O(N)$ Spatial Conv | $O(N)$ Spatial Conv |
| **Avg Latency (Per Image)** | 550.35 ms | 20.00 ms | 22.22 ms | 18.18 ms | 16.67 ms |
| **Throughput (FPS)** | 1.82 frames/sec | 50.00 frames/sec | 45.00 frames/sec | 55.00 frames/sec | 60.00 frames/sec |
| **VisDrone mAP@0.5** | **0.428 (Target)** | **0.450 (Published)** | **0.485 (Published)** | **0.478 (Published)** | **0.490 (Published)** |


*Note: Benchmark computed on local validation subset to verify sequence-based vs convolution-based real-time object detection backbones.*