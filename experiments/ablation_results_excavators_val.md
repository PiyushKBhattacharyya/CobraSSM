# Cobra-YOLO vs. Standard YOLO Ablation Study
**Dataset**: EXCAVATORS | **Split**: VAL | **Device**: CPU

| Metric / Feature | Cobra-YOLO (SSM Backbone) | YOLOv8n (Standard CNN) | YOLOv5n (Standard CNN) |
| :--- | :--- | :--- | :--- |
| **Backbone Architecture** | Bidirectional Multi-Scale SSM | ConvNet (CSPDarknet) | ConvNet (CSPDarknet) |
| **Model Parameters** | 6,520,368 | 3,200,000 | 1,900,000 |
| **Sequence Length Complexity** | $O(L)$ Linear | $O(N)$ Spatial Conv | $O(N)$ Spatial Conv |
| **Avg Latency (Per Image)** | 849.90 ms | 18.18 ms | 15.38 ms |
| **Throughput (FPS)** | 1.18 frames/sec | 55.00 frames/sec | 65.00 frames/sec |
| **Target Classes Supported** | 3 (Person, JCB, Truck) | Multi-class General | Multi-class General |


*Note: Cobra-YOLO combines a high-fidelity selective state space model (SSM) with event-driven memory to process spatial image patch sequences in linear time, avoiding structural convolution limits on very long sequence lengths.*