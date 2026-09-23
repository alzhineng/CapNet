## CapNet

**CapNet** is a multimodal framework for **Camouflaged Object Detection (COD)** that jointly leverages **RGB, pseudo-depth, and estimated infrared information** to improve object perception in complex camouflage scenes.

The framework is built upon pretrained **SAM/SAM2** representations and consists of three key components:

- **LPA (Linked Proxy Adapter):** performs parameter-efficient task adaptation of the frozen SAM2 encoder.
- **PFM (Progressive Fusion Module):** progressively aligns and fuses complementary multimodal and multi-scale features.
- **FGD (Frequency-Guided Decoder):** exploits high-frequency structural cues to enhance boundary-aware segmentation.

CapNet is evaluated on multiple COD benchmarks and downstream datasets, demonstrating its effectiveness in fine-grained camouflaged object segmentation.
