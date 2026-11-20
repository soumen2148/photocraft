# Local AI Smart Selection for PhotoCraft

Upgrades PhotoCraft's built-in **Select › Subject**, **Remove Background**, and **Object Selection** (`W` box drag) with local CPU-friendly ONNX segmentation models—designed to run smoothly even on dual-core laptops with 8 GB RAM and integrated graphics.

## Two Integration Modes

| Mode                             | Entry Point               | When to Use                                                                                   |
| -------------------------------- | ------------------------- | --------------------------------------------------------------------------------------------- |
| **1. Signed Portable Auto-Hook** | `Start_PhotoCraft_AI.bat` | Works out of the box with official signed `photocraft` desktop releases over Control Protocol |
| **2. Native Engine Backend**     | `ai_segment_server.py`    | Pairs directly with `LocalSegmentationClient` (`crates/automation/src/ai_segment.rs`)         |

## Model Architecture & Licences

Every model in this pipeline is permissively licensed (`Apache-2.0`) and runs locally on `CPUExecutionProvider`:

| Stage / Tool                       | Model (`Apache-2.0`)                   | Size    | Role                                                                            |
| ---------------------------------- | -------------------------------------- | ------- | ------------------------------------------------------------------------------- |
| **1. Foreground Subject Detector** | `YOLOX-Nano`                           | 3.6 MB  | Detects prominent foreground people and held accessories in ~70 ms              |
| **2a. Portrait & Hair Matting**    | `MODNet`                               | 25.3 MB | Isolates people (+ held items) with fine hair alpha while ignoring cars/sofas   |
| **2b. General Object Fallback**    | `IS-Net` (`isnet-general-use`)         | 170 MB  | Segments non-human subjects (products, pets, vehicles, food) at full resolution |
| **3. Box / Point Object Tool**     | `SAM 2.1 Hiera Tiny` (quantized ONNX)  | 73.5 MB | Segments any arbitrary box-dragged object with cached vision embeddings         |

## Quick Start

1. Install Python 3.10+ dependencies:

   ```sh
   pip install onnxruntime pillow numpy
   ```

2. Download the ONNX weights once:

   ```sh
   python tools/ai-smart-selection/download_models.py
   ```

3. Launch PhotoCraft with the invisible menu auto-hook:
   - **Windows:** Double-click `tools/ai-smart-selection/Start_PhotoCraft_AI.bat` (set `PHOTOCRAFT_PORTABLE_DIR` if your portable PhotoCraft folder is not at `../photocraft-portable`).
   - **Custom Rust build:** Start `python tools/ai-smart-selection/ai_segment_server.py` (`127.0.0.1:17890`) and launch your compiled `photocraft` binary.
