"""Downloads the permissive (Apache-2.0) ONNX segmentation models for PhotoCraft AI Smart Selection.

Models downloaded into `models/` (or `PHOTOCRAFT_AI_MODELS_DIR`):
  - `yolox_nano.onnx` (3.6 MB, Apache-2.0) — fast foreground person & held-object detector
  - `modnet.onnx` (25.3 MB, Apache-2.0) — portrait & hair alpha matting
  - `isnet-general-use.onnx` (170 MB, Apache-2.0) — general dichotomous object segmentation
  - `vision_encoder_quantized.onnx` + `.onnx_data` (52.5 MB, Apache-2.0) — SAM 2.1 Hiera Tiny encoder
  - `prompt_encoder_mask_decoder.onnx` + `.onnx_data` (21.0 MB, Apache-2.0) — SAM 2.1 Hiera Tiny decoder
"""
import os
import time
import urllib.request

DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODELS_DIR = os.environ.get("PHOTOCRAFT_AI_MODELS_DIR", DEFAULT_MODELS_DIR)

MODEL_MANIFEST = [
    (
        "yolox_nano.onnx",
        "https://huggingface.co/hr16/yolox-onnx/resolve/main/yolox_nano.onnx",
    ),
    (
        "modnet.onnx",
        "https://huggingface.co/Xenova/modnet/resolve/main/onnx/model.onnx",
    ),
    (
        "isnet-general-use.onnx",
        "https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx",
    ),
    (
        "vision_encoder_quantized.onnx",
        "https://huggingface.co/onnx-community/sam2.1-hiera-tiny-ONNX/resolve/main/onnx/vision_encoder_quantized.onnx",
    ),
    (
        "vision_encoder_quantized.onnx_data",
        "https://huggingface.co/onnx-community/sam2.1-hiera-tiny-ONNX/resolve/main/onnx/vision_encoder_quantized.onnx_data",
    ),
    (
        "prompt_encoder_mask_decoder.onnx",
        "https://huggingface.co/onnx-community/sam2.1-hiera-tiny-ONNX/resolve/main/onnx/prompt_encoder_mask_decoder.onnx",
    ),
    (
        "prompt_encoder_mask_decoder.onnx_data",
        "https://huggingface.co/onnx-community/sam2.1-hiera-tiny-ONNX/resolve/main/onnx/prompt_encoder_mask_decoder.onnx_data",
    ),
]


def download_file(url: str, dest: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Photocraft-dev"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest + ".part", "wb") as out:
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            out.write(chunk)
    os.replace(dest + ".part", dest)


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"Target models directory: {MODELS_DIR}")
    for filename, url in MODEL_MANIFEST:
        dest = os.path.join(MODELS_DIR, filename)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            print(f"  [skip] {filename} ({size_mb:.1f} MB already present)")
            continue
        t0 = time.perf_counter()
        print(f"  [down] {filename} ...", flush=True)
        download_file(url, dest)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"  [done] {filename} ({size_mb:.1f} MB in {time.perf_counter() - t0:.1f}s)")
    print("All AI segmentation models are ready.")


if __name__ == "__main__":
    main()
