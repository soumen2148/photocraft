"""Local AI Segmentation HTTP Server for PhotoCraft (`127.0.0.1:17890`).

Powers `crates/automation/src/ai_segment.rs` (`LocalSegmentationClient`):
  - `subject` -> human-prioritized YOLOX-Nano + MODNet matting (falling back to IS-Net for non-human scenes)
  - `box` / `point` -> SAM 2.1 Hiera Tiny prompted object segmentation
"""
import hashlib
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

from photocraft_ai_hook import LocalAiModels

HOST = "127.0.0.1"
PORT = int(os.environ.get("PHOTOCRAFT_AI_PORT", "17890"))


class ThreadSafeAiEngine:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._models = LocalAiModels()
        try:
            self._models.warm_models()
        except Exception:
            pass

    def subject_mask(self, w: int, h: int, rgb_bytes: bytes) -> bytes:
        with self._lock:
            img = Image.frombytes("RGB", (w, h), rgb_bytes)
            alpha = self._models.subject_alpha(img)
            return alpha.tobytes()

    def sam_box_or_point(self, w: int, h: int, rgb_bytes: bytes, box: list[float] | None, point: list[float] | None) -> bytes:
        with self._lock:
            img = Image.frombytes("RGB", (w, h), rgb_bytes)
            if box is not None and len(box) == 4:
                x0, y0, x1, y1 = box
                xywh = [float(x0), float(y0), max(1.0, float(x1 - x0)), max(1.0, float(y1 - y0))]
                return self._models.sam_box_alpha(img, xywh).tobytes()
            px, py = (point or [w / 2.0, h / 2.0])[:2]
            xywh = [max(0.0, float(px) - 32.0), max(0.0, float(py) - 32.0), 64.0, 64.0]
            return self._models.sam_box_alpha(img, xywh).tobytes()


ENGINE = ThreadSafeAiEngine()


class SegmentHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        print(f"[PhotoCraft-AI] {fmt % args}", flush=True)

    def do_GET(self) -> None:
        if self.path == "/health":
            body = json.dumps(
                {"ok": True, "models": ["yolox-nano", "modnet", "isnet-general-use", "sam2.1-hiera-tiny"]}
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/segment":
            self.send_error(404)
            return
        try:
            t0 = time.perf_counter()
            meta = json.loads(self.headers.get("X-Photocraft-Meta", "{}"))
            length = int(self.headers.get("Content-Length", "0"))
            w, h = int(meta["width"]), int(meta["height"])
            rgb = self.rfile.read(length)
            if len(rgb) != w * h * 3:
                self.send_error(400, "Invalid RGB payload size")
                return
            op = meta.get("op", "subject")
            if op == "subject":
                mask = ENGINE.subject_mask(w, h, rgb)
            elif op == "box":
                mask = ENGINE.sam_box_or_point(w, h, rgb, meta.get("box"), None)
            elif op == "point":
                mask = ENGINE.sam_box_or_point(w, h, rgb, None, meta.get("point"))
            else:
                self.send_error(400, f"Unknown op: {op}")
                return
            dt = time.perf_counter() - t0
            print(f"[PhotoCraft-AI] {op} ({w}x{h}) completed in {dt:.2f}s", flush=True)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(mask)))
            self.end_headers()
            self.wfile.write(mask)
        except Exception as exc:
            print(f"[PhotoCraft-AI] Error: {exc}", flush=True)
            self.send_error(500, str(exc))


if __name__ == "__main__":
    print(f"PhotoCraft Local AI Helper listening on http://{HOST}:{PORT} ...", flush=True)
    ThreadingHTTPServer((HOST, PORT), SegmentHandler).serve_forever()
