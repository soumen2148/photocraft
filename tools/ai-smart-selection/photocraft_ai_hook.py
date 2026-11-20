"""Invisible Native Menu Auto-Hook for Signed PhotoCraft Desktop Releases.

Runs silently in the background alongside `photocraft` over its loopback Control Protocol (`127.0.0.1:7878`):
  - `Select › Subject` -> human-prioritized YOLOX-Nano + MODNet matting (or IS-Net for non-human objects)
  - `Remove Background` -> non-destructive AI Layer Mask on the active layer
  - `Object Selection` (`W` box drag) -> SAM 2.1 Hiera Tiny prompted object selection
"""
import ctypes
import hashlib
import hmac
import json
import os
import secrets
import socket
import subprocess
import sys
import time

import numpy as np
import onnxruntime as ort
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def _resolve_portable_dir() -> str:
    env_dir = os.environ.get("PHOTOCRAFT_PORTABLE_DIR")
    if env_dir:
        return os.path.abspath(env_dir)
    sibling = os.path.join(ROOT_DIR, "photocraft-portable")
    if os.path.exists(sibling):
        return sibling
    return os.path.join(HERE, "photocraft-portable")


def _resolve_models_dir() -> str:
    env_dir = os.environ.get("PHOTOCRAFT_AI_MODELS_DIR")
    if env_dir:
        return os.path.abspath(env_dir)
    for candidate in (
        os.path.join(HERE, "models"),
        os.path.join(ROOT_DIR, "demo-output", "models"),
    ):
        if os.path.isdir(candidate):
            return candidate
    return os.path.join(HERE, "models")


PORTABLE_DIR = _resolve_portable_dir()
APP_EXE = os.path.join(PORTABLE_DIR, "photocraft.exe" if os.name == "nt" else "photocraft")
DATA_DIR = os.path.join(PORTABLE_DIR, "PhotoCraftData")
PREFS_FILE = os.path.join(DATA_DIR, "preferences.json")
TOKEN_FILE = os.path.join(DATA_DIR, "control.token")
MODEL_DIR = _resolve_models_dir()
SCRATCH_DIR = os.path.join(os.path.dirname(MODEL_DIR), ".ai_scratch")
LOG_FILE = os.path.join(SCRATCH_DIR, "ai_hook.log")

CONTROL_HOST = "127.0.0.1"
CONTROL_PORT = int(os.environ.get("PHOTOCRAFT_CONTROL_PORT_HOOK", "7878"))
HOOK_LOCK_PORT = int(os.environ.get("PHOTOCRAFT_AI_LOCK_PORT", "17891"))
READ_ROOT = os.path.expanduser("~")

ISNET_PATH = os.path.join(MODEL_DIR, "isnet-general-use.onnx")
MODNET_PATH = os.path.join(MODEL_DIR, "modnet.onnx")
YOLOX_PATH = os.path.join(MODEL_DIR, "yolox_nano.onnx")
SAM_ENC_PATH = os.path.join(MODEL_DIR, "vision_encoder_quantized.onnx")
SAM_DEC_PATH = os.path.join(MODEL_DIR, "prompt_encoder_mask_decoder.onnx")

COCO_BG_CLASSES = {2, 3, 5, 7, 13, 56, 57, 59, 60}

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SCRATCH_DIR, exist_ok=True)


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _clean_recent_files_in_prefs() -> None:
    if not os.path.exists(PREFS_FILE):
        return
    try:
        with open(PREFS_FILE, "r", encoding="utf-8") as f:
            prefs = json.load(f)
        fh = prefs.get("fileHandling")
        if isinstance(fh, dict) and isinstance(fh.get("recentFiles"), list):
            orig = fh["recentFiles"]
            cleaned = [
                p for p in orig
                if not (
                    isinstance(p, str)
                    and (
                        "_bridge_" in os.path.basename(p)
                        or "_hook_" in os.path.basename(p)
                        or ".ai_scratch" in p
                    )
                )
            ]
            if len(cleaned) != len(orig):
                fh["recentFiles"] = cleaned
                with open(PREFS_FILE, "w", encoding="utf-8") as f:
                    json.dump(prefs, f, indent=2)
    except Exception as e:
        log(f"Prefs cleanup skipped: {e}")


def _find_listening_pid(port: int) -> int | None:
    if os.name != "nt":
        return None
    try:
        out = subprocess.check_output(
            ["netstat", "-ano", "-p", "TCP"],
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        target = f"127.0.0.1:{port}"
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[1] == target and parts[3].upper() == "LISTENING":
                return int(parts[4])
    except Exception:
        pass
    return None


def _list_photocraft_pids() -> list[int]:
    if os.name != "nt":
        return []
    pids: list[int] = []
    try:
        out = subprocess.check_output(
            ["tasklist", "/FI", "IMAGENAME eq photocraft.exe", "/FO", "CSV", "/NH"],
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        for line in out.splitlines():
            line = line.strip()
            if line.startswith('"photocraft.exe"'):
                cols = [c.strip('"') for c in line.split('","')]
                if len(cols) >= 2 and cols[1].isdigit():
                    pids.append(int(cols[1]))
    except Exception:
        pass
    return pids


def _kill_pid(pid: int) -> None:
    if os.name != "nt":
        return
    try:
        subprocess.run(
            ["taskkill", "/F", "/PID", str(pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except Exception:
        pass


def _find_desktop_hwnds(pid: int) -> list[int]:
    if os.name != "nt":
        return [1]
    found_hwnds: list[int] = []
    try:
        user32 = ctypes.windll.user32
        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

        def _cb(hwnd, _lparam):
            proc_id = ctypes.c_ulong()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(proc_id))
            if proc_id.value == pid and user32.GetWindow(hwnd, 4) == 0:
                length = user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buf = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buf, length + 1)
                    if "PhotoCraft" in buf.value or user32.IsWindowVisible(hwnd):
                        found_hwnds.append(hwnd)
            return True

        user32.EnumWindows(WNDENUMPROC(_cb), 0)
    except Exception:
        pass
    return found_hwnds


def _bring_pid_to_front(pid: int) -> None:
    if os.name != "nt":
        return
    try:
        user32 = ctypes.windll.user32
        for hwnd in _find_desktop_hwnds(pid):
            if not user32.IsWindowVisible(hwnd):
                user32.ShowWindow(hwnd, 5)
            elif user32.IsIconic(hwnd):
                user32.ShowWindow(hwnd, 9)
            user32.SetForegroundWindow(hwnd)
    except Exception:
        pass


def _make_sess(path: str) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.enable_cpu_mem_arena = False
    opts.enable_mem_pattern = False
    opts.intra_op_num_threads = 3
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])


def _guided_refine(guide_rgb: np.ndarray, mask_f: np.ndarray, r: int = 6, eps: float = 5e-4) -> np.ndarray:
    def box_mean(arr: np.ndarray, rad: int) -> np.ndarray:
        pad = np.pad(arr, ((rad, rad), (rad, rad)), mode="edge")
        ii = np.pad(pad, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
        area = float((2 * rad + 1) ** 2)
        return (
            ii[2 * rad + 1 :, 2 * rad + 1 :]
            - ii[: -2 * rad - 1, 2 * rad + 1 :]
            - ii[2 * rad + 1 :, : -2 * rad - 1]
            + ii[: -2 * rad - 1, : -2 * rad - 1]
        ) / area

    luma = 0.299 * guide_rgb[:, :, 0] + 0.587 * guide_rgb[:, :, 1] + 0.114 * guide_rgb[:, :, 2]
    m_i = box_mean(luma, r)
    m_p = box_mean(mask_f, r)
    var_i = np.maximum(box_mean(luma * luma, r) - m_i * m_i, 0.0)
    a = (box_mean(luma * mask_f, r) - m_i * m_p) / (var_i + eps)
    b = m_p - a * m_i
    q = np.clip(box_mean(a, r) * luma + box_mean(b, r), 0.0, 1.0)
    return np.where((mask_f > 0.03) & (mask_f < 0.97), q, mask_f)


class LocalAiModels:
    def __init__(self) -> None:
        self._yolox: ort.InferenceSession | None = None
        self._modnet: ort.InferenceSession | None = None
        self._isnet: ort.InferenceSession | None = None
        self._sam_enc: ort.InferenceSession | None = None
        self._sam_dec: ort.InferenceSession | None = None
        self._cached_sam_key: str | None = None
        self._cached_sam_embeds: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._cached_subj_key: str | None = None
        self._cached_subj_alpha: np.ndarray | None = None

    def warm_models(self) -> None:
        if self._yolox is None and os.path.exists(YOLOX_PATH):
            self._yolox = _make_sess(YOLOX_PATH)
        if self._modnet is None and os.path.exists(MODNET_PATH):
            self._modnet = _make_sess(MODNET_PATH)

    def _detect_coco(self, img: Image.Image) -> list[tuple[int, float, list[float]]]:
        if not os.path.exists(YOLOX_PATH):
            return []
        if self._yolox is None:
            self._yolox = _make_sess(YOLOX_PATH)
        w, h = img.size
        r = min(416.0 / w, 416.0 / h)
        nw, nh = int(round(w * r)), int(round(h * r))
        resized = np.asarray(img.resize((nw, nh), Image.Resampling.BILINEAR), dtype=np.float32)[:, :, ::-1]
        padded = np.full((416, 416, 3), 114.0, dtype=np.float32)
        padded[:nh, :nw, :] = resized
        inp = np.transpose(padded, (2, 0, 1))[np.newaxis]
        out = self._yolox.run(None, {self._yolox.get_inputs()[0].name: inp})[0][0]

        grids, strides = [], []
        for st in (8, 16, 32):
            gh, gw = 416 // st, 416 // st
            yv, xv = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
            grids.append(np.stack((xv, yv), axis=-1).reshape(-1, 2))
            strides.append(np.full((gh * gw, 1), st, dtype=np.float32))
        grid = np.concatenate(grids, axis=0)
        stride = np.concatenate(strides, axis=0)

        xy = (out[:, :2] + grid) * stride
        wh = np.exp(out[:, 2:4]) * stride
        boxes = np.concatenate([xy - wh / 2.0, xy + wh / 2.0], axis=1) / r
        scores = out[:, 4:5] * out[:, 5:]
        cls_ids = np.argmax(scores, axis=1)
        cls_scores = scores[np.arange(len(scores)), cls_ids]
        keep = cls_scores > 0.35
        boxes, cls_ids, cls_scores = boxes[keep], cls_ids[keep], cls_scores[keep]

        order = cls_scores.argsort()[::-1]
        dets: list[tuple[int, float, list[float]]] = []
        while len(order) > 0:
            i = order[0]
            dets.append((int(cls_ids[i]), float(cls_scores[i]), [float(v) for v in boxes[i]]))
            if len(order) == 1:
                break
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_o = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
            iou = inter / (area_i + area_o - inter + 1e-6)
            order = order[1:][iou < 0.45]
        return dets

    def _foreground_human_roi(self, w: int, h: int, dets: list[tuple[int, float, list[float]]]) -> np.ndarray | None:
        img_area = float(w * h)
        persons = []
        for cid, sc, b in dets:
            if cid == 0 and sc >= 0.42:
                x0, y0, x1, y1 = max(0.0, b[0]), max(0.0, b[1]), min(float(w), b[2]), min(float(h), b[3])
                area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
                if area >= 0.015 * img_area:
                    cx = 0.5 * (x0 + x1) / w
                    center_w = 1.0 - 0.5 * abs(cx - 0.5) * 2.0
                    edge_pen = 0.30 if (x0 <= 0.01 * w or x1 >= 0.99 * w) else 1.0
                    prom = area * sc * center_w * edge_pen
                    persons.append((prom, area, [x0, y0, x1, y1]))
        if not persons:
            return None

        persons.sort(key=lambda t: t[0], reverse=True)
        top_prom = persons[0][0]
        chosen_boxes = [b for prom, _area, b in persons if prom >= 0.38 * top_prom]

        for cid, sc, b in dets:
            if cid != 0 and cid not in COCO_BG_CLASSES and sc >= 0.35:
                for pb in list(chosen_boxes):
                    ix0, iy0 = max(pb[0], b[0]), max(pb[1], b[1])
                    ix1, iy1 = min(pb[2], b[2]), min(pb[3], b[3])
                    if ix1 > ix0 - 25 and iy1 > iy0 - 25:
                        chosen_boxes.append([max(0.0, b[0]), max(0.0, b[1]), min(float(w), b[2]), min(float(h), b[3])])
                        break

        roi_small = np.zeros((256, 256), dtype=np.float32)
        sx, sy = 256.0 / w, 256.0 / h
        for x0, y0, x1, y1 in chosen_boxes:
            bw, bh = x1 - x0, y1 - y0
            px0 = int(max(0.0, (x0 - 0.32 * bw) * sx))
            py0 = int(max(0.0, (y0 - 0.10 * bh) * sy))
            px1 = int(min(256.0, (x1 + 0.32 * bw) * sx))
            py1 = int(min(256.0, (y1 + 0.08 * bh) * sy))
            roi_small[py0:py1, px0:px1] = 1.0

        return np.asarray(
            Image.fromarray((roi_small * 255).astype(np.uint8), "L").resize((w, h), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0

    def subject_alpha(self, img: Image.Image) -> np.ndarray:
        key = hashlib.blake2b(img.tobytes(), digest_size=16).hexdigest()
        if self._cached_subj_key == key and self._cached_subj_alpha is not None:
            return self._cached_subj_alpha

        w, h = img.size
        rgb_f = np.asarray(img, dtype=np.float32) / 255.0
        dets = self._detect_coco(img)
        human_roi = self._foreground_human_roi(w, h, dets)

        if human_roi is not None and os.path.exists(MODNET_PATH):
            if self._modnet is None:
                self._modnet = _make_sess(MODNET_PATH)
            scale = 512.0 / min(w, h)
            mw = max(32, int(round(w * scale / 32.0)) * 32)
            mh = max(32, int(round(h * scale / 32.0)) * 32)
            m_in = (np.asarray(img.resize((mw, mh), Image.Resampling.BILINEAR), dtype=np.float32) / 127.5) - 1.0
            m_in = np.transpose(m_in, (2, 0, 1))[np.newaxis]
            matte = self._modnet.run(None, {self._modnet.get_inputs()[0].name: m_in})[0][0, 0]
            mask_f = np.asarray(
                Image.fromarray((np.clip(matte, 0.0, 1.0) * 255.0).astype(np.uint8), "L").resize((w, h), Image.Resampling.BILINEAR),
                dtype=np.float32,
            ) / 255.0
            mask_f = mask_f * human_roi
            log("Subject pipeline: Human-prioritized YOLOX + MODNet portrait matting.")
        else:
            if self._isnet is None:
                self._isnet = _make_sess(ISNET_PATH)
            arr = np.asarray(img.resize((1024, 1024), Image.Resampling.BILINEAR), dtype=np.float32)
            mx = float(arr.max())
            if mx > 0:
                arr /= mx
            inp = np.transpose(arr - 0.5, (2, 0, 1))[np.newaxis, :, :, :]
            pred = self._isnet.run(None, {self._isnet.get_inputs()[0].name: inp})[0][0, 0, :, :]
            mn, mx = float(pred.min()), float(pred.max())
            if mx > mn:
                pred = (pred - mn) / (mx - mn)
            mask_f = np.asarray(
                Image.fromarray((np.clip(pred, 0.0, 1.0) * 255.0).astype(np.uint8), "L").resize((w, h), Image.Resampling.BILINEAR),
                dtype=np.float32,
            ) / 255.0
            log("Subject pipeline: General-object IS-Net dichotomous segmentation.")

        refined = _guided_refine(rgb_f, mask_f)
        u8 = (np.clip(refined, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        alpha = np.where(u8 < 12, 0, np.where(u8 > 245, 255, u8)).astype(np.uint8)
        self._cached_subj_key = key
        self._cached_subj_alpha = alpha
        return alpha

    def sam_box_alpha(self, img: Image.Image, box_xywh: list[float]) -> np.ndarray:
        if self._sam_enc is None:
            self._sam_enc = _make_sess(SAM_ENC_PATH)
        if self._sam_dec is None:
            self._sam_dec = _make_sess(SAM_DEC_PATH)
        w, h = img.size
        key = hashlib.blake2b(img.tobytes(), digest_size=16).hexdigest()
        if self._cached_sam_key != key or self._cached_sam_embeds is None:
            arr = np.asarray(img.resize((1024, 1024), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            pv = np.transpose((arr - mean) / std, (2, 0, 1))[np.newaxis, :, :, :]
            self._cached_sam_embeds = tuple(self._sam_enc.run(None, {"pixel_values": pv}))
            self._cached_sam_key = key
        e0, e1, e2 = self._cached_sam_embeds
        sx, sy = 1024.0 / w, 1024.0 / h
        x0, y0, bw, bh = box_xywh
        pad_x, pad_y = max(bw * 0.06, 8.0), max(bh * 0.06, 8.0)
        bx0 = max(0.0, x0 - pad_x) * sx
        by0 = max(0.0, y0 - pad_y) * sy
        bx1 = min(float(w), x0 + bw + pad_x) * sx
        by1 = min(float(h), y0 + bh + pad_y) * sy
        boxes = np.array([[[bx0, by0, bx1, by1]]], dtype=np.float32)
        pts = np.zeros((1, 1, 1, 2), dtype=np.float32)
        lbls = np.full((1, 1, 1), -1, dtype=np.int64)
        ious, masks, _ = self._sam_dec.run(
            None,
            {
                "input_points": pts,
                "input_labels": lbls,
                "input_boxes": boxes,
                "image_embeddings.0": e0,
                "image_embeddings.1": e1,
                "image_embeddings.2": e2,
            },
        )
        best = int(np.argmax(ious[0, 0]))
        logits = np.clip(masks[0, 0, best].astype(np.float32), -50.0, 50.0)
        prob = 1.0 / (1.0 + np.exp(-logits))
        u8 = np.asarray(
            Image.fromarray((prob * 255.0).round().astype(np.uint8), "L").resize((w, h), Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )
        return np.where(u8 < 12, 0, np.where(u8 > 245, 255, u8)).astype(np.uint8)


class Bridge:
    def __init__(self) -> None:
        self._id = 100
        self._sock: socket.socket | None = None
        self._rfile = None
        self._wfile = None

    def close(self) -> None:
        for obj in (self._wfile, self._rfile, self._sock):
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass
        self._sock = None
        self._rfile = None
        self._wfile = None

    def _ensure_connected(self) -> None:
        if self._sock is not None:
            return
        with open(TOKEN_FILE, "r", encoding="ascii") as f:
            token = f.read().strip().lower()
        sock = socket.create_connection((CONTROL_HOST, CONTROL_PORT), timeout=2.5)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(55.0)
        rfile = sock.makefile("r", encoding="utf-8")
        wfile = sock.makefile("w", encoding="utf-8")

        c_nonce = secrets.token_hex(32)
        wfile.write(json.dumps({"id": 1, "method": "auth.challenge", "params": {"clientNonce": c_nonce}}) + "\n")
        wfile.flush()
        ch_line = rfile.readline()
        if not ch_line:
            sock.close()
            raise ConnectionError("EOF during auth.challenge")
        ch = json.loads(ch_line)
        if not ch.get("ok"):
            sock.close()
            raise RuntimeError(ch.get("error", "challenge error"))
        s_nonce = ch["result"]["serverNonce"].lower()
        s_proof = ch["result"]["serverProof"].lower()
        exp_s = hmac.new(
            token.encode("ascii"),
            f"photocraft-control-v1 server\n{c_nonce}\n{s_nonce}".encode("ascii"),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(s_proof, exp_s):
            sock.close()
            raise RuntimeError("server proof mismatch")
        c_proof = hmac.new(
            token.encode("ascii"),
            f"photocraft-control-v1 client\n{c_nonce}\n{s_nonce}".encode("ascii"),
            hashlib.sha256,
        ).hexdigest()
        wfile.write(json.dumps({"id": 2, "method": "auth", "params": {"proof": c_proof}}) + "\n")
        wfile.flush()
        au_line = rfile.readline()
        if not au_line:
            sock.close()
            raise ConnectionError("EOF during auth")
        au = json.loads(au_line)
        if not au.get("ok"):
            sock.close()
            raise RuntimeError(au.get("error", "auth error"))

        self._sock = sock
        self._rfile = rfile
        self._wfile = wfile

    def call(self, method: str, params: dict | None = None):
        last_err: Exception | None = None
        for _attempt in range(2):
            try:
                self._ensure_connected()
                self._id += 1
                self._wfile.write(json.dumps({"id": self._id, "method": method, "params": params or {}}) + "\n")
                self._wfile.flush()
                line = self._rfile.readline()
                if not line:
                    raise ConnectionError("EOF reading control response")
                rep = json.loads(line)
                if not rep.get("ok"):
                    raise RuntimeError(rep.get("error", "command error"))
                return rep["result"]
            except (OSError, ConnectionError) as e:
                last_err = e
                self.close()
        raise last_err if last_err else RuntimeError("control connection failed")

    def cmd(self, command: str, params: dict | None = None):
        return self.call("engine.execute", {"command": command, "params": params or {}})


def load_alpha_into_photocraft(bridge: Bridge, alpha: np.ndarray, as_layer_mask: bool, operation: str = "new") -> None:
    orig_info = bridge.cmd("document.inspect")
    orig_layer = orig_info.get("activeLayer")

    h, w = alpha.shape
    carrier = np.zeros((h, w, 4), dtype=np.uint8)
    carrier[:, :, 3] = alpha
    carrier_abs = os.path.join(SCRATCH_DIR, "_hook_alpha_carrier.png")
    Image.fromarray(carrier, "RGBA").save(carrier_abs, compress_level=1)

    bridge.call("app.open", {"path": carrier_abs})
    bridge.cmd("select.all", {})
    bridge.cmd("edit.copy", {})
    bridge.cmd("file.close", {})

    bridge.cmd("edit.pasteSpecial.pasteInPlace", {})
    tmp_info = bridge.cmd("document.inspect")
    tmp_layer = tmp_info.get("activeLayer")

    bridge.cmd("select.loadSelection", {"channel": "transparency", "layer": tmp_layer, "operation": operation})
    bridge.cmd("layer.delete", {"layer": tmp_layer})

    if orig_layer is not None:
        try:
            bridge.cmd("layer.select", {"layer": orig_layer})
        except Exception:
            pass

    if as_layer_mask:
        try:
            bridge.cmd("layer.layerMask.revealSelection", {})
        except Exception:
            try:
                bridge.cmd("layer.new.layerFromBackground", {})
            except Exception:
                pass
            bridge.cmd("layer.layerMask.revealSelection", {})


def diff_box(before: list[int] | None, after: list[int] | None) -> tuple[list[float] | None, str]:
    if not after:
        return None, "new"
    if not before:
        return [float(v) for v in after], "new"
    bx0, by0, bw, bh = before
    ax0, ay0, aw, ah = after
    bx1, by1 = bx0 + bw, by0 + bh
    ax1, ay1 = ax0 + aw, ay0 + ah
    if ax0 < bx0 - 15:
        return [float(ax0), float(ay0), float(bx0 - ax0 + 20), float(ah)], "add"
    if ax1 > bx1 + 15:
        return [float(bx1 - 20), float(ay0), float(ax1 - bx1 + 20), float(ah)], "add"
    if ay0 < by0 - 15:
        return [float(ax0), float(ay0), float(aw), float(by0 - ay0 + 20)], "add"
    if ay1 > by1 + 15:
        return [float(ax0), float(by1 - 20), float(aw), float(ay1 - by1 + 20)], "add"
    return [float(v) for v in after], "new"


def intercept_once(bridge: Bridge, models: LocalAiModels, trigger: str, after_info: dict) -> int:
    t0 = time.perf_counter()
    after_bounds = after_info.get("selectionBounds")
    log(f"Intercepted '{trigger}' on '{after_info.get('name')}' (rev={after_info.get('revision')})")

    bridge.cmd("edit.undo", {})
    before_info = bridge.cmd("document.inspect")
    before_bounds = before_info.get("selectionBounds")

    src_rel = "_hook_clean_src.png"
    src_abs = os.path.join(SCRATCH_DIR, src_rel)
    bridge.call("app.save", {"path": src_rel})
    with Image.open(src_abs) as raw_im:
        img = raw_im.convert("RGB")

    if trigger == "Select Subject":
        ui = bridge.call("ui.inspect", {})
        sel_mode_idx = ui.get("selectionMode", 0)
        op = {0: "new", 1: "add", 2: "subtract", 3: "intersect"}.get(sel_mode_idx, "new")
        alpha = models.subject_alpha(img)
        load_alpha_into_photocraft(bridge, alpha, as_layer_mask=False, operation=op)
    elif trigger == "Remove Background":
        alpha = models.subject_alpha(img)
        load_alpha_into_photocraft(bridge, alpha, as_layer_mask=True, operation="new")
    elif trigger == "Object Selection":
        ui = bridge.call("ui.inspect", {})
        sel_mode_idx = ui.get("selectionMode", 0)
        box, inferred_op = diff_box(before_bounds, after_bounds)
        op = {1: "add", 2: "subtract", 3: "intersect"}.get(sel_mode_idx, inferred_op)
        if box:
            alpha = models.sam_box_alpha(img, box)
            load_alpha_into_photocraft(bridge, alpha, as_layer_mask=False, operation=op)

    final_rev = bridge.cmd("document.inspect").get("revision", -1)
    log(f"Completed AI replacement for '{trigger}' in {time.perf_counter() - t0:.2f}s (new rev={final_rev})")
    return final_rev


def ensure_single_hooked_photocraft(extra_files: list[str]) -> tuple[Bridge, int | None]:
    hooked_pid = _find_listening_pid(CONTROL_PORT)
    if hooked_pid is not None and not _find_desktop_hwnds(hooked_pid):
        log(f"Closing hidden/off-desktop PhotoCraft PID {hooked_pid}")
        _kill_pid(hooked_pid)
        time.sleep(0.4)
        hooked_pid = None

    for pid in _list_photocraft_pids():
        if pid != hooked_pid:
            log(f"Closing disconnected PhotoCraft instance PID {pid}")
            _kill_pid(pid)

    if hooked_pid is None:
        _clean_recent_files_in_prefs()
        cmd = [
            APP_EXE,
            "--control",
            str(CONTROL_PORT),
            "--control-token-file",
            TOKEN_FILE,
            "--automation-read-root",
            READ_ROOT,
            "--automation-write-root",
            SCRATCH_DIR,
        ]
        if extra_files:
            cmd.append("--")
            cmd.extend(extra_files)
        kwargs: dict = {"cwd": PORTABLE_DIR}
        if os.name == "nt":
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = 1
            kwargs["startupinfo"] = si
        subprocess.Popen(cmd, **kwargs)
        log(f"Launched PhotoCraft with control port {CONTROL_PORT}.")

    bridge = Bridge()
    connected = False
    for _ in range(60):
        if os.path.exists(TOKEN_FILE):
            try:
                bridge.call("ui.inspect", {})
                connected = True
                break
            except Exception:
                pass
        time.sleep(0.4)

    if not connected:
        return bridge, None

    if hooked_pid is not None and extra_files:
        for fpath in extra_files:
            if os.path.exists(fpath):
                try:
                    bridge.call("app.open", {"path": os.path.abspath(fpath)})
                except Exception:
                    pass

    hooked_pid = _find_listening_pid(CONTROL_PORT)
    try:
        bridge.call("ui.focus", {})
    except Exception:
        pass
    if hooked_pid:
        _bring_pid_to_front(hooked_pid)

    return bridge, hooked_pid


def main() -> None:
    args = sys.argv[1:]
    ensure_gui = False
    extra_files: list[str] = []
    for a in args:
        if a == "--ensure-gui":
            ensure_gui = True
        elif not a.startswith("-"):
            extra_files.append(a)

    if ensure_gui:
        bridge, hooked_pid = ensure_single_hooked_photocraft(extra_files)
        if hooked_pid is None:
            log("PhotoCraft control port did not open during --ensure-gui; exiting.")
            sys.exit(1)
    else:
        bridge = Bridge()

    lock_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        lock_sock.bind(("127.0.0.1", HOOK_LOCK_PORT))
        lock_sock.listen(1)
    except OSError:
        bridge.close()
        sys.exit(0)

    log("PhotoCraft Invisible AI Auto-Hook active (single-instance lock acquired).")

    connected = False
    for _ in range(50):
        if os.path.exists(TOKEN_FILE):
            try:
                bridge.call("ui.inspect", {})
                connected = True
                break
            except Exception:
                pass
        time.sleep(0.4)

    if not connected:
        log("PhotoCraft control port not reachable; exiting.")
        bridge.close()
        lock_sock.close()
        sys.exit(0)

    models = LocalAiModels()
    try:
        models.warm_models()
        log("Pre-warmed YOLOX-Nano + MODNet ONNX models in memory.")
    except Exception as e:
        log(f"Model warm-up deferred: {e}")

    seen_rev: dict[str, int] = {}
    missed_pings = 0

    try:
        while True:
            try:
                bridge.call("ui.inspect", {})
                missed_pings = 0
            except Exception:
                missed_pings += 1
                if missed_pings >= 6:
                    log("PhotoCraft closed; AI Auto-Hook shutting down cleanly.")
                    _clean_recent_files_in_prefs()
                    break
                time.sleep(0.5)
                continue

            try:
                doc = bridge.cmd("document.inspect", {})
                doc_key = f"{doc.get('name', '')}:{doc.get('width', 0)}x{doc.get('height', 0)}"
                rev = int(doc.get("revision", 0))
                hist = doc.get("history") or []

                prev = seen_rev.get(doc_key)
                if prev is None:
                    seen_rev[doc_key] = rev
                elif rev != prev:
                    last_step = hist[-1] if hist else ""
                    if last_step in ("Select Subject", "Remove Background", "Object Selection"):
                        new_rev = intercept_once(bridge, models, last_step, doc)
                        seen_rev[doc_key] = new_rev
                    else:
                        seen_rev[doc_key] = rev
            except Exception:
                pass

            time.sleep(0.30)
    finally:
        bridge.close()
        lock_sock.close()


if __name__ == "__main__":
    main()
