"""
experts/fire_expert.py
======================
Expertos de detección de FUEGO y HUMO.

Ambos comparten el mismo modelo YOLO de detección (fire_model) y la misma
función HSV+motion+YOLO (_process_fire_smoke). Para evitar ejecutar la
detección YOLO dos veces por frame se usa un detector compartido
(_FireSmokeDetector) con caché por índice de frame.

Uso en main.py:
    from experts.fire_expert import make_fire_smoke_experts
    fire_expert, smoke_expert = make_fire_smoke_experts()
    experts = [..., fire_expert, smoke_expert]
"""

import queue
import threading
from collections import deque
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
import clip
from ultralytics import YOLO

from core.base_expert import BaseExpert

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
clip_device = "cuda" if torch.cuda.is_available() else "cpu"
VIEW_SIZE   = (640, 480)

# ---------------------------------------------------------------------------
# Constantes FIRE / SMOKE — YOLO raw
# ---------------------------------------------------------------------------
FIRE_HISTORY_LEN   = 18
FIRE_MIN_HITS      = 2
FIRE_PRE_CROP_CONF = 0.10
FIRE_ZOOM_CONF     = 0.20
FIRE_CROP_PAD      = 200
FIRE_MASK_MIN_AREA = 100   # V8: subido de 60
FIRE_MOTION_THR    = 14
FIRE_MIN_BBOX_AREA = 3000  # V8: nuevo — descarta bboxes YOLO pequeños (w*h px²)
SMOKE_HISTORY_LEN  = 30
SMOKE_MIN_HITS     = 8
SMOKE_CONF_THR     = 0.08

# ---------------------------------------------------------------------------
# Constantes CLIP FIRE
# ---------------------------------------------------------------------------
FIRE_CLIP_BUFFER_SIZE  = 20
FIRE_CLIP_STRIDE       = 2
FIRE_CLIP_DIFF_TH      = 3.0   # V8: subido de 2.0
FIRE_CLIP_MIN_HITS     = 5     # V8: subido de 3
FIRE_CLIP_SET_SCORE_TH = 3.5   # V8: subido de 2.5
FIRE_CLIP_TOPK         = 4
FIRE_CONSEC_MIN        = 3     # V8: subido de 2
FIRE_TTL               = 90

# ---------------------------------------------------------------------------
# Constantes CLIP SMOKE
# ---------------------------------------------------------------------------
SMOKE_CLIP_BUFFER_SIZE  = 20
SMOKE_CLIP_STRIDE       = 2
SMOKE_CLIP_DIFF_TH      = 1.5
SMOKE_CLIP_MIN_HITS     = 3
SMOKE_CLIP_SET_SCORE_TH = 2.0
SMOKE_CLIP_TOPK         = 4
SMOKE_CONSEC_MIN        = 2
SMOKE_TTL               = 120

# ---------------------------------------------------------------------------
# Prompts CLIP FIRE
# ---------------------------------------------------------------------------
FIRE_POS_PROMPTS = [
    "a building on fire with visible flames",
    "a vehicle on fire with flames coming out",
    "a fire burning on the street outdoors",
    "bright orange flames burning intensely",
    "flames coming from a window in a building",
    "a trash fire with visible flames and smoke",
    "a small fire on the ground with flames",
    "a fire with flames and thick smoke rising",
    "an active fire with visible flames in an urban scene",
    "a wildfire flame front burning vegetation",
]
FIRE_NEG_PROMPTS = [
    "car headlights at night on a road",
    "traffic lights and city lights at night",
    "sunset or sunrise lighting on buildings",
    "reflections of orange light on glass or metal",
    "street lamps and illuminated signs at night",
    "a normal street scene with cars and no fire",
    "a bright lamp or spotlight with no fire",
    "a construction site light with no flames",
    "a bonfire image on a screen or billboard, not real fire",
    "an indoor warm light scene with no flames",
]

# ---------------------------------------------------------------------------
# Prompts CLIP SMOKE
# ---------------------------------------------------------------------------
SMOKE_POS_PROMPTS = [
    "thick smoke rising from a fire outdoors",
    "black smoke coming from a burning vehicle",
    "gray smoke plume rising into the sky",
    "smoke billowing from a building",
    "dense smoke cloud over a street",
    "smoke rising from debris after a fire",
    "smoke coming out of a window",
    "a street scene with visible smoke haze from a fire",
    "a smoke plume from an accident or fire scene",
    "heavy smoke with low visibility",
]
SMOKE_NEG_PROMPTS = [
    "foggy street scene with fog not smoke",
    "clouds in the sky on a normal day",
    "steam rising from a manhole or vent",
    "dust cloud from construction with no fire",
    "car exhaust smoke from a tailpipe",
    "a normal street scene with no smoke",
    "haze from sunlight or camera exposure",
    "blur or compression artifacts in a video",
    "mist near water or a fountain",
    "smoke-like lighting glare but no smoke",
]


# ===========================================================================
# Detector compartido YOLO (instanciado una sola vez, compartido por
# FireExpert y SmokeExpert)
# ===========================================================================
class _FireSmokeDetector:
    """
    Encapsula el modelo YOLO de fuego/humo y la lógica HSV+motion+YOLO.
    El resultado se cachea por frame_idx para no ejecutarlo dos veces.
    """

    def __init__(self, model_path: str):
        self._fire_model        = YOLO(model_path)
        self._fire_prev_gray    = None
        self._last_raw_clip_crop = None
        self._last_fire_bboxes: list = []
        self.fire_history  = deque(maxlen=FIRE_HISTORY_LEN)
        self.smoke_history = deque(maxlen=SMOKE_HISTORY_LEN)

        # Caché por frame
        self._cached_frame_idx: int   = -1
        self._cached_result: Optional[dict] = None

    def process(self, frame: np.ndarray, frame_idx: int) -> dict:
        """
        Ejecuta la detección YOLO+HSV para el frame dado.
        Si ya fue procesado este frame_idx devuelve el resultado en caché.

        Retorna:
            {
                "fire_raw":      bool,
                "smoke_raw":     bool,
                "fire_bboxes":   list[(x,y,w,h)],
                "raw_clip_crop": np.ndarray | None,
                "hits_f":        int,
                "hits_s":        int,
            }
        """
        if frame_idx == self._cached_frame_idx and self._cached_result is not None:
            return self._cached_result

        H, W = frame.shape[:2]

        # HSV mask (tonos rojos/naranja)
        hsv     = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        m1      = cv2.inRange(hsv, np.array([0,   60, 120]), np.array([40,  255, 255]))
        m2      = cv2.inRange(hsv, np.array([170, 60, 120]), np.array([179, 255, 255]))
        m_color = cv2.morphologyEx(
            cv2.bitwise_or(m1, m2), cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1
        )

        # Motion mask
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._fire_prev_gray is not None:
            _, md      = cv2.threshold(cv2.absdiff(gray, self._fire_prev_gray),
                                       FIRE_MOTION_THR, 255, cv2.THRESH_BINARY)
            m_final    = cv2.bitwise_and(m_color, md)
        else:
            m_final = m_color
        self._fire_prev_gray = gray

        # Contornos
        cnts, _ = cv2.findContours(m_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for c in cnts:
            if cv2.contourArea(c) > FIRE_MASK_MIN_AREA:
                x, y, w, h = cv2.boundingRect(c)
                candidates.append((
                    (x, y, w, h),
                    (max(0, x-FIRE_CROP_PAD), max(0, y-FIRE_CROP_PAD),
                     min(W, x+w+FIRE_CROP_PAD), min(H, y+h+FIRE_CROP_PAD))
                ))

        f_active, s_active  = False, False
        raw_clip_crop       = None
        confirmed_bboxes    = []

        for tight, cb in candidates[:5]:
            patch = frame[cb[1]:cb[3], cb[0]:cb[2]]
            if patch.size == 0:
                continue
            results = self._fire_model.predict(patch, conf=FIRE_PRE_CROP_CONF, verbose=False)[0]
            for rb in results.boxes:
                cls, conf = int(rb.cls[0]), float(rb.conf[0])
                if cls == 0 and conf >= FIRE_ZOOM_CONF:
                    tx, ty, tw, th = tight
                    if tw * th < FIRE_MIN_BBOX_AREA:
                        continue   # V8: descarta bboxes pequeños
                    f_active = True
                    confirmed_bboxes.append(tight)
                elif cls == 1 and conf >= SMOKE_CONF_THR:
                    s_active = True
            if f_active or s_active:
                raw_clip_crop            = cv2.resize(patch, VIEW_SIZE)
                self._last_raw_clip_crop = raw_clip_crop

        self.fire_history.append(f_active)
        self.smoke_history.append(s_active)
        hits_f = sum(self.fire_history)
        hits_s = sum(self.smoke_history)

        if confirmed_bboxes:
            self._last_fire_bboxes = confirmed_bboxes
        if hits_f == 0 and hits_s == 0:
            self._last_fire_bboxes = []

        result = {
            "fire_raw":      hits_f >= FIRE_MIN_HITS,
            "smoke_raw":     hits_s >= SMOKE_MIN_HITS,
            "fire_bboxes":   self._last_fire_bboxes,
            "raw_clip_crop": raw_clip_crop,
            "hits_f":        hits_f,
            "hits_s":        hits_s,
        }
        self._cached_frame_idx = frame_idx
        self._cached_result    = result
        return result


# ===========================================================================
# FireExpert
# ===========================================================================
class FireExpert(BaseExpert):
    """
    Detecta FUEGO usando el canal fire del detector compartido + CLIP.
    """

    def __init__(self, shared_detector: _FireSmokeDetector):
        self._detector = shared_detector

        # Estado CLIP FIRE
        self.fire_clip_buffer: list          = []
        self.fire_clip_confirmed: bool       = False
        self.fire_clip_consec_pos: int       = 0
        self.fire_clip_ttl: int              = 0
        self._fire_clip_processing: bool     = False

        # CLIP
        self._model      = None
        self._preprocess = None
        self._pos_text   = None
        self._neg_text   = None

        # Queue
        self._q: queue.Queue = queue.Queue(maxsize=2)

    # ------------------------------------------------------------------
    @property
    def label(self) -> str:
        return "FUEGO"

    @property
    def is_active(self) -> bool:
        return self.fire_clip_confirmed

    # ------------------------------------------------------------------
    def load(self, model, preprocess) -> None:
        self._model      = model
        self._preprocess = preprocess

        with torch.no_grad():
            pfeat = model.encode_text(clip.tokenize(FIRE_POS_PROMPTS).to(clip_device))
            nfeat = model.encode_text(clip.tokenize(FIRE_NEG_PROMPTS).to(clip_device))
            pfeat /= pfeat.norm(dim=-1, keepdim=True)
            nfeat /= nfeat.norm(dim=-1, keepdim=True)
            self._pos_text = pfeat.mean(0, keepdim=True)
            self._neg_text = nfeat.mean(0, keepdim=True)
            self._pos_text /= self._pos_text.norm(dim=-1, keepdim=True)
            self._neg_text /= self._neg_text.norm(dim=-1, keepdim=True)

        threading.Thread(target=self._worker_loop, daemon=True).start()

    # ------------------------------------------------------------------
    def process_heuristics(self, frame: np.ndarray, tracker_data: Dict[str, Any]) -> None:
        det = self._detector.process(frame, tracker_data["frame_idx"])

        hits_f       = det["hits_f"]
        hits_s       = det["hits_s"]
        raw_clip     = det["raw_clip_crop"]
        suspicion    = hits_f > 0 or hits_s > 0

        if suspicion:
            crop = raw_clip if raw_clip is not None else self._detector._last_raw_clip_crop
            if crop is not None:
                self.fire_clip_buffer.append(crop)
        else:
            self.fire_clip_buffer = []

        if len(self.fire_clip_buffer) >= FIRE_CLIP_BUFFER_SIZE:
            if not self._fire_clip_processing:
                try:
                    self._q.put_nowait(list(self.fire_clip_buffer))
                    self._fire_clip_processing = True
                except queue.Full:
                    pass
            self.fire_clip_buffer = []

        # TTL
        if self.fire_clip_confirmed:
            self.fire_clip_ttl -= 1
            if self.fire_clip_ttl <= 0:
                self.fire_clip_confirmed  = False
                self.fire_clip_consec_pos = 0

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, frames: List[np.ndarray]) -> dict:
        sampled     = frames[::FIRE_CLIP_STRIDE]
        diffs, hits = [], 0
        scale       = float(self._model.logit_scale.exp().item())
        for fr in sampled:
            if fr is None or fr.size == 0:
                continue
            pil  = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            img  = self._preprocess(pil).unsqueeze(0).to(clip_device)
            feat = self._model.encode_image(img)
            feat /= feat.norm(dim=-1, keepdim=True)
            lp   = float(scale * (feat @ self._pos_text.T).item())
            ln   = float(scale * (feat @ self._neg_text.T).item())
            diff = lp - ln
            diffs.append(diff)
            if diff > FIRE_CLIP_DIFF_TH:
                hits += 1
        if not diffs:
            return {"detected": False, "score": 0.0, "set_score": 0.0, "hits": 0}
        da        = np.array(diffs, dtype=np.float32)
        topk      = min(FIRE_CLIP_TOPK, len(da))
        set_score = float(np.sort(da)[-topk:].mean())
        detected  = (hits >= FIRE_CLIP_MIN_HITS) and (set_score > FIRE_CLIP_SET_SCORE_TH)
        return {"detected": detected, "score": set_score, "set_score": set_score, "hits": hits}

    # ------------------------------------------------------------------
    def get_display_data(self) -> Dict[str, Any]:
        bboxes = self._detector._last_fire_bboxes if self.fire_clip_confirmed else []
        return {"bboxes": bboxes, "color": (0, 69, 255)}

    # ------------------------------------------------------------------
    def _worker_loop(self) -> None:
        while True:
            frames = self._q.get()
            try:
                res = self.predict(frames)
                if res["detected"]:
                    self.fire_clip_consec_pos += 1
                else:
                    self.fire_clip_consec_pos = 0
                if self.fire_clip_consec_pos >= FIRE_CONSEC_MIN:
                    self.fire_clip_confirmed = True
                    self.fire_clip_ttl       = FIRE_TTL
            except Exception:
                pass
            self._fire_clip_processing = False


# ===========================================================================
# SmokeExpert
# ===========================================================================
class SmokeExpert(BaseExpert):
    """
    Detecta HUMO usando el canal smoke del detector compartido + CLIP.
    """

    def __init__(self, shared_detector: _FireSmokeDetector):
        self._detector = shared_detector

        # Estado CLIP SMOKE
        self.smoke_clip_buffer: list          = []
        self.smoke_clip_confirmed: bool       = False
        self.smoke_clip_consec_pos: int       = 0
        self.smoke_clip_ttl: int              = 0
        self._smoke_clip_processing: bool     = False

        # CLIP
        self._model      = None
        self._preprocess = None
        self._pos_text   = None
        self._neg_text   = None

        # Queue
        self._q: queue.Queue = queue.Queue(maxsize=2)

    # ------------------------------------------------------------------
    @property
    def label(self) -> str:
        return "HUMO"

    @property
    def is_active(self) -> bool:
        return self.smoke_clip_confirmed

    # ------------------------------------------------------------------
    def load(self, model, preprocess) -> None:
        self._model      = model
        self._preprocess = preprocess

        with torch.no_grad():
            pfeat = model.encode_text(clip.tokenize(SMOKE_POS_PROMPTS).to(clip_device))
            nfeat = model.encode_text(clip.tokenize(SMOKE_NEG_PROMPTS).to(clip_device))
            pfeat /= pfeat.norm(dim=-1, keepdim=True)
            nfeat /= nfeat.norm(dim=-1, keepdim=True)
            self._pos_text = pfeat.mean(0, keepdim=True)
            self._neg_text = nfeat.mean(0, keepdim=True)
            self._pos_text /= self._pos_text.norm(dim=-1, keepdim=True)
            self._neg_text /= self._neg_text.norm(dim=-1, keepdim=True)

        threading.Thread(target=self._worker_loop, daemon=True).start()

    # ------------------------------------------------------------------
    def process_heuristics(self, frame: np.ndarray, tracker_data: Dict[str, Any]) -> None:
        det      = self._detector.process(frame, tracker_data["frame_idx"])
        hits_f   = det["hits_f"]
        hits_s   = det["hits_s"]
        raw_clip = det["raw_clip_crop"]
        suspicion = hits_f > 0 or hits_s > 0

        if suspicion:
            crop = raw_clip if raw_clip is not None else self._detector._last_raw_clip_crop
            if crop is not None:
                self.smoke_clip_buffer.append(crop)
        else:
            self.smoke_clip_buffer = []

        if len(self.smoke_clip_buffer) >= SMOKE_CLIP_BUFFER_SIZE:
            if not self._smoke_clip_processing:
                try:
                    self._q.put_nowait(list(self.smoke_clip_buffer))
                    self._smoke_clip_processing = True
                except queue.Full:
                    pass
            self.smoke_clip_buffer = []

        # TTL
        if self.smoke_clip_confirmed:
            self.smoke_clip_ttl -= 1
            if self.smoke_clip_ttl <= 0:
                self.smoke_clip_confirmed  = False
                self.smoke_clip_consec_pos = 0

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, frames: List[np.ndarray]) -> dict:
        sampled     = frames[::SMOKE_CLIP_STRIDE]
        diffs, hits = [], 0
        scale       = float(self._model.logit_scale.exp().item())
        for fr in sampled:
            if fr is None or fr.size == 0:
                continue
            pil  = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            img  = self._preprocess(pil).unsqueeze(0).to(clip_device)
            feat = self._model.encode_image(img)
            feat /= feat.norm(dim=-1, keepdim=True)
            lp   = float(scale * (feat @ self._pos_text.T).item())
            ln   = float(scale * (feat @ self._neg_text.T).item())
            diff = lp - ln
            diffs.append(diff)
            if diff > SMOKE_CLIP_DIFF_TH:
                hits += 1
        if not diffs:
            return {"detected": False, "score": 0.0, "set_score": 0.0, "hits": 0}
        da        = np.array(diffs, dtype=np.float32)
        topk      = min(SMOKE_CLIP_TOPK, len(da))
        set_score = float(np.sort(da)[-topk:].mean())
        detected  = (hits >= SMOKE_CLIP_MIN_HITS) and (set_score > SMOKE_CLIP_SET_SCORE_TH)
        return {"detected": detected, "score": set_score, "set_score": set_score, "hits": hits}

    # ------------------------------------------------------------------
    def _worker_loop(self) -> None:
        while True:
            frames = self._q.get()
            try:
                res = self.predict(frames)
                if res["detected"]:
                    self.smoke_clip_consec_pos += 1
                else:
                    self.smoke_clip_consec_pos = 0
                if self.smoke_clip_consec_pos >= SMOKE_CONSEC_MIN:
                    self.smoke_clip_confirmed = True
                    self.smoke_clip_ttl       = SMOKE_TTL
            except Exception:
                pass
            self._smoke_clip_processing = False


# ===========================================================================
# Factory pública
# ===========================================================================
def make_fire_smoke_experts(
    fire_model_path: str = ".checkpoints/best_large.pt",
) -> Tuple["FireExpert", "SmokeExpert"]:
    """
    Crea FireExpert y SmokeExpert compartiendo el mismo modelo YOLO.
    Úsala siempre en lugar de instanciar los expertos directamente.

    Ejemplo:
        fire_expert, smoke_expert = make_fire_smoke_experts()
        experts = [FightExpert(), CrashExpert(), fire_expert, smoke_expert]
    """
    shared = _FireSmokeDetector(fire_model_path)
    return FireExpert(shared), SmokeExpert(shared)
