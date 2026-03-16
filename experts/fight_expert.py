"""
experts/fight_expert.py
=======================
Experto de detección de PELEAS.
Lógica idéntica a ArconteV3Prod._process_fight + CLIPFightExpert.
"""

import queue
import threading
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
from PIL import Image
import torch
import clip

from core.base_expert import BaseExpert

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
clip_device = "cuda" if torch.cuda.is_available() else "cpu"
VIEW_SIZE   = (640, 480)

# ---------------------------------------------------------------------------
# Constantes FIGHT (Dev2 — IoU/distance + smoothed CLIP)
# ---------------------------------------------------------------------------
F_MIN_IOU_CRITICAL = 0.02
F_DIST_MAX_LIMIT   = 250
F_STICKY_FRAMES    = 15
F_PAD_PIXELS       = 120
F_BUFFER_SIZE      = 20
F_DIFF_HIT_TH      = 4.2
F_SET_SCORE_TH     = 4.8
F_MIN_POS_FRAMES   = 5
F_CONSEC_MIN       = 2
F_SMOOTH_FACTOR    = 0.7


class FightExpert(BaseExpert):
    """
    Detecta peleas callejeras usando:
      1. Heurística espacial: par de personas con IoU > F_MIN_IOU_CRITICAL
         o distancia < F_DIST_MAX_LIMIT (sticky crop de F_STICKY_FRAMES frames).
      2. Buffer de F_BUFFER_SIZE crops enviado al worker CLIP.
      3. Confirmación: hits >= F_MIN_POS_FRAMES Y smoothed_score > F_SET_SCORE_TH
         durante F_CONSEC_MIN buffers consecutivos.
    """

    # ------------------------------------------------------------------
    # Inicialización
    # ------------------------------------------------------------------
    def __init__(self):
        # Estado FIGHT
        self.fight_last_valid_crop_bbox = None
        self.fight_sticky_counter       = 0
        self.fight_last_crop            = None
        self.fight_buffer: list         = []
        self.fight_is_detected: bool    = False
        self.fight_consec_pos: float    = 0
        self._fight_processing: bool    = False

        # CLIP
        self._model        = None
        self._preprocess   = None
        self._pos_text     = None
        self._neg_text     = None
        self._smoothed_score: float = 0.0

        # Queue del worker
        self._q: queue.Queue = queue.Queue(maxsize=2)

    # ------------------------------------------------------------------
    # BaseExpert — propiedades obligatorias
    # ------------------------------------------------------------------
    @property
    def label(self) -> str:
        return "PELEA"

    @property
    def is_active(self) -> bool:
        return self.fight_is_detected

    # ------------------------------------------------------------------
    # BaseExpert — load
    # ------------------------------------------------------------------
    def load(self, model, preprocess) -> None:
        """Carga CLIP, codifica prompts y lanza el worker thread."""
        self._model      = model
        self._preprocess = preprocess

        with torch.no_grad():
            pos = ["violent physical fight, punching and kicking", "real street brawling"]
            neg = ["people walking normally", "pedestrians standing", "peaceful street"]
            pf = model.encode_text(clip.tokenize(pos).to(clip_device))
            nf = model.encode_text(clip.tokenize(neg).to(clip_device))
            self._pos_text = (pf / pf.norm(dim=-1, keepdim=True)).mean(0, keepdim=True)
            self._neg_text = (nf / nf.norm(dim=-1, keepdim=True)).mean(0, keepdim=True)

        threading.Thread(target=self._worker_loop, daemon=True).start()

    # ------------------------------------------------------------------
    # BaseExpert — process_heuristics
    # ------------------------------------------------------------------
    def process_heuristics(self, frame: np.ndarray, tracker_data: Dict[str, Any]) -> None:
        self._process_fight(frame, tracker_data["persons_xyxy"], tracker_data["persons_ids"])

    # ------------------------------------------------------------------
    # BaseExpert — predict
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, frames: List[np.ndarray]) -> dict:
        diffs, hits = [], 0
        scale = float(self._model.logit_scale.exp().item())
        for fr in frames:
            if fr is None or fr.size == 0:
                continue
            img  = self._preprocess(
                Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            ).unsqueeze(0).to(clip_device)
            feat = self._model.encode_image(img)
            feat /= feat.norm(dim=-1, keepdim=True)
            diff = float(scale * (feat @ self._pos_text.T - feat @ self._neg_text.T).item())
            diffs.append(diff)
            if diff > F_DIFF_HIT_TH:
                hits += 1
        if not diffs:
            return {"detected": False, "score": 0.0, "hits": 0, "raw": 0.0}
        raw_score = float(np.sort(diffs)[-5:].mean()) if len(diffs) >= 5 else 0.0
        self._smoothed_score = (
            F_SMOOTH_FACTOR * raw_score + (1 - F_SMOOTH_FACTOR) * self._smoothed_score
        )
        detected = (hits >= F_MIN_POS_FRAMES) and (self._smoothed_score > F_SET_SCORE_TH)
        return {"detected": detected, "score": self._smoothed_score, "hits": hits, "raw": raw_score}

    # ------------------------------------------------------------------
    # BaseExpert — get_display_data
    # ------------------------------------------------------------------
    def get_display_data(self) -> Dict[str, Any]:
        return {"last_crop": self.fight_last_crop} if self.fight_last_crop is not None else {}

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------
    def _worker_loop(self) -> None:
        while True:
            frames = self._q.get()
            try:
                res = self.predict(frames)
                if res["detected"]:
                    self.fight_consec_pos = min(6, self.fight_consec_pos + 1)
                else:
                    self.fight_consec_pos = max(0, self.fight_consec_pos - 1)
                self.fight_is_detected = (self.fight_consec_pos >= F_CONSEC_MIN)
            except Exception:
                pass
            self._fight_processing = False

    # ------------------------------------------------------------------
    # Helpers (extraídos de ArconteV3Prod sin modificar)
    # ------------------------------------------------------------------
    @staticmethod
    def _fight_get_iou(b1, b2) -> float:
        xA, yA = max(b1[0], b2[0]), max(b1[1], b2[1])
        xB, yB = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter  = max(0, xB - xA) * max(0, yB - yA)
        return inter / float(
            (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter + 1e-6
        )

    @staticmethod
    def _crop_from_box(frame: np.ndarray, box, pad: int) -> Optional[np.ndarray]:
        H, W = frame.shape[:2]
        x1 = int(max(0, box[0] - pad))
        y1 = int(max(0, box[1] - pad))
        x2 = int(min(W, box[2] + pad))
        y2 = int(min(H, box[3] + pad))
        if x2 <= x1 or y2 <= y1:
            return None
        c = frame[y1:y2, x1:x2]
        return cv2.resize(c, VIEW_SIZE) if c is not None and c.size > 0 else None

    # ------------------------------------------------------------------
    # Pipeline principal (idéntico a ArconteV3Prod._process_fight)
    # ------------------------------------------------------------------
    def _process_fight(self, frame: np.ndarray, person_xyxy, person_ids) -> None:
        n = len(person_xyxy)

        # Buscar el mejor par por IoU + distancia
        best_pair = None
        max_iou   = 0
        for i in range(n):
            for j in range(i + 1, n):
                b1, b2 = person_xyxy[i], person_xyxy[j]
                iou  = self._fight_get_iou(b1, b2)
                dist = float(np.linalg.norm(b1[:2] - b2[:2]))
                if iou > F_MIN_IOU_CRITICAL or dist < F_DIST_MAX_LIMIT:
                    if iou >= max_iou:
                        max_iou   = iou
                        best_pair = [
                            min(b1[0], b2[0]), min(b1[1], b2[1]),
                            max(b1[2], b2[2]), max(b1[3], b2[3]),
                        ]

        if best_pair:
            self.fight_last_valid_crop_bbox = best_pair
            self.fight_sticky_counter       = F_STICKY_FRAMES
        else:
            self.fight_sticky_counter = max(0, self.fight_sticky_counter - 1)

        # Crop y buffer
        if self.fight_last_valid_crop_bbox is not None and self.fight_sticky_counter > 0:
            p_crop = self._crop_from_box(
                frame, np.array(self.fight_last_valid_crop_bbox), F_PAD_PIXELS
            )
            if p_crop is not None:
                self.fight_last_crop = p_crop
                self.fight_buffer.append(p_crop)
                if len(self.fight_buffer) >= F_BUFFER_SIZE:
                    if not self._fight_processing:
                        try:
                            self._q.put_nowait(list(self.fight_buffer))
                            self._fight_processing = True
                        except queue.Full:
                            pass
                    self.fight_buffer = []
        else:
            self.fight_buffer      = []
            self.fight_consec_pos  = max(0, self.fight_consec_pos - 0.2)
            self.fight_is_detected = (self.fight_consec_pos >= F_CONSEC_MIN)
