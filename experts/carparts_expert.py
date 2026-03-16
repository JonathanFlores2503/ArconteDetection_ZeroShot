"""
experts/carparts_expert.py
==========================
Experto de detección de ROBO DE AUTOPARTES.
Lógica idéntica a CarPartsExpert_Debug_V3.py encapsulada en BaseExpert.

Pipeline:
  1. Heurística espacial: personas dentro del perímetro de un vehículo
     (dist < CP_PERIMETRO_PX). Puntuación por proximidad + movimiento.
  2. Target-lock: fija el foco en el ladrón activo cuando score >= CP_LOCK_SCORE_TH.
  3. Crop de grupo: si hay cómplice a <= CP_GRUPO_DIST_PX, bbox expandido.
  4. Buffer corto (6 frames) → CLIP worker.
  5. Fast-track: si un solo frame supera CP_FAST_TRACK_TH → alerta inmediata.
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
# Constantes
# ---------------------------------------------------------------------------
CP_PERIMETRO_PX    = 120   # distancia borde vehículo para considerar interacción
CP_STICKY_FRAMES   = 20    # frames que se mantiene el crop tras perder la interacción
CP_PAD_SOLO        = 15    # padding cuando hay 1 ladrón → zoom máximo en manos/faro
CP_PAD_GRUPO       = 40    # padding cuando hay 2 personas (cómplice)
CP_GRUPO_DIST_PX   = 200   # distancia px entre personas para crop de grupo
CP_LOCK_SCORE_TH   = 3.5   # score CLIP que activa target-lock

CP_BUFFER_SIZE     = 6     # buffer corto = respuesta en tiempo real
CP_DIFF_HIT_TH     = 2.6   # umbral hit por frame
CP_ALERT_THRESHOLD = 3.0   # umbral score suavizado para declarar detección
CP_HIT_FRAMES_REQ  = 2     # basta con 2 frames sospechosos en el buffer para alertar
CP_CONSEC_MIN      = 2     # batches consecutivos positivos para confirmar alerta
CP_SMOOTH_FACTOR   = 0.7   # EMA del score
CP_FAST_TRACK_TH   = 4.0   # diff por frame → retorno inmediato sin esperar buffer


class CarPartsExpert(BaseExpert):
    """
    Detecta robo de autopartes (faros, retrovisores, etc.) usando:
      - Interacción persona-vehículo por proximidad y movimiento.
      - CLIP Zero-Shot con buffer corto de 6 frames.
      - Fast-track para alerta inmediata sin esperar buffer completo.
    """

    # ------------------------------------------------------------------
    # Inicialización
    # ------------------------------------------------------------------
    def __init__(self):
        # Estado del crop
        self.cp_buffer: list             = []
        self.cp_is_detected: bool        = False
        self.cp_consec_pos: float        = 0.0
        self.cp_last_crop                = None
        self.cp_sticky_counter: int      = 0
        self.cp_last_valid_bbox          = None
        self.cp_last_score: float        = 0.0
        self.cp_last_raw: float          = 0.0
        self.cp_last_hits: int           = 0
        self.cp_clips_analyzed: int      = 0
        self._cp_was_confirmed: bool     = False
        self._cp_processing: bool        = False
        self.cp_last_p_id                = None
        self.cp_last_v_id                = None

        # Selección de objetivo inteligente
        self.cp_prev_boxes: dict         = {}
        self.cp_target_locked: bool      = False
        self.cp_locked_p_id              = None
        self.cp_crop_mode: str           = "solo"

        # CLIP
        self._model        = None
        self._preprocess   = None
        self._pos_text     = None
        self._neg_text     = None
        self._smoothed_score: float = 0.0

        # Queue
        self._q: queue.Queue = queue.Queue(maxsize=2)

    # ------------------------------------------------------------------
    # BaseExpert — propiedades
    # ------------------------------------------------------------------
    @property
    def label(self) -> str:
        return "ROBO"

    @property
    def is_active(self) -> bool:
        return self.cp_is_detected

    # ------------------------------------------------------------------
    # BaseExpert — load
    # ------------------------------------------------------------------
    def load(self, model, preprocess) -> None:
        self._model      = model
        self._preprocess = preprocess

        with torch.no_grad():
            pos = [
                "a person prying out car headlights with tools",
                "hands pulling off a vehicle headlight",
                "someone dismantling the front lights of a car",
                "a thief stealing headlights from a vehicle",
            ]
            neg = [
                "a person walking past a car",
                "owner opening a car door",
                "a parked car with lights intact",
                "street background",
            ]
            pf = model.encode_text(clip.tokenize(pos).to(clip_device))
            nf = model.encode_text(clip.tokenize(neg).to(clip_device))
            self._pos_text = (pf / pf.norm(dim=-1, keepdim=True)).mean(0, keepdim=True)
            self._neg_text = (nf / nf.norm(dim=-1, keepdim=True)).mean(0, keepdim=True)

        threading.Thread(target=self._worker_loop, daemon=True).start()

    # ------------------------------------------------------------------
    # BaseExpert — process_heuristics
    # ------------------------------------------------------------------
    def process_heuristics(self, frame: np.ndarray, tracker_data: Dict[str, Any]) -> None:
        # Usa vehicles_all_xyxy (incluye motos clase 3) si está disponible
        vehicle_xyxy = tracker_data.get("vehicles_all_xyxy", tracker_data["vehicles_xyxy"])
        vehicle_ids  = tracker_data.get("vehicles_all_ids",  tracker_data["vehicles_ids"])

        self._process_carparts(
            frame,
            tracker_data["persons_xyxy"],
            tracker_data["persons_ids"],
            vehicle_xyxy,
            vehicle_ids,
        )

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
            pil  = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            img  = self._preprocess(pil).unsqueeze(0).to(clip_device)
            feat = self._model.encode_image(img)
            feat /= feat.norm(dim=-1, keepdim=True)
            diff = float(scale * (feat @ self._pos_text.T - feat @ self._neg_text.T).item())
            diffs.append(diff)

            # Fast-track: retorno inmediato sin esperar al resto del buffer
            if diff >= CP_FAST_TRACK_TH:
                self._smoothed_score = diff
                return {
                    "detected":   True,
                    "score":      self._smoothed_score,
                    "hits":       1,
                    "raw":        diff,
                    "fast_track": True,
                }

            if diff > CP_DIFF_HIT_TH:
                hits += 1

        if not diffs:
            return {"detected": False, "score": 0.0, "hits": 0, "raw": 0.0, "fast_track": False}

        raw_score = (
            float(np.sort(diffs)[-3:].mean()) if len(diffs) >= 3 else float(np.mean(diffs))
        )
        self._smoothed_score = (
            CP_SMOOTH_FACTOR * raw_score + (1 - CP_SMOOTH_FACTOR) * self._smoothed_score
        )
        detected = (hits >= CP_HIT_FRAMES_REQ) and (self._smoothed_score > CP_ALERT_THRESHOLD)
        return {
            "detected":   detected,
            "score":      self._smoothed_score,
            "hits":       hits,
            "raw":        raw_score,
            "fast_track": False,
        }

    # ------------------------------------------------------------------
    # BaseExpert — get_display_data
    # ------------------------------------------------------------------
    def get_display_data(self) -> Dict[str, Any]:
        return {"last_crop": self.cp_last_crop} if self.cp_last_crop is not None else {}

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------
    def _worker_loop(self) -> None:
        while True:
            frames = self._q.get()
            try:
                res = self.predict(frames)
                self.cp_clips_analyzed += 1
                self.cp_last_score = res["score"]
                self.cp_last_raw   = res["raw"]
                self.cp_last_hits  = res["hits"]

                if res.get("fast_track"):
                    self.cp_consec_pos = 10
                elif res["detected"]:
                    self.cp_consec_pos = min(6, self.cp_consec_pos + 1)
                else:
                    self.cp_consec_pos = max(0, self.cp_consec_pos - 1)

                self.cp_is_detected = (self.cp_consec_pos >= CP_CONSEC_MIN)
            except Exception:
                pass
            self._cp_processing = False

    # ------------------------------------------------------------------
    # Helpers (idénticos a CarPartsDetector)
    # ------------------------------------------------------------------
    @staticmethod
    def _dist_persona_vehiculo(box_p, box_v) -> float:
        cx = (box_p[0] + box_p[2]) / 2.0
        cy = box_p[3]
        dx = max(box_v[0] - cx, 0.0, cx - box_v[2])
        dy = max(box_v[1] - cy, 0.0, cy - box_v[3])
        return float(np.sqrt(dx * dx + dy * dy))

    @staticmethod
    def _score_candidato(box_p, box_v, prev_box_p) -> float:
        cx = (box_p[0] + box_p[2]) / 2.0
        cy = box_p[3]
        dx = max(box_v[0] - cx, 0.0, cx - box_v[2])
        dy = max(box_v[1] - cy, 0.0, cy - box_v[3])
        dist       = float(np.sqrt(dx * dx + dy * dy)) + 1e-3
        prox_score = 100.0 / dist

        mov_score = 0.0
        if prev_box_p is not None:
            xA    = max(box_p[0], prev_box_p[0]); yA = max(box_p[1], prev_box_p[1])
            xB    = min(box_p[2], prev_box_p[2]); yB = min(box_p[3], prev_box_p[3])
            inter = max(0.0, xB - xA) * max(0.0, yB - yA)
            a1    = (box_p[2] - box_p[0]) * (box_p[3] - box_p[1])
            a2    = (prev_box_p[2] - prev_box_p[0]) * (prev_box_p[3] - prev_box_p[1])
            iou   = inter / (a1 + a2 - inter + 1e-6)
            mov_score = (1.0 - iou) * 50.0

        return prox_score + mov_score

    @staticmethod
    def _crop_union(frame: np.ndarray, box_p, box_v, pad: int) -> Optional[np.ndarray]:
        H, W   = frame.shape[:2]
        cx_p   = (box_p[0] + box_p[2]) / 2.0
        cy_p   = box_p[3]
        vx_near = box_v[0] if cx_p < (box_v[0] + box_v[2]) / 2 else box_v[2]
        vy_near = box_v[1] if cy_p < (box_v[1] + box_v[3]) / 2 else box_v[3]
        x1 = int(max(0, min(box_p[0], vx_near) - pad))
        y1 = int(max(0, min(box_p[1], vy_near) - pad))
        x2 = int(min(W, max(box_p[2], vx_near) + pad))
        y2 = int(min(H, max(box_p[3], vy_near) + pad))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            return None
        return cv2.resize(crop, VIEW_SIZE)

    # ------------------------------------------------------------------
    # Pipeline principal (idéntico a CarPartsDetector._process_carparts)
    # ------------------------------------------------------------------
    def _process_carparts(
        self,
        frame:        np.ndarray,
        person_xyxy,
        person_ids,
        vehicle_xyxy,
        vehicle_ids,
    ) -> None:

        # 1. Encontrar candidatos dentro del perímetro
        candidatos = []
        for i, box_p in enumerate(person_xyxy):
            p_id = int(person_ids[i]) if i < len(person_ids) else -1
            for j, box_v in enumerate(vehicle_xyxy):
                dist = self._dist_persona_vehiculo(box_p, box_v)
                if dist < CP_PERIMETRO_PX:
                    v_id  = int(vehicle_ids[j]) if j < len(vehicle_ids) else -1
                    score = self._score_candidato(box_p, box_v, self.cp_prev_boxes.get(p_id))
                    candidatos.append((score, dist, box_p.copy(), box_v.copy(), p_id, v_id))

        # Actualizar historial de bboxes
        ids_visibles = set()
        for i, box_p in enumerate(person_xyxy):
            p_id = int(person_ids[i]) if i < len(person_ids) else -1
            self.cp_prev_boxes[p_id] = box_p.copy()
            ids_visibles.add(p_id)
        self.cp_prev_boxes = {k: v for k, v in self.cp_prev_boxes.items() if k in ids_visibles}

        # 2. Seleccionar objetivo
        best_box_p = best_box_v = None
        best_p_id  = best_v_id  = None
        pad        = CP_PAD_SOLO
        self.cp_crop_mode = "solo"

        if candidatos:
            # Target-lock
            if self.cp_target_locked and self.cp_locked_p_id is not None:
                locked = [c for c in candidatos if c[4] == self.cp_locked_p_id]
                if locked:
                    _, _, best_box_p, best_box_v, best_p_id, best_v_id = locked[0]
                else:
                    self.cp_target_locked = False
                    self.cp_locked_p_id   = None

            if best_box_p is None:
                candidatos.sort(key=lambda x: -x[0])
                _, _, best_box_p, best_box_v, best_p_id, best_v_id = candidatos[0]

            # 3. Crop de grupo
            if len(candidatos) >= 2:
                box_p2  = candidatos[1][2]
                cx1     = (best_box_p[0] + best_box_p[2]) / 2
                cy1     = (best_box_p[1] + best_box_p[3]) / 2
                cx2     = (box_p2[0]    + box_p2[2])    / 2
                cy2     = (box_p2[1]    + box_p2[3])    / 2
                dist_pp = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
                if dist_pp < CP_GRUPO_DIST_PX:
                    best_box_p = np.array([
                        min(best_box_p[0], box_p2[0]),
                        min(best_box_p[1], box_p2[1]),
                        max(best_box_p[2], box_p2[2]),
                        max(best_box_p[3], box_p2[3]),
                    ], dtype=np.float32)
                    pad = CP_PAD_GRUPO
                    self.cp_crop_mode = "grupo"

        # 4. Target-lock: activar/desactivar según score CLIP
        if self.cp_last_score >= CP_LOCK_SCORE_TH and best_p_id is not None:
            self.cp_target_locked = True
            self.cp_locked_p_id   = best_p_id
        elif self.cp_last_score < 1.0:
            self.cp_target_locked = False
            self.cp_locked_p_id   = None

        # 5. Generar y bufferizar crop
        if best_box_p is not None:
            self.cp_last_valid_bbox = (best_box_p, best_box_v)
            self.cp_sticky_counter  = CP_STICKY_FRAMES
            self.cp_last_p_id       = best_p_id
            self.cp_last_v_id       = best_v_id
        else:
            self.cp_sticky_counter = max(0, self.cp_sticky_counter - 1)

        if self.cp_last_valid_bbox is not None and self.cp_sticky_counter > 0:
            box_p, box_v = self.cp_last_valid_bbox
            p_crop = self._crop_union(frame, box_p, box_v, pad)
            if p_crop is not None:
                self.cp_last_crop = p_crop
                self.cp_buffer.append(p_crop)
                if len(self.cp_buffer) >= CP_BUFFER_SIZE:
                    if not self._cp_processing:
                        try:
                            self._q.put_nowait(list(self.cp_buffer))
                            self._cp_processing = True
                        except queue.Full:
                            pass
                    self.cp_buffer = []
        else:
            self.cp_buffer      = []
            self.cp_consec_pos  = max(0, self.cp_consec_pos - 0.2)
            self.cp_is_detected = (self.cp_consec_pos >= CP_CONSEC_MIN)
