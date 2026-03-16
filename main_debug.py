"""
main_debug.py — Orquestador Arconte con cuadrícula de debug
============================================================
Misma lógica de detección que main.py, pero la salida visual es un
grid 3×2 que muestra exactamente qué crop recibe cada experto.

Layout (cada celda = CELL_SIZE = 320×240):
  ┌─────────────────┬─────────────────┬─────────────────┐
  │  MAIN + YOLO    │  PELEA (crop)   │  CHOQUE (crop)  │
  ├─────────────────┼─────────────────┼─────────────────┤
  │  FUEGO  (crop)  │  HUMO   (crop)  │  ROBO   (crop)  │
  └─────────────────┴─────────────────┴─────────────────┘
  [======================== INFO BAR ====================]

Archivos generados:
  grabaciones/ARCONTE_DEBUG_LIVE.jpg      ← frame actual (sobreescrito)
  grabaciones/DEBUG_<ts>_F<n>_<label>.jpg ← snapshot en cada transición
  grabaciones/ARCONTE_DEBUG_<ts>.avi      ← video del grid completo

Uso:
  python main_debug.py video.mp4
  python main_debug.py video.mp4 --full-res
  python main_debug.py video.mp4 --fire-model /ruta/best_large.pt
"""

import os
import sys
import time
from typing import List, Dict, Any, Set, Optional

import cv2
import numpy as np
import torch
import clip

from core.base_expert import BaseExpert
from core.tracker import ArconteTracker
from experts.fight_expert import FightExpert
from experts.crash_expert import CrashExpert
from experts.fire_expert import make_fire_smoke_experts, FireExpert, SmokeExpert
from experts.carparts_expert import CarPartsExpert

# ---------------------------------------------------------------------------
# Configuración del grid de debug
# ---------------------------------------------------------------------------
CELL_SIZE  = (320, 240)                         # tamaño de cada celda
GRID_COLS  = 3
GRID_ROWS  = 2
GRID_W     = CELL_SIZE[0] * GRID_COLS           # 960
GRID_H     = CELL_SIZE[1] * GRID_ROWS           # 480
INFO_H     = 90                                 # altura de la barra de info
TOTAL_SIZE = (GRID_W, GRID_H + INFO_H)          # (960, 570)

SNAPSHOT_ON_TRANSITION = True

OUTPUT_FPS   = 20.0
OUTPUT_CODEC = "XVID"

# Colores por experto (BGR)
EXPERT_COLORS = {
    "PELEA":  (0,   0,   220),
    "CHOQUE": (0,   140, 255),
    "FUEGO":  (0,   69,  255),
    "HUMO":   (160, 160, 160),
    "ROBO":   (0,   0,   180),
}
COLOR_IDLE   = (30, 30, 30)
COLOR_NORMAL = (0, 80, 0)


# ---------------------------------------------------------------------------
# Construcción de paneles
# ---------------------------------------------------------------------------
def _make_panel(
    crop:         Optional[np.ndarray],
    title:        str,
    is_active:    bool,
    score_str:    str,
    extra_lines:  List[str] = None,
) -> np.ndarray:
    """
    Construye una celda CELL_SIZE con:
      - Imagen del crop (o fondo negro "sin crop")
      - Barra de estado superior (50px): TITULO [ON/OFF] score
      - Líneas de debug opcionales en la parte inferior
      - Borde de alerta si activo
    """
    W, H = CELL_SIZE
    color_active = EXPERT_COLORS.get(title, (0, 0, 200))

    if crop is not None and crop.size > 0:
        panel = cv2.resize(crop.copy(), CELL_SIZE)
    else:
        panel = np.zeros((H, W, 3), dtype=np.uint8)
        cv2.putText(panel, "sin crop", (W // 2 - 45, H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (70, 70, 70), 1, cv2.LINE_AA)

    # Barra de estado superior
    bar_color = color_active if is_active else COLOR_IDLE
    cv2.rectangle(panel, (0, 0), (W, 46), bar_color, -1)
    state  = "[ON] " if is_active else "[OFF]"
    label  = f"{title} {state}  {score_str}"
    cv2.putText(panel, label, (6, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    # Líneas de debug en la parte inferior
    if extra_lines:
        y_start = H - 14 * len(extra_lines) - 4
        bg_h    = 14 * len(extra_lines) + 6
        cv2.rectangle(panel, (0, H - bg_h), (W, H), (0, 0, 0), -1)
        for i, line in enumerate(extra_lines):
            cv2.putText(panel, line, (4, y_start + 13 * i + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

    # Borde de alerta
    if is_active:
        cv2.rectangle(panel, (0, 0), (W - 1, H - 1), color_active, 3)

    return panel


def _make_main_panel(
    frame:       np.ndarray,
    tracker_data: Dict[str, Any],
    anomalies:   Set[str],
    fps:         float,
    f_idx:       int,
) -> np.ndarray:
    """
    Panel principal: frame con boxes YOLO dibujados y barra de estado.
    Azul = persona | Verde = vehículo
    """
    W, H   = CELL_SIZE
    panel  = cv2.resize(frame.copy(), CELL_SIZE)
    sx     = W / frame.shape[1]
    sy     = H / frame.shape[0]

    # Dibujar personas
    for box in tracker_data.get("persons_xyxy", []):
        x1, y1, x2, y2 = (int(box[0]*sx), int(box[1]*sy),
                           int(box[2]*sx), int(box[3]*sy))
        cv2.rectangle(panel, (x1, y1), (x2, y2), (255, 100, 0), 1)

    # Dibujar vehículos (todos, incluyendo motos)
    for box in tracker_data.get("vehicles_all_xyxy",
                                tracker_data.get("vehicles_xyxy", [])):
        x1, y1, x2, y2 = (int(box[0]*sx), int(box[1]*sy),
                           int(box[2]*sx), int(box[3]*sy))
        cv2.rectangle(panel, (x1, y1), (x2, y2), (0, 200, 80), 1)

    # Barra de estado
    if anomalies:
        txt       = " | ".join(sorted(anomalies))
        bar_color = (0, 0, 120)
        cv2.rectangle(panel, (0, 0), (W - 1, H - 1), (0, 0, 200), 3)
    else:
        txt       = "NORMAL"
        bar_color = COLOR_NORMAL

    cv2.rectangle(panel, (0, 0), (W, 46), bar_color, -1)
    cv2.putText(panel, txt, (6, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Frame info (esquina inferior)
    info = f"F:{f_idx}  {fps:.1f}fps  {time.strftime('%H:%M:%S')}"
    cv2.rectangle(panel, (0, H - 20), (W, H), (0, 0, 0), -1)
    cv2.putText(panel, info, (4, H - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

    return panel


def _make_info_bar(experts: List[BaseExpert], anomalies: Set[str], f_idx: int) -> np.ndarray:
    """
    Barra horizontal inferior con el estado resumido de cada experto.
    """
    bar = np.zeros((INFO_H, GRID_W, 3), dtype=np.uint8)
    cv2.rectangle(bar, (0, 0), (GRID_W, INFO_H),
                  (0, 0, 60) if anomalies else (0, 40, 0), -1)

    # Título global
    status = " | ".join(sorted(anomalies)) if anomalies else "NORMAL"
    cv2.putText(bar, f"  {status}   F:{f_idx}", (6, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Estado por experto en una línea
    col_w = GRID_W // len(experts)
    for i, expert in enumerate(experts):
        x  = i * col_w
        dd = _get_expert_debug_state(expert)
        active_color = EXPERT_COLORS.get(expert.label, (200, 200, 200))
        text_color   = active_color if expert.is_active else (120, 120, 120)
        cv2.putText(bar, f"{expert.label}:{dd['short']}", (x + 4, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA)
        cv2.putText(bar, dd["detail"], (x + 4, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1, cv2.LINE_AA)
        if i > 0:
            cv2.line(bar, (x, 30), (x, INFO_H), (60, 60, 60), 1)

    return bar


# ---------------------------------------------------------------------------
# Extracción de estado de debug por tipo de experto
# ---------------------------------------------------------------------------
def _get_expert_debug_state(expert: BaseExpert) -> Dict[str, str]:
    """Devuelve campos de debug según el tipo de experto."""

    if isinstance(expert, FightExpert):
        cp  = expert.fight_consec_pos
        st  = expert.fight_sticky_counter
        buf = len(expert.fight_buffer)
        return {
            "short":  f"cp={cp:.1f}  st={st}",
            "detail": f"buf={buf}/{expert.__class__.__module__}",
        }

    if isinstance(expert, CrashExpert):
        cp     = expert.crash_consec_pos
        locked = "LOCK" if expert.crash_locked else "free"
        buf    = len(expert.crash_buffer)
        cnt    = expert.crash_counter
        return {
            "short":  f"cp={cp}  {locked}  cnt={cnt:.0f}",
            "detail": f"buf={buf}/64  ever={expert.crash_ever_detected}",
        }

    if isinstance(expert, FireExpert):
        cp  = expert.fire_clip_consec_pos
        ttl = expert.fire_clip_ttl
        buf = len(expert.fire_clip_buffer)
        hf  = sum(expert._detector.fire_history)
        return {
            "short":  f"cp={cp}  ttl={ttl}  hf={hf}",
            "detail": f"buf={buf}/20",
        }

    if isinstance(expert, SmokeExpert):
        cp  = expert.smoke_clip_consec_pos
        ttl = expert.smoke_clip_ttl
        buf = len(expert.smoke_clip_buffer)
        hs  = sum(expert._detector.smoke_history)
        return {
            "short":  f"cp={cp}  ttl={ttl}  hs={hs}",
            "detail": f"buf={buf}/20",
        }

    if isinstance(expert, CarPartsExpert):
        sc  = expert.cp_last_score
        raw = expert.cp_last_raw
        cp  = expert.cp_consec_pos
        buf = len(expert.cp_buffer)
        return {
            "short":  f"sc={sc:.2f}  raw={raw:.2f}  cp={cp:.1f}",
            "detail": f"buf={buf}/6  clips={expert.cp_clips_analyzed}  {expert.cp_crop_mode}",
        }

    return {"short": "", "detail": ""}


def _get_expert_crop(expert: BaseExpert) -> Optional[np.ndarray]:
    """Obtiene el último crop de cada experto para mostrarlo en su panel."""
    if isinstance(expert, FightExpert):
        return expert.fight_last_crop
    if isinstance(expert, CrashExpert):
        return expert.crash_last_crop
    if isinstance(expert, (FireExpert, SmokeExpert)):
        return expert._detector._last_raw_clip_crop
    if isinstance(expert, CarPartsExpert):
        return expert.cp_last_crop
    data = expert.get_display_data()
    return data.get("last_crop")


def _get_expert_score_str(expert: BaseExpert) -> str:
    """Cadena de score compacta para la barra de cada panel."""
    if isinstance(expert, FightExpert):
        return f"cp={expert.fight_consec_pos:.1f}  st={expert.fight_sticky_counter}"
    if isinstance(expert, CrashExpert):
        return f"cp={expert.crash_consec_pos}  {'LOCK' if expert.crash_locked else 'free'}"
    if isinstance(expert, FireExpert):
        return f"cp={expert.fire_clip_consec_pos}  ttl={expert.fire_clip_ttl}"
    if isinstance(expert, SmokeExpert):
        return f"cp={expert.smoke_clip_consec_pos}  ttl={expert.smoke_clip_ttl}"
    if isinstance(expert, CarPartsExpert):
        return f"sc={expert.cp_last_score:.2f}  raw={expert.cp_last_raw:.2f}"
    return ""


def _get_expert_extra_lines(expert: BaseExpert) -> List[str]:
    """Líneas de debug en la parte inferior de cada panel."""
    if isinstance(expert, FightExpert):
        buf = len(expert.fight_buffer)
        return [f"buf:{buf}/20  proc:{expert._fight_processing}"]

    if isinstance(expert, CrashExpert):
        buf = len(expert.crash_buffer)
        return [
            f"buf:{buf}/64  cnt:{expert.crash_counter:.0f}  type:{expert.crash_interaction_type}",
            f"proc:{expert._crash_processing}  ever:{expert.crash_ever_detected}",
        ]

    if isinstance(expert, FireExpert):
        buf = len(expert.fire_clip_buffer)
        hf  = sum(expert._detector.fire_history)
        return [f"buf:{buf}/20  hits_f:{hf}/18  proc:{expert._fire_clip_processing}"]

    if isinstance(expert, SmokeExpert):
        buf = len(expert.smoke_clip_buffer)
        hs  = sum(expert._detector.smoke_history)
        return [f"buf:{buf}/20  hits_s:{hs}/30  proc:{expert._smoke_clip_processing}"]

    if isinstance(expert, CarPartsExpert):
        buf = len(expert.cp_buffer)
        return [
            f"buf:{buf}/6  hits:{expert.cp_last_hits}  clips:{expert.cp_clips_analyzed}",
            f"lock:{expert.cp_target_locked}  mode:{expert.cp_crop_mode}  st:{expert.cp_sticky_counter}",
        ]

    return []


# ---------------------------------------------------------------------------
# Ensamblado del grid
# ---------------------------------------------------------------------------
def _build_grid(
    frame:        np.ndarray,
    experts:      List[BaseExpert],
    tracker_data: Dict[str, Any],
    anomalies:    Set[str],
    fps:          float,
    f_idx:        int,
) -> np.ndarray:
    """
    Construye el grid completo 960×570:
      [MAIN][PELEA][CHOQUE]
      [FUEGO][HUMO][ROBO]
      [========INFO BAR========]
    """
    p_main = _make_main_panel(frame, tracker_data, anomalies, fps, f_idx)

    panels = [p_main]
    for expert in experts:
        crop       = _get_expert_crop(expert)
        score_str  = _get_expert_score_str(expert)
        extra      = _get_expert_extra_lines(expert)
        panels.append(_make_panel(crop, expert.label, expert.is_active, score_str, extra))

    # Rellenar hasta 6 paneles si hay menos de 5 expertos
    while len(panels) < GRID_COLS * GRID_ROWS:
        panels.append(np.zeros((CELL_SIZE[1], CELL_SIZE[0], 3), dtype=np.uint8))

    row0 = np.hstack(panels[0:3])
    row1 = np.hstack(panels[3:6])
    grid = np.vstack([row0, row1])

    # Líneas divisorias
    W, H = CELL_SIZE
    cv2.line(grid, (W,   0), (W,   H * 2), (60, 60, 60), 1)
    cv2.line(grid, (W*2, 0), (W*2, H * 2), (60, 60, 60), 1)
    cv2.line(grid, (0,   H), (W*3, H),     (60, 60, 60), 1)

    # Info bar
    info_bar = _make_info_bar(experts, anomalies, f_idx)

    return np.vstack([grid, info_bar])


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------
def run(
    source:          str,
    fire_model_path: str   = None,
    input_size:      tuple = (320, 240),
) -> None:

    # -----------------------------------------------------------------------
    # 1. CLIP
    # -----------------------------------------------------------------------
    clip_device     = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model_path = ".checkpoints/ViT-L-14.pt"
    print(f"[DEBUG] Cargando CLIP desde {clip_model_path} en {clip_device} ...")
    clip_model, clip_preprocess = clip.load(clip_model_path, device=clip_device)
    clip_model.eval()

    # -----------------------------------------------------------------------
    # 2. Tracker + Expertos
    # -----------------------------------------------------------------------
    tracker = ArconteTracker()

    fire_kwargs = {"fire_model_path": fire_model_path} if fire_model_path else {}
    fire_expert, smoke_expert = make_fire_smoke_experts(**fire_kwargs)

    experts: List[BaseExpert] = [
        FightExpert(),
        CrashExpert(),
        fire_expert,
        smoke_expert,
        CarPartsExpert(),
    ]

    for expert in experts:
        expert.load(clip_model, clip_preprocess)
        print(f"[DEBUG] {expert!r} listo.")

    # -----------------------------------------------------------------------
    # 3. Video entrada / salida
    # -----------------------------------------------------------------------
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir: {source}")
        sys.exit(1)

    os.makedirs("grabaciones", exist_ok=True)
    ts     = time.strftime("%Y%m%d_%H%M%S")
    fourcc = cv2.VideoWriter_fourcc(*OUTPUT_CODEC)
    out    = cv2.VideoWriter(
        f"grabaciones/ARCONTE_DEBUG_{ts}.avi",
        fourcc, OUTPUT_FPS, TOTAL_SIZE,
    )

    input_tag = f"{input_size[0]}×{input_size[1]}" if input_size else "original"
    print(f"\n{'='*60}")
    print("  ARCONTE FRAMEWORK — DEBUG GRID")
    print(f"  Fuente     : {source}")
    print(f"  Device     : {clip_device}")
    print(f"  Input size : {input_tag}")
    print(f"  Grid size  : {TOTAL_SIZE[0]}×{TOTAL_SIZE[1]}")
    print(f"{'='*60}\n")

    # -----------------------------------------------------------------------
    # 4. Loop
    # -----------------------------------------------------------------------
    prev_anomalies: Set[str] = set()
    fps_ema: float           = 0.0
    t_prev: float            = time.perf_counter()
    f_idx:  int              = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        t_now   = time.perf_counter()
        fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
        t_prev  = t_now

        # Normalización de entrada
        if input_size is not None:
            frame = cv2.resize(frame, input_size)

        # A) Tracking
        tracker_data: Dict[str, Any] = tracker.process(frame)

        # B) Heurísticas
        for expert in experts:
            expert.process_heuristics(frame, tracker_data)

        # C) Anomalías
        anomalies: Set[str] = {expert.label for expert in experts if expert.is_active}

        # D) Log de transiciones
        transition = (anomalies != prev_anomalies)
        if transition:
            ts_log = time.strftime("%H:%M:%S")
            if anomalies:
                print(f"[{ts_log}] F:{f_idx}  ON  → {' | '.join(sorted(anomalies))}")
            else:
                print(f"[{ts_log}] F:{f_idx}  OFF → NORMAL")

        # E) Construir grid
        grid = _build_grid(frame, experts, tracker_data, anomalies, fps_ema, f_idx)

        # F) Guardar
        out.write(grid)
        cv2.imwrite("grabaciones/ARCONTE_DEBUG_LIVE.jpg", grid,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])

        if transition and SNAPSHOT_ON_TRANSITION:
            label     = "_".join(sorted(anomalies)) if anomalies else "NORMAL"
            snap_path = f"grabaciones/DEBUG_{time.strftime('%Y%m%d_%H%M%S')}_F{f_idx}_{label}.jpg"
            cv2.imwrite(snap_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f"  → snapshot: {snap_path}")

        prev_anomalies = anomalies
        f_idx += 1

    # -----------------------------------------------------------------------
    # 5. Limpieza
    # -----------------------------------------------------------------------
    cap.release()
    out.release()
    print(f"\n[DEBUG] Frames procesados: {f_idx}")
    print(f"[DEBUG] Video guardado en grabaciones/ARCONTE_DEBUG_{ts}.avi")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Arconte Debug Grid")
    parser.add_argument("source",                           help="Video o cámara")
    parser.add_argument("--fire-model", default=None,       help="Ruta a best_large.pt")
    parser.add_argument("--full-res",   action="store_true",
                        help="Usar resolución original (sin normalizar a 320×240)")
    args = parser.parse_args()

    input_size = None if args.full_res else (320, 240)
    run(args.source, fire_model_path=args.fire_model, input_size=input_size)
