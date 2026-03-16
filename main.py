"""
main.py — Orquestador de Arconte Framework
==========================================
Responsabilidades ÚNICAS de este módulo:
    1. Cargar CLIP una sola vez (memoria GPU compartida).
    2. Instanciar el Tracker y los Expertos.
    3. Ejecutar el bucle principal de video.
    4. Pasar tracker_data a cada experto cada frame.
    5. Recolectar anomalías activas y dibujar el overlay.
    6. Escribir el video de salida y el frame LIVE.

Para agregar un nuevo experto: instanciarlo y añadirlo a la lista `experts`.
Para desactivar uno: comentar su línea en esa lista.
"""

import sys
import time
from typing import List, Dict, Any, Set

import cv2
import numpy as np
import torch
import clip

from core.base_expert import BaseExpert
from core.tracker import ArconteTracker
from experts.fight_expert import FightExpert
from experts.crash_expert import CrashExpert
from experts.fire_expert import make_fire_smoke_experts
from experts.carparts_expert import CarPartsExpert

# ---------------------------------------------------------------------------
# Configuración global del Orquestador
# ---------------------------------------------------------------------------
VIEW_SIZE       = (640, 480)
OUTPUT_FPS      = 20.0
OUTPUT_CODEC    = "XVID"
LIVE_FRAME_PATH = "ARCONTE_LIVE_V9.jpg"

LABEL_COLORS: Dict[str, tuple] = {
    "PELEA":  (0,   0,   255),
    "CHOQUE": (0,   165, 255),
    "FUEGO":  (0,   60,  255),
    "HUMO":   (180, 180, 180),
    "ROBO":   (0,   0,   180),
    "NORMAL": (0,   200, 0),
}


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------
def _draw_overlay(
    frame:      np.ndarray,
    anomalies:  Set[str],
    experts:    List[BaseExpert],
    fps:        float,
    src_size:   tuple = None,
) -> None:
    """
    src_size: (W_original, H_original) del frame antes de escalar a VIEW_SIZE.
              Se usa para convertir las coordenadas de los bboxes al espacio
              de display. Si es None se asume que ya están en VIEW_SIZE.
    """
    h, w = frame.shape[:2]
    sx = w / src_size[0] if src_size else 1.0
    sy = h / src_size[1] if src_size else 1.0

    # Barra de estado
    if anomalies:
        label_text = " | ".join(sorted(anomalies))
        bar_color  = LABEL_COLORS.get(next(iter(sorted(anomalies))), (0, 0, 120))
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)
    else:
        label_text = "NORMAL"
        bar_color  = (0, 80, 0)

    cv2.rectangle(frame, (0, 0), (w, 44), bar_color, -1)
    cv2.putText(
        frame, label_text,
        (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, f"FPS:{fps:.1f}",
        (w - 95, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
    )

    # Datos visuales de cada experto activo
    # Los bboxes de fuego vienen en formato (x, y, w, h) en coords originales
    for expert in experts:
        if not expert.is_active:
            continue
        data  = expert.get_display_data()
        color = data.get("color", (0, 0, 255))

        for bbox in data.get("bboxes", []):
            mx, my, mw, mh = bbox
            x1 = int(mx * sx)
            y1 = int(my * sy)
            x2 = int((mx + mw) * sx)
            y2 = int((my + mh) * sy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                frame, expert.label,
                (x1, max(y1 - 8, 50)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
            )

        extra = data.get("extra_text", "")
        if extra:
            cv2.putText(
                frame, extra, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1,
            )


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------
def run(
    source: str,
    fire_model_path: str = None,
    input_size: tuple = (320, 240),
) -> None:
    """
    Bucle principal de Arconte.

    Parámetros:
        source          : Ruta al video o índice de cámara.
        fire_model_path : Ruta al modelo YOLO de fuego/humo (opcional).
        input_size      : Resolución a la que se normalizan los frames de
                          entrada ANTES de cualquier procesamiento.
                          Por defecto (320, 240) para maximizar velocidad.
                          Pasar None para usar la resolución original del video.
    """

    # -----------------------------------------------------------------------
    # 1. CLIP — única instancia compartida por todos los expertos
    # -----------------------------------------------------------------------
    clip_device     = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model_path = ".checkpoints/ViT-L-14.pt"
    print(f"[Arconte] Cargando CLIP ViT-L/14 desde {clip_model_path} en {clip_device} ...")
    clip_model, clip_preprocess = clip.load(clip_model_path, device=clip_device)
    clip_model.eval()
    print("[Arconte] CLIP listo.")

    # -----------------------------------------------------------------------
    # 2. Tracker
    # -----------------------------------------------------------------------
    tracker = ArconteTracker()

    # -----------------------------------------------------------------------
    # 3. Expertos
    # -----------------------------------------------------------------------
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
        print(f"[Arconte] {expert!r} listo.")

    # -----------------------------------------------------------------------
    # 4. Video entrada / salida
    # -----------------------------------------------------------------------
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir: {source}")
        sys.exit(1)

    fourcc = cv2.VideoWriter_fourcc(*OUTPUT_CODEC)
    out    = cv2.VideoWriter("output_arconte.avi", fourcc, OUTPUT_FPS, VIEW_SIZE)

    input_tag = f"{input_size[0]}×{input_size[1]}" if input_size else "original"
    print(f"\n{'='*50}")
    print("  ARCONTE FRAMEWORK — Produccion")
    print(f"  Fuente      : {source}")
    print(f"  Device      : {clip_device}")
    print(f"  Input size  : {input_tag}")
    print(f"{'='*50}\n")

    # -----------------------------------------------------------------------
    # 5. Estado del orquestador
    # -----------------------------------------------------------------------
    prev_anomalies: Set[str] = set()
    fps_ema: float           = 0.0
    t_prev: float            = time.perf_counter()
    f_idx:  int              = 0

    # -----------------------------------------------------------------------
    # 6. Bucle principal
    # -----------------------------------------------------------------------
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # FPS (EMA)
        t_now   = time.perf_counter()
        fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
        t_prev  = t_now

        # Normalización de resolución de entrada (opcional)
        if input_size is not None:
            frame = cv2.resize(frame, input_size)

        # Tamaño de trabajo (para escalar bboxes al overlay)
        H_src, W_src = frame.shape[:2]

        # A) Tracking
        tracker_data: Dict[str, Any] = tracker.process(frame)

        # B) Heurísticas de cada experto
        for expert in experts:
            expert.process_heuristics(frame, tracker_data)

        # C) Anomalías activas
        anomalies: Set[str] = {expert.label for expert in experts if expert.is_active}

        # D) Log de transiciones
        if anomalies != prev_anomalies:
            ts = time.strftime("%H:%M:%S")
            if anomalies:
                print(f"[{ts}] F:{f_idx}  ANOMALIA ON  → {' | '.join(sorted(anomalies))}")
            else:
                print(f"[{ts}] F:{f_idx}  ANOMALIA OFF → NORMAL")
            prev_anomalies = anomalies

        # E) Overlay
        display = cv2.resize(frame, VIEW_SIZE)
        _draw_overlay(display, anomalies, experts, fps_ema, src_size=(W_src, H_src))

        # F) Salidas
        out.write(display)
        cv2.imwrite(LIVE_FRAME_PATH, display)
        # cv2.imshow("Arconte", display)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     print("[Arconte] Salida por usuario.")
        #     break

        f_idx += 1

    # -----------------------------------------------------------------------
    # 7. Limpieza
    # -----------------------------------------------------------------------
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\n{'='*50}")
    print(f"  Frames procesados : {f_idx}")
    print(f"{'='*50}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Arconte Framework")
    parser.add_argument("source",                          help="Video o cámara")
    parser.add_argument("--fire-model", default=None,      help="Ruta a best_large.pt")
    parser.add_argument("--full-res",   action="store_true",
                        help="Usar resolución original del video (sin normalizar a 320×240)")
    args = parser.parse_args()

    input_size = None if args.full_res else (320, 240)
    run(args.source, fire_model_path=args.fire_model, input_size=input_size)
