import cv2
import numpy as np
import os
import json
import time
from src.tracking import get_holistic_detector, process_frame, draw_landmarks
from src.landmarks import extract_frame_vector, normalize_vector, SequenceBuffer, SEQUENCE_LENGTH

# ─── Configuración ────────────────────────────────────────────────────────────

DATA_PATH        = os.path.join("data", "processed")
LABELS_PATH      = os.path.join("data", "labels.json")
SEQUENCES_PER_SIGN = 40   # secuencias a grabar por seña
COUNTDOWN        = 3      # segundos de cuenta regresiva antes de cada secuencia


# ─── Manejo de etiquetas ──────────────────────────────────────────────────────

def load_labels():
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_labels(labels):
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)


def register_sign(sign_name, labels):
    if sign_name not in labels:
        labels[sign_name] = len(labels)
        save_labels(labels)
        print(f"Seña registrada: '{sign_name}' → clase {labels[sign_name]}")
    return labels[sign_name]


# ─── Utilidades de pantalla ───────────────────────────────────────────────────

def put_text(frame, text, y, color=(255, 255, 255), scale=0.7, thickness=2):
    cv2.putText(frame, text, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def draw_progress_bar(frame, current, total, y=60):
    w = frame.shape[1]
    bar_w = int((current / total) * (w - 40))
    cv2.rectangle(frame, (20, y), (w - 20, y + 14), (60, 60, 60), -1)
    cv2.rectangle(frame, (20, y), (20 + bar_w, y + 14), (0, 200, 100), -1)


# ─── Cuenta regresiva ─────────────────────────────────────────────────────────

def countdown(cap, detector, sign_name, seq_index, total_seqs):
    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        results = process_frame(frame, detector)
        frame = draw_landmarks(frame, results)

        elapsed = time.time() - start
        remaining = int(COUNTDOWN - elapsed) + 1

        put_text(frame, f"Preparando: '{sign_name}'", 30, (0, 255, 255))
        put_text(frame, f"Secuencia {seq_index + 1} de {total_seqs}", 60, (200, 200, 200))
        put_text(frame, f"Grabando en {remaining}...", 100, (0, 200, 255), scale=1.2, thickness=3)
        put_text(frame, "Coloca tu mano y espera", frame.shape[0] - 20, (180, 180, 180), scale=0.5)

        cv2.imshow("APRENDIA1 - Collector", frame)
        cv2.waitKey(1)

        if elapsed >= COUNTDOWN:
            break


# ─── Grabación de una secuencia ───────────────────────────────────────────────

def record_sequence(cap, detector, sign_name, seq_index, total_seqs, save_path):
    buffer = SequenceBuffer()

    while not buffer.is_ready():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        results = process_frame(frame, detector)
        frame = draw_landmarks(frame, results)

        vector = normalize_vector(extract_frame_vector(results))
        buffer.add(vector)

        progress = len(buffer.buffer)
        draw_progress_bar(frame, progress, SEQUENCE_LENGTH, y=70)
        put_text(frame, f"GRABANDO: '{sign_name}'", 30, (0, 100, 255), scale=0.8, thickness=2)
        put_text(frame, f"Frame {progress}/{SEQUENCE_LENGTH}", 55, (200, 200, 200), scale=0.5)
        put_text(frame, f"Secuencia {seq_index + 1}/{total_seqs}", frame.shape[0] - 20,
                 (180, 180, 180), scale=0.5)

        cv2.imshow("APRENDIA1 - Collector", frame)
        cv2.waitKey(1)

    if buffer.is_ready():
        np.save(save_path, buffer.get_sequence())
        print(f"  Guardado: {save_path} — shape {buffer.get_sequence().shape}")
        return True
    return False


# ─── Flujo principal ──────────────────────────────────────────────────────────

def collect_sign(sign_name):
    labels = load_labels()
    register_sign(sign_name, labels)

    sign_dir = os.path.join(DATA_PATH, sign_name)
    os.makedirs(sign_dir, exist_ok=True)

    # Detecta cuántas secuencias ya existen para no sobreescribir
    existing = len([f for f in os.listdir(sign_dir) if f.endswith(".npy")])
    target   = existing + SEQUENCES_PER_SIGN

    print(f"\nRecolectando '{sign_name}': {existing} existentes, grabando {SEQUENCES_PER_SIGN} nuevas")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: no se pudo abrir la camara")
        return

    with get_holistic_detector() as detector:
        for i in range(existing, target):
            countdown(cap, detector, sign_name, i, target)
            save_path = os.path.join(sign_dir, f"{i}.npy")
            ok = record_sequence(cap, detector, sign_name, i, target, save_path)
            if not ok:
                print(f"  ERROR en secuencia {i}, se omite")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nListo. '{sign_name}' tiene ahora {target} secuencias en {sign_dir}")


if __name__ == "__main__":
    print("=== APRENDIA1 — Recolector de señas ===")
    sign = input("Nombre de la seña a grabar (ej: hola, gracias, ayuda): ").strip().lower()
    if sign:
        collect_sign(sign)
    else:
        print("Nombre vacío, cancelado.")

        # ─── Recolección desde video ──────────────────────────────────────────────────

def collect_from_video(video_path, sign_name):
    """
    Procesa un video existente y extrae secuencias de landmarks.
    Úsalo para alimentar el dataset sin necesidad de grabación en vivo.
    """
    if not os.path.exists(video_path):
        print(f"ERROR: no se encontró el video en {video_path}")
        return

    labels = load_labels()
    register_sign(sign_name, labels)

    sign_dir = os.path.join(DATA_PATH, sign_name)
    os.makedirs(sign_dir, exist_ok=True)

    existing = len([f for f in os.listdir(sign_dir) if f.endswith(".npy")])
    cap      = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nProcesando video: {video_path}")
    print(f"Frames totales en video: {total_frames}")
    print(f"Secuencias de 30 frames que generará: ~{total_frames // SEQUENCE_LENGTH}")

    buffer    = SequenceBuffer()
    seq_index = existing

    with get_holistic_detector() as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                # Video terminó — si el buffer tiene algo útil lo guarda
                if len(buffer.buffer) >= 15:
                    # Rellena con el último frame para completar la secuencia
                    while not buffer.is_ready():
                        buffer.add(buffer.buffer[-1])
                    save_path = os.path.join(sign_dir, f"{seq_index}.npy")
                    np.save(save_path, buffer.get_sequence())
                    print(f"  Guardado (relleno): {save_path}")
                    seq_index += 1
                break

            results = process_frame(frame, detector)
            frame   = draw_landmarks(frame, results)

            vector  = normalize_vector(extract_frame_vector(results))
            buffer.add(vector)

            # Cada vez que el buffer llena 30 frames, guarda y reinicia
            if buffer.is_ready():
                save_path = os.path.join(sign_dir, f"{seq_index}.npy")
                np.save(save_path, buffer.get_sequence())
                print(f"  Guardado: {save_path}")
                seq_index += 1
                buffer.reset()

            # Muestra progreso en ventana
            frame_actual = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            draw_progress_bar(frame, frame_actual, total_frames, y=70)
            put_text(frame, f"Procesando: '{sign_name}'",  30, (0, 255, 255))
            put_text(frame, f"Video frame {frame_actual}/{total_frames}", 55,
                     (200, 200, 200), scale=0.5)
            put_text(frame, f"Secuencias guardadas: {seq_index - existing}", 80,
                     (0, 200, 100), scale=0.5)

            cv2.imshow("APRENDIA1 - Video Collector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nListo. '{sign_name}' tiene {seq_index} secuencias totales en {sign_dir}")


def collect_from_folder(folder_path, sign_name):
    """
    Procesa todos los videos (.mp4, .avi, .mov) dentro de una carpeta
    y los acumula como secuencias de la misma seña.
    """
    extensions = (".mp4", ".avi", ".mov", ".mkv")
    videos = [
        os.path.join(folder_path, f)
        for f in sorted(os.listdir(folder_path))
        if f.lower().endswith(extensions)
    ]

    if not videos:
        print(f"No se encontraron videos en {folder_path}")
        return

    print(f"Videos encontrados: {len(videos)}")
    for video_path in videos:
        print(f"\n── {video_path}")
        collect_from_video(video_path, sign_name)