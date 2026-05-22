from src.tracking import get_holistic_detector, process_frame, draw_landmarks
from src.landmarks import extract_frame_vector, normalize_vector, SequenceBuffer
import cv2

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    buffer = SequenceBuffer()

    with get_holistic_detector() as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            results = process_frame(frame, detector)
            frame = draw_landmarks(frame, results)

            # Extrae y normaliza el vector del frame actual
            vector = extract_frame_vector(results)
            vector_norm = normalize_vector(vector)
            buffer.add(vector_norm)

            # Cuando el buffer tiene 30 frames, la secuencia está lista
            estado = f"Buffer: {len(buffer.buffer)}/30"
            if buffer.is_ready():
                secuencia = buffer.get_sequence()  # (30, 1629)
                estado = f"Secuencia lista: {secuencia.shape}"

            cv2.putText(frame, estado, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("APRENDIA1 - Holistic", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()