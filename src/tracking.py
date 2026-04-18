import cv2
import mediapipe as mp


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def get_hand_detector(max_hands=2, detection_conf=0.7, tracking_conf=0.5):
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max_hands,
        min_detection_confidence=detection_conf,
        min_tracking_confidence=tracking_conf,
    )


def process_frame(frame, detector):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = detector.process(rgb)
    rgb.flags.writeable = True
    return results


def draw_landmarks(frame, results):
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_lms,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )
    return frame


def run_live_tracking():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: no se pudo abrir la camara")
        return

    print("Tracking activo — presiona Q para salir")
    detected_frames = 0
    total_frames = 0

    with get_hand_detector() as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            total_frames += 1

            results = process_frame(frame, detector)

            if results.multi_hand_landmarks:
                detected_frames += 1
                frame = draw_landmarks(frame, results)

                for i, hand_lms in enumerate(results.multi_hand_landmarks):
                    label = results.multi_handedness[i].classification[0].label
                    cv2.putText(frame, label, (10, 30 + i * 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            tasa = (detected_frames / total_frames * 100) if total_frames else 0
            cv2.putText(frame, f"Deteccion: {tasa:.1f}%", (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("APRENDIA1 - Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Resumen: {detected_frames}/{total_frames} frames con mano detectada ({tasa:.1f}%)")


if __name__ == "__main__":
    run_live_tracking()