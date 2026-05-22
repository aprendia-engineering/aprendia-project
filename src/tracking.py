import cv2
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def get_holistic_detector(detection_conf=0.7, tracking_conf=0.5):
    return mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
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
    # Cara
    mp_draw.draw_landmarks(
        frame,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style(),
    )
    # Pose
    mp_draw.draw_landmarks(
        frame,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        mp_styles.get_default_pose_landmarks_style(),
    )
    # Mano izquierda
    mp_draw.draw_landmarks(
        frame,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_styles.get_default_hand_landmarks_style(),
        mp_styles.get_default_hand_connections_style(),
    )
    # Mano derecha
    mp_draw.draw_landmarks(
        frame,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_styles.get_default_hand_landmarks_style(),
        mp_styles.get_default_hand_connections_style(),
    )
    return frame


def run_live_tracking():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: no se pudo abrir la camara")
        return

    print("Holistic tracking activo — presiona Q para salir")

    with get_holistic_detector() as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            results = process_frame(frame, detector)
            frame = draw_landmarks(frame, results)

            cv2.imshow("APRENDIA1 - Holistic", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_live_tracking()