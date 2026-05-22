import numpy as np


# ─── Constantes ───────────────────────────────────────────────────────────────

N_FACE   = 468   # puntos de cara
N_POSE   = 33    # puntos de cuerpo
N_HAND   = 21    # puntos por mano
N_COORDS = 3     # x, y, z por punto

# Tamaño del vector completo por frame
# cara(468×3) + pose(33×3) + mano_izq(21×3) + mano_der(21×3) = 1629 valores
FRAME_VECTOR_SIZE = (N_FACE + N_POSE + N_HAND + N_HAND) * N_COORDS  # 1629

# Para señas dinámicas: cuántos frames forma una secuencia
SEQUENCE_LENGTH = 30  # ~1 segundo a 30fps


# ─── Extracción por región ────────────────────────────────────────────────────

def extract_face(results):
    """468 landmarks de cara → array (468, 3). Zeros si no hay detección."""
    if results.face_landmarks:
        return np.array(
            [[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]
        )
    return np.zeros((N_FACE, N_COORDS))


def extract_pose(results):
    """33 landmarks de pose → array (33, 3). Zeros si no hay detección."""
    if results.pose_landmarks:
        return np.array(
            [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
        )
    return np.zeros((N_POSE, N_COORDS))


def extract_left_hand(results):
    """21 landmarks mano izquierda → array (21, 3). Zeros si no hay detección."""
    if results.left_hand_landmarks:
        return np.array(
            [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
        )
    return np.zeros((N_HAND, N_COORDS))


def extract_right_hand(results):
    """21 landmarks mano derecha → array (21, 3). Zeros si no hay detección."""
    if results.right_hand_landmarks:
        return np.array(
            [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
        )
    return np.zeros((N_HAND, N_COORDS))


# ─── Vector completo por frame ────────────────────────────────────────────────

def extract_frame_vector(results):
    """
    Combina todas las regiones en un vector plano de 1629 valores.
    Orden: cara + pose + mano_izq + mano_der
    Este es el vector que se guarda en disco y se manda por WebSocket.
    """
    face  = extract_face(results).flatten()
    pose  = extract_pose(results).flatten()
    lhand = extract_left_hand(results).flatten()
    rhand = extract_right_hand(results).flatten()
    return np.concatenate([face, pose, lhand, rhand])


# ─── Normalización ────────────────────────────────────────────────────────────

def normalize_vector(vector):
    """
    Normalización min-max al rango [0, 1].
    Hace el vector invariante a la posición del cuerpo en el encuadre.
    Se aplica antes de alimentar el clasificador.
    """
    min_val = vector.min()
    max_val = vector.max()
    if max_val - min_val == 0:
        return vector
    return (vector - min_val) / (max_val - min_val)


# ─── Secuencias para señas dinámicas ─────────────────────────────────────────

class SequenceBuffer:
    """
    Buffer circular que acumula frames hasta formar una secuencia completa.
    Para señas dinámicas (LSTM): shape final → (SEQUENCE_LENGTH, 1629)
    Para señas estáticas: usa extract_frame_vector directo sin este buffer.
    """

    def __init__(self, length=SEQUENCE_LENGTH):
        self.length = length
        self.buffer = []

    def add(self, frame_vector):
        self.buffer.append(frame_vector)
        if len(self.buffer) > self.length:
            self.buffer.pop(0)

    def is_ready(self):
        return len(self.buffer) == self.length

    def get_sequence(self):
        """Devuelve array (SEQUENCE_LENGTH, 1629) listo para el modelo LSTM."""
        return np.array(self.buffer)

    def reset(self):
        self.buffer = []


# ─── Serialización para WebSocket / FastAPI ───────────────────────────────────

def vector_to_dict(results):
    """
    Convierte los landmarks en un dict JSON-serializable.
    Este es el formato que Flutter recibe por WebSocket para animar el avatar.
    """
    def region_to_list(landmarks, n):
        if landmarks:
            return [[lm.x, lm.y, lm.z] for lm in landmarks.landmark]
        return [[0.0, 0.0, 0.0]] * n

    return {
        "face":       region_to_list(results.face_landmarks,       N_FACE),
        "pose":       region_to_list(results.pose_landmarks,        N_POSE),
        "left_hand":  region_to_list(results.left_hand_landmarks,   N_HAND),
        "right_hand": region_to_list(results.right_hand_landmarks,  N_HAND),
    }