"""
head_pose.py
============
Head pose estimation for the Driver Monitoring System.

Mode 1 (Haar proxy): uses ratio of frontal vs profile face detections.
Mode 2 (MediaPipe + solvePnP): uses 3-D model points and MediaPipe
landmarks to compute yaw, pitch, and roll.

Author: Daifi Meriem, Intern at Expleo Group Maroc for Stellantis
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
YAW_THRESHOLD_DEG: float = 25.0
PITCH_THRESHOLD_DEG: float = 20.0
ROLL_THRESHOLD_DEG: float = 20.0

MODEL_POINTS: np.ndarray = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0),
], dtype=np.float64)

MP_POSE_INDICES: List[int] = [1, 152, 263, 33, 287, 57]


@dataclass
class HeadPoseResult:
    """Head pose estimation output.

    Args:
        yaw: Rotation around vertical axis (degrees).
        pitch: Rotation around lateral axis (degrees).
        roll: Rotation around longitudinal axis (degrees).
        is_distracted: True if any angle exceeds its threshold.
        rotation_vector: OpenCV rvec (None for Haar mode).
        translation_vector: OpenCV tvec (None for Haar mode).
    """
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    is_distracted: bool = False
    rotation_vector: Optional[np.ndarray] = None
    translation_vector: Optional[np.ndarray] = None


class HeadPoseEstimator:
    """Estimate head pose from face landmarks or Haar proxy.

    Args:
        use_mediapipe: If True use solvePnP; if False use Haar proxy.
        yaw_threshold: Distraction yaw threshold in degrees.
        pitch_threshold: Distraction pitch threshold in degrees.
        roll_threshold: Distraction roll threshold in degrees.
        frame_width: Camera frame width.
        frame_height: Camera frame height.
    """

    def __init__(
        self,
        use_mediapipe: bool = True,
        yaw_threshold: float = YAW_THRESHOLD_DEG,
        pitch_threshold: float = PITCH_THRESHOLD_DEG,
        roll_threshold: float = ROLL_THRESHOLD_DEG,
        frame_width: int = 640,
        frame_height: int = 480,
    ) -> None:
        self.use_mediapipe = use_mediapipe
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.roll_threshold = roll_threshold
        self._camera_matrix: np.ndarray = self._build_camera_matrix(frame_width, frame_height)
        self._dist_coeffs: np.ndarray = np.zeros((4, 1), dtype=np.float64)
        self._profile_cascade: Optional[cv2.CascadeClassifier] = None
        if not use_mediapipe:
            self._init_profile_cascade()

    @staticmethod
    def _build_camera_matrix(w: int, h: int) -> np.ndarray:
        """Approximate pinhole camera matrix.

        Args:
            w: Frame width in pixels.
            h: Frame height in pixels.

        Returns:
            3x3 camera intrinsic matrix.
        """
        focal = w
        cx, cy = w / 2.0, h / 2.0
        return np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=np.float64)

    def _init_profile_cascade(self) -> None:
        """Load Haar profile face cascade for proxy mode."""
        try:
            cv_data = cv2.data.haarcascades
            path = f"{cv_data}haarcascade_profileface.xml"
            self._profile_cascade = cv2.CascadeClassifier(path)
        except Exception as exc:
            logger.warning("Could not load profile cascade: %s", exc)

    def estimate(
        self,
        landmarks: List[Tuple[float, float]],
        frame: Optional[np.ndarray] = None,
    ) -> HeadPoseResult:
        """Estimate head pose.

        Args:
            landmarks: List of (x, y) pixel coordinates from MediaPipe.
            frame: BGR frame required for Haar proxy mode.

        Returns:
            HeadPoseResult with yaw/pitch/roll and distraction flag.
        """
        if self.use_mediapipe and len(landmarks) > max(MP_POSE_INDICES):
            return self._estimate_pnp(landmarks)
        if frame is not None:
            return self._estimate_haar_proxy(frame)
        return HeadPoseResult()

    def _estimate_pnp(self, landmarks: List[Tuple[float, float]]) -> HeadPoseResult:
        """Compute yaw/pitch/roll using solvePnP.

        Args:
            landmarks: Full MediaPipe landmark list (pixel coords).

        Returns:
            HeadPoseResult with solved angles.
        """
        image_points = np.array(
            [landmarks[i] for i in MP_POSE_INDICES], dtype=np.float64
        )
        success, rvec, tvec = cv2.solvePnP(
            MODEL_POINTS, image_points, self._camera_matrix,
            self._dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return HeadPoseResult()

        rot_mat, _ = cv2.Rodrigues(rvec)
        angles = self._rotation_matrix_to_euler(rot_mat)
        pitch, yaw, roll = float(angles[0]), float(angles[1]), float(angles[2])

        distracted = (
            abs(yaw) > self.yaw_threshold
            or abs(pitch) > self.pitch_threshold
            or abs(roll) > self.roll_threshold
        )
        return HeadPoseResult(
            yaw=yaw, pitch=pitch, roll=roll, is_distracted=distracted,
            rotation_vector=rvec, translation_vector=tvec,
        )

    def _estimate_haar_proxy(self, frame: np.ndarray) -> HeadPoseResult:
        """Approximate head yaw via profile vs frontal detection.

        Args:
            frame: BGR frame.

        Returns:
            HeadPoseResult with approximate yaw.
        """
        if self._profile_cascade is None or self._profile_cascade.empty():
            return HeadPoseResult()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        profiles = self._profile_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
        )
        approx_yaw = float(min(len(profiles) * 15.0, 90.0))
        distracted = approx_yaw > self.yaw_threshold
        return HeadPoseResult(yaw=approx_yaw, pitch=0.0, roll=0.0, is_distracted=distracted)

    @staticmethod
    def _rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to Euler angles (pitch, yaw, roll) in degrees.

        Args:
            R: 3x3 rotation matrix.

        Returns:
            Array [pitch, yaw, roll] in degrees.
        """
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0.0
        return np.degrees(np.array([x, y, z]))

    def update_frame_size(self, width: int, height: int) -> None:
        """Update camera matrix when frame resolution changes.

        Args:
            width: New frame width.
            height: New frame height.
        """
        self._camera_matrix = self._build_camera_matrix(width, height)
