"""
face_eye_detection.py
=====================
Dual-mode face and eye detection for the Driver Monitoring System.

Supports:
  - Mode 1: Haar Cascade (frontal + profile face, eye ROI)
  - Mode 2: MediaPipe FaceMesh (468 landmarks, EAR-ready)

Both modes produce a unified DetectionResult.

Author: Daifi Meriem, Intern at Expleo Group Maroc for Stellantis
"""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HAAR_SCALE_FACTOR: float = 1.1
HAAR_MIN_NEIGHBORS: int = 5
HAAR_MIN_SIZE: Tuple[int, int] = (30, 30)

LEFT_EYE_INDICES: List[int] = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES: List[int] = [33, 160, 158, 133, 153, 144]
MOUTH_INDICES: List[int] = [61, 291, 0, 17, 269, 405, 314, 13, 84, 375, 78, 191]

MEDIAPIPE_MAX_FACES: int = 1
MEDIAPIPE_MIN_DETECTION_CONFIDENCE: float = 0.5
MEDIAPIPE_MIN_TRACKING_CONFIDENCE: float = 0.5


class DetectionMode(Enum):
    """Available detection backend modes."""
    HAAR = auto()
    MEDIAPIPE = auto()


@dataclass
class BoundingBox:
    """Axis-aligned bounding box.

    Args:
        x: Left pixel coordinate.
        y: Top pixel coordinate.
        w: Width in pixels.
        h: Height in pixels.
    """
    x: int
    y: int
    w: int
    h: int

    @property
    def as_tuple(self) -> Tuple[int, int, int, int]:
        """Return (x, y, w, h) tuple."""
        return self.x, self.y, self.w, self.h

    @property
    def center(self) -> Tuple[int, int]:
        """Return (cx, cy) center of the box."""
        return self.x + self.w // 2, self.y + self.h // 2


@dataclass
class DetectionResult:
    """Unified detection output regardless of backend mode.

    Args:
        face_bbox: Bounding box of the detected face, or None.
        eye_bboxes: List of bounding boxes for detected eyes.
        landmarks: List of (x, y) pixel landmark coordinates.
        left_eye_landmarks: 6-point list for left eye EAR.
        right_eye_landmarks: 6-point list for right eye EAR.
        mouth_landmarks: Landmark list for MAR computation.
        face_detected: True if at least one face was found.
        mode: Which DetectionMode was used.
    """
    face_bbox: Optional[BoundingBox] = None
    eye_bboxes: List[BoundingBox] = field(default_factory=list)
    landmarks: List[Tuple[float, float]] = field(default_factory=list)
    left_eye_landmarks: List[Tuple[float, float]] = field(default_factory=list)
    right_eye_landmarks: List[Tuple[float, float]] = field(default_factory=list)
    mouth_landmarks: List[Tuple[float, float]] = field(default_factory=list)
    face_detected: bool = False
    mode: DetectionMode = DetectionMode.HAAR


class FaceEyeDetector:
    """Dual-mode face and eye detector.

    Args:
        mode: DetectionMode.HAAR or DetectionMode.MEDIAPIPE.
        face_cascade_path: Path to Haar frontal face XML.
        profile_cascade_path: Path to Haar profile face XML.
        eye_cascade_path: Path to Haar eye XML.
    """

    def __init__(
        self,
        mode: DetectionMode = DetectionMode.MEDIAPIPE,
        face_cascade_path: str = "",
        profile_cascade_path: str = "",
        eye_cascade_path: str = "",
    ) -> None:
        self.mode = mode
        self._face_cascade: Optional[cv2.CascadeClassifier] = None
        self._profile_cascade: Optional[cv2.CascadeClassifier] = None
        self._eye_cascade: Optional[cv2.CascadeClassifier] = None
        self._mp_face_mesh = None
        self._face_mesh = None

        if mode == DetectionMode.HAAR:
            self._init_haar(face_cascade_path, profile_cascade_path, eye_cascade_path)
        else:
            self._init_mediapipe()

    def _init_haar(self, face_path: str, profile_path: str, eye_path: str) -> None:
        """Load Haar cascade classifiers."""
        cv_data = cv2.data.haarcascades
        face_path = face_path or f"{cv_data}haarcascade_frontalface_default.xml"
        profile_path = profile_path or f"{cv_data}haarcascade_profileface.xml"
        eye_path = eye_path or f"{cv_data}haarcascade_eye.xml"

        self._face_cascade = cv2.CascadeClassifier(face_path)
        self._profile_cascade = cv2.CascadeClassifier(profile_path)
        self._eye_cascade = cv2.CascadeClassifier(eye_path)

        if self._face_cascade.empty():
            raise RuntimeError(f"Failed to load frontal face cascade: {face_path}")
        logger.info("Haar cascades loaded successfully.")

    def _init_mediapipe(self) -> None:
        """Initialise MediaPipe FaceMesh."""
        try:
            import mediapipe as mp
            self._mp_face_mesh = mp.solutions.face_mesh
            self._face_mesh = self._mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=MEDIAPIPE_MAX_FACES,
                refine_landmarks=True,
                min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
            )
            logger.info("MediaPipe FaceMesh initialised.")
        except ImportError as exc:
            raise ImportError(
                "mediapipe is required for MEDIAPIPE mode. "
                "Install it with: pip install mediapipe"
            ) from exc

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run detection on a single frame.

        Args:
            frame: BGR (or grayscale) frame from OpenCV.

        Returns:
            DetectionResult with unified format.
        """
        if frame is None or frame.size == 0:
            return DetectionResult(mode=self.mode)

        if self.mode == DetectionMode.HAAR:
            return self._detect_haar(frame)
        return self._detect_mediapipe(frame)

    def switch_mode(self, mode: DetectionMode) -> None:
        """Switch detection backend at runtime.

        Args:
            mode: New DetectionMode to use.
        """
        if mode == self.mode:
            return
        self.mode = mode
        if mode == DetectionMode.HAAR:
            if self._face_cascade is None:
                self._init_haar("", "", "")
        else:
            if self._face_mesh is None:
                self._init_mediapipe()
        logger.info("Switched detection mode to %s", mode.name)

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None

    def __enter__(self) -> "FaceEyeDetector":
        return self

    def __exit__(self, *_) -> bool:
        self.close()
        return False

    def _detect_haar(self, frame: np.ndarray) -> DetectionResult:
        """Run Haar cascade detection.

        Args:
            frame: Input BGR frame.

        Returns:
            DetectionResult populated from Haar results.
        """
        gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if len(frame.shape) == 3
            else frame.copy()
        )
        gray = cv2.equalizeHist(gray)

        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=HAAR_SCALE_FACTOR,
            minNeighbors=HAAR_MIN_NEIGHBORS, minSize=HAAR_MIN_SIZE,
        )

        if len(faces) == 0 and not self._profile_cascade.empty():
            faces = self._profile_cascade.detectMultiScale(
                gray, scaleFactor=HAAR_SCALE_FACTOR,
                minNeighbors=HAAR_MIN_NEIGHBORS, minSize=HAAR_MIN_SIZE,
            )

        if len(faces) == 0:
            return DetectionResult(mode=DetectionMode.HAAR)

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_bbox = BoundingBox(int(x), int(y), int(w), int(h))

        roi_gray = gray[y:y + h, x:x + w]
        eyes_raw = self._eye_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20),
        )
        eye_bboxes: List[BoundingBox] = []
        for ex, ey, ew, eh in eyes_raw[:2]:
            eye_bboxes.append(BoundingBox(int(x + ex), int(y + ey), int(ew), int(eh)))

        return DetectionResult(
            face_bbox=face_bbox, eye_bboxes=eye_bboxes,
            face_detected=True, mode=DetectionMode.HAAR,
        )

    def _detect_mediapipe(self, frame: np.ndarray) -> DetectionResult:
        """Run MediaPipe FaceMesh detection.

        Args:
            frame: Input BGR frame.

        Returns:
            DetectionResult populated from MediaPipe results.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if len(frame.shape) == 3 else frame
        h, w = frame.shape[:2]

        results = self._face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return DetectionResult(mode=DetectionMode.MEDIAPIPE)

        face_landmarks = results.multi_face_landmarks[0]
        lm = face_landmarks.landmark

        all_lm: List[Tuple[float, float]] = [
            (lm[i].x * w, lm[i].y * h) for i in range(len(lm))
        ]

        xs = [p[0] for p in all_lm]
        ys = [p[1] for p in all_lm]
        bx, by = int(min(xs)), int(min(ys))
        bw, bh = int(max(xs) - bx), int(max(ys) - by)
        face_bbox = BoundingBox(bx, by, bw, bh)

        left_eye = [all_lm[i] for i in LEFT_EYE_INDICES]
        right_eye = [all_lm[i] for i in RIGHT_EYE_INDICES]
        mouth = [all_lm[i] for i in MOUTH_INDICES]

        return DetectionResult(
            face_bbox=face_bbox, eye_bboxes=[], landmarks=all_lm,
            left_eye_landmarks=left_eye, right_eye_landmarks=right_eye,
            mouth_landmarks=mouth, face_detected=True,
            mode=DetectionMode.MEDIAPIPE,
        )
