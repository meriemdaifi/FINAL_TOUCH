"""
behavioral_analysis.py
======================
Behavioral indicator computation for the Driver Monitoring System.

Functions:
  - compute_ear: Eye Aspect Ratio (6-point formula)
  - compute_perclos: Percentage of Eye Closure (sliding window)
  - blink_rate: Blinks per minute
  - compute_mar: Mouth Aspect Ratio for yawn detection
  - combine_indicators: Fuse CNN + geometric cues

Author: Daifi Meriem, Intern at Expleo Group Maroc for Stellantis
"""

import logging
import time
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EAR_BLINK_THRESHOLD: float = 0.21
EAR_CLOSED_THRESHOLD: float = 0.25
MAR_YAWN_THRESHOLD: float = 0.6
PERCLOS_FATIGUE_THRESHOLD: float = 0.35
PERCLOS_WINDOW_SECONDS: float = 60.0
BLINK_RATE_WINDOW_SECONDS: float = 60.0
CNN_CLOSED_CLASS: int = 1
CNN_YAWNING_CLASS: int = 2


def _dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Euclidean distance between two 2-D points.

    Args:
        p1: First point (x, y).
        p2: Second point (x, y).

    Returns:
        Euclidean distance as float.
    """
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def compute_ear(landmarks: List[Tuple[float, float]]) -> float:
    """Compute Eye Aspect Ratio (EAR) from 6 eye landmarks.

    Standard formula:
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Landmark order:
        0=outer corner, 1=upper-outer, 2=upper-inner,
        3=inner corner, 4=lower-inner, 5=lower-outer

    Args:
        landmarks: List of exactly 6 (x, y) tuples.

    Returns:
        EAR value in [0, ~0.5]. Returns 0.0 if landmarks are invalid.
    """
    if len(landmarks) < 6:
        logger.debug("compute_ear: insufficient landmarks (%d)", len(landmarks))
        return 0.0

    p1, p2, p3, p4, p5, p6 = landmarks[:6]
    vertical1 = _dist(p2, p6)
    vertical2 = _dist(p3, p5)
    horizontal = _dist(p1, p4)

    if horizontal < 1e-6:
        return 0.0

    return (vertical1 + vertical2) / (2.0 * horizontal)


def compute_perclos(
    ear_history: Deque[float],
    threshold: float = EAR_CLOSED_THRESHOLD,
) -> float:
    """Compute PERCLOS (Percentage of Eye Closure) over a sliding window.

    PERCLOS is the fraction of frames where EAR < threshold.

    Args:
        ear_history: Deque of recent EAR values.
        threshold: EAR below which eye is considered closed.

    Returns:
        PERCLOS value in [0.0, 1.0].
    """
    if len(ear_history) == 0:
        return 0.0
    closed_count = sum(1 for ear in ear_history if ear < threshold)
    return closed_count / len(ear_history)


def blink_rate(
    blink_timestamps: List[float],
    window_seconds: float = BLINK_RATE_WINDOW_SECONDS,
) -> float:
    """Compute blinks per minute within a time window.

    Args:
        blink_timestamps: List of epoch timestamps for each blink.
        window_seconds: Time window to consider (default 60 s).

    Returns:
        Blinks per minute as float.
    """
    if not blink_timestamps:
        return 0.0
    now = time.time()
    recent = [t for t in blink_timestamps if now - t <= window_seconds]
    return len(recent) * (60.0 / window_seconds)


def compute_mar(landmarks: List[Tuple[float, float]]) -> float:
    """Compute Mouth Aspect Ratio (MAR) for yawn detection.

    Uses simplified formula:
        MAR = (||p2-p8|| + ||p4-p6||) / (2 * ||p1-p5||)

    Expects at least 8 landmarks:
        0=left corner, 1=upper-left, 2=upper-mid, 3=upper-right,
        4=right corner, 5=lower-right, 6=lower-mid, 7=lower-left

    Args:
        landmarks: List of at least 8 (x, y) tuples.

    Returns:
        MAR value. Returns 0.0 if landmarks are invalid.
    """
    if len(landmarks) < 8:
        logger.debug("compute_mar: insufficient landmarks (%d)", len(landmarks))
        return 0.0

    p1 = landmarks[0]
    p2 = landmarks[2]
    p4 = landmarks[3]
    p5 = landmarks[4]
    p6 = landmarks[5]
    p8 = landmarks[7]

    vertical1 = _dist(p2, p8)
    vertical2 = _dist(p4, p6)
    horizontal = _dist(p1, p5)

    if horizontal < 1e-6:
        return 0.0

    return (vertical1 + vertical2) / (2.0 * horizontal)


def combine_indicators(
    ear: float,
    cnn_prediction: Optional[int],
    perclos: float,
    mar: float,
    ear_threshold: float = EAR_CLOSED_THRESHOLD,
    mar_threshold: float = MAR_YAWN_THRESHOLD,
    perclos_threshold: float = PERCLOS_FATIGUE_THRESHOLD,
) -> dict:
    """Fuse CNN prediction with geometric indicators.

    Args:
        ear: Current Eye Aspect Ratio.
        cnn_prediction: CNN class (0=OPEN, 1=CLOSED, 2=YAWNING) or None.
        perclos: Current PERCLOS value [0, 1].
        mar: Current Mouth Aspect Ratio.
        ear_threshold: EAR below which eye is closed.
        mar_threshold: MAR above which yawning.
        perclos_threshold: PERCLOS above which fatigue.

    Returns:
        Dict with 'eye_closed', 'yawning', 'drowsy', 'confidence'.
    """
    eye_closed_geo = ear < ear_threshold
    eye_closed_cnn = cnn_prediction == CNN_CLOSED_CLASS if cnn_prediction is not None else False
    yawning_geo = mar > mar_threshold
    yawning_cnn = cnn_prediction == CNN_YAWNING_CLASS if cnn_prediction is not None else False

    eye_closed = eye_closed_geo or eye_closed_cnn
    yawning = yawning_geo or yawning_cnn
    fatigue = perclos > perclos_threshold
    drowsy = fatigue or (eye_closed and yawning)

    active = sum([eye_closed_geo, eye_closed_cnn, yawning, fatigue])
    confidence = min(active / 4.0, 1.0)

    return {
        "eye_closed": eye_closed,
        "yawning": yawning,
        "drowsy": drowsy,
        "confidence": confidence,
    }
