"""
video_acquisition.py
====================
Real-time video acquisition module for the Driver Monitoring System.

Provides webcam capture with configurable resolution, FPS targeting,
frame preprocessing and performance metrics.

Author: Daifi Meriem, Intern at Expleo Group Maroc for Stellantis
"""

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Generator, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CAMERA_INDEX: int = 0
DEFAULT_WIDTH: int = 640
DEFAULT_HEIGHT: int = 480
DEFAULT_FPS: int = 30
TARGET_FPS: int = 20
MAX_LATENCY_MS: float = 100.0
FPS_HISTORY_LEN: int = 30


@dataclass
class AcquisitionConfig:
    """Configuration for video acquisition.

    Args:
        camera_index: OpenCV camera index (0 = default webcam).
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Requested capture frame rate.
        target_width: Width after preprocessing resize (0 = no resize).
        target_height: Height after preprocessing resize (0 = no resize).
        convert_to_gray: If True, convert frames to grayscale.
    """
    camera_index: int = DEFAULT_CAMERA_INDEX
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    fps: int = DEFAULT_FPS
    target_width: int = 0
    target_height: int = 0
    convert_to_gray: bool = False


@dataclass
class FrameMetrics:
    """Per-frame performance metrics.

    Args:
        timestamp: Epoch time when frame was captured.
        capture_latency_ms: Time from capture start to return (ms).
        fps: Instantaneous FPS estimate.
        frame_index: Sequential frame number since start.
    """
    timestamp: float = 0.0
    capture_latency_ms: float = 0.0
    fps: float = 0.0
    frame_index: int = 0


class VideoAcquisition:
    """Real-time webcam capture with preprocessing and performance tracking.

    Supports context-manager usage::

        with VideoAcquisition(config) as cam:
            for frame, metrics in cam:
                process(frame)

    Args:
        config: AcquisitionConfig instance.
    """

    def __init__(self, config: Optional[AcquisitionConfig] = None) -> None:
        self._config: AcquisitionConfig = config or AcquisitionConfig()
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_index: int = 0
        self._ts_history: deque = deque(maxlen=FPS_HISTORY_LEN)
        self._running: bool = False

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open the video capture device."""
        cfg = self._config
        self._cap = cv2.VideoCapture(cfg.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera index {cfg.camera_index}"
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        self._cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        self._running = True
        logger.info(
            "Camera %d opened: %dx%d @ %d FPS",
            cfg.camera_index, cfg.width, cfg.height, cfg.fps,
        )

    def close(self) -> None:
        """Release the video capture device."""
        self._running = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera released.")

    def __enter__(self) -> "VideoAcquisition":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    # ------------------------------------------------------------------
    # Frame capture
    # ------------------------------------------------------------------

    def read_frame(self) -> Tuple[Optional[np.ndarray], FrameMetrics]:
        """Capture and preprocess a single frame.

        Returns:
            Tuple of (preprocessed frame or None on failure, FrameMetrics).
        """
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("Camera is not open. Call open() first.")

        t_start = time.monotonic()
        ret, frame = self._cap.read()
        t_end = time.monotonic()

        if not ret or frame is None:
            logger.warning("Failed to read frame #%d", self._frame_index)
            metrics = FrameMetrics(
                timestamp=time.time(),
                capture_latency_ms=(t_end - t_start) * 1000,
                fps=0.0,
                frame_index=self._frame_index,
            )
            return None, metrics

        # Preprocessing
        frame = self._preprocess(frame)

        # Metrics
        now = time.time()
        self._ts_history.append(now)
        fps = self._compute_fps()
        latency_ms = (t_end - t_start) * 1000

        if latency_ms > MAX_LATENCY_MS:
            logger.warning(
                "Frame latency %.1f ms exceeds %.1f ms budget",
                latency_ms, MAX_LATENCY_MS,
            )

        metrics = FrameMetrics(
            timestamp=now,
            capture_latency_ms=latency_ms,
            fps=fps,
            frame_index=self._frame_index,
        )
        self._frame_index += 1
        return frame, metrics

    # ------------------------------------------------------------------
    # Iterator interface
    # ------------------------------------------------------------------

    def __iter__(self) -> Generator[Tuple[np.ndarray, FrameMetrics], None, None]:
        """Iterate over captured frames until the camera is closed."""
        while self._running:
            frame, metrics = self.read_frame()
            if frame is None:
                continue
            yield frame, metrics

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Apply resize and color conversion as configured.

        Args:
            frame: Raw BGR frame from OpenCV.

        Returns:
            Preprocessed frame.
        """
        cfg = self._config
        if cfg.target_width > 0 and cfg.target_height > 0:
            frame = cv2.resize(
                frame, (cfg.target_width, cfg.target_height),
                interpolation=cv2.INTER_LINEAR,
            )
        if cfg.convert_to_gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def _compute_fps(self) -> float:
        """Estimate current FPS from recent frame timestamps.

        Returns:
            Frames per second as a float.
        """
        if len(self._ts_history) < 2:
            return 0.0
        elapsed = self._ts_history[-1] - self._ts_history[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._ts_history) - 1) / elapsed

    @property
    def fps(self) -> float:
        """Current estimated FPS."""
        return self._compute_fps()

    @property
    def frame_index(self) -> int:
        """Total frames captured since open()."""
        return self._frame_index

    @property
    def is_open(self) -> bool:
        """True if the capture device is currently open."""
        return self._cap is not None and self._cap.isOpened()
