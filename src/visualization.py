"""
visualization.py
================
Visualization overlay module for the Driver Monitoring System.

Draws bounding boxes, landmarks, text overlays, head pose axes,
and color-coded status indicators on camera frames.

Author: Daifi Meriem, Intern at Expleo Group Maroc for Stellantis
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color constants (BGR)
# ---------------------------------------------------------------------------
COLOR_GREEN: Tuple[int, int, int] = (0, 255, 0)
COLOR_YELLOW: Tuple[int, int, int] = (0, 255, 255)
COLOR_RED: Tuple[int, int, int] = (0, 0, 255)
COLOR_WHITE: Tuple[int, int, int] = (255, 255, 255)
COLOR_BLACK: Tuple[int, int, int] = (0, 0, 0)
COLOR_BLUE: Tuple[int, int, int] = (255, 0, 0)
COLOR_ORANGE: Tuple[int, int, int] = (0, 165, 255)

FONT: int = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE: float = 0.6
FONT_THICKNESS: int = 2

# State -> color mapping
STATE_COLORS = {
    "ATTENTIVE": COLOR_GREEN,
    "WARNING_DROWSINESS": COLOR_YELLOW,
    "WARNING_DISTRACTION": COLOR_ORANGE,
    "FATIGUE": COLOR_RED,
    "DISTRACTED_HEAD": COLOR_RED,
    "DISTRACTED_NO_FACE": COLOR_RED,
    "EMERGENCY": COLOR_RED,
}


class Visualizer:
    """Visualization overlay engine for DMS frames.

    Args:
        show_landmarks: Draw MediaPipe landmarks on frame.
        show_dashboard: Draw info panel on the side.
        dashboard_width: Width of the side info panel in pixels.
    """

    def __init__(
        self,
        show_landmarks: bool = True,
        show_dashboard: bool = True,
        dashboard_width: int = 300,
    ) -> None:
        self.show_landmarks = show_landmarks
        self.show_dashboard = show_dashboard
        self.dashboard_width = dashboard_width
        self._flash_counter: int = 0

    def draw_face_bbox(
        self, frame: np.ndarray, bbox: Optional[Tuple[int, int, int, int]],
        color: Tuple[int, int, int] = COLOR_GREEN, thickness: int = 2,
    ) -> np.ndarray:
        """Draw face bounding box on the frame.

        Args:
            frame: BGR image.
            bbox: (x, y, w, h) or None.
            color: BGR color tuple.
            thickness: Line thickness.

        Returns:
            Annotated frame.
        """
        if bbox is None:
            return frame
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        return frame

    def draw_eye_bboxes(
        self, frame: np.ndarray, bboxes: List[Tuple[int, int, int, int]],
        color: Tuple[int, int, int] = COLOR_BLUE, thickness: int = 1,
    ) -> np.ndarray:
        """Draw eye bounding boxes on the frame.

        Args:
            frame: BGR image.
            bboxes: List of (x, y, w, h) tuples.
            color: BGR color tuple.
            thickness: Line thickness.

        Returns:
            Annotated frame.
        """
        for x, y, w, h in bboxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        return frame

    def draw_landmarks(
        self, frame: np.ndarray, landmarks: List[Tuple[float, float]],
        color: Tuple[int, int, int] = COLOR_GREEN, radius: int = 1,
    ) -> np.ndarray:
        """Draw MediaPipe landmarks on the frame.

        Args:
            frame: BGR image.
            landmarks: List of (x, y) pixel coordinates.
            color: BGR color tuple.
            radius: Circle radius.

        Returns:
            Annotated frame.
        """
        if not self.show_landmarks:
            return frame
        for x, y in landmarks:
            cv2.circle(frame, (int(x), int(y)), radius, color, -1)
        return frame

    def draw_text_overlay(
        self,
        frame: np.ndarray,
        ear: float = 0.0,
        perclos: float = 0.0,
        state: str = "ATTENTIVE",
        fps: float = 0.0,
        mar: float = 0.0,
    ) -> np.ndarray:
        """Overlay EAR, PERCLOS, state, and FPS text on the frame.

        Args:
            frame: BGR image.
            ear: Current EAR value.
            perclos: Current PERCLOS percentage.
            state: Current driver state string.
            fps: Current FPS.
            mar: Current MAR value.

        Returns:
            Annotated frame.
        """
        color = STATE_COLORS.get(state, COLOR_WHITE)
        texts = [
            f"State: {state}",
            f"EAR: {ear:.3f}",
            f"PERCLOS: {perclos:.1%}",
            f"MAR: {mar:.3f}",
            f"FPS: {fps:.1f}",
        ]
        y_offset = 30
        for text in texts:
            cv2.putText(frame, text, (10, y_offset), FONT, FONT_SCALE, color, FONT_THICKNESS)
            y_offset += 30
        return frame

    def draw_head_pose_axes(
        self,
        frame: np.ndarray,
        nose_point: Tuple[int, int],
        rvec: Optional[np.ndarray] = None,
        tvec: Optional[np.ndarray] = None,
        camera_matrix: Optional[np.ndarray] = None,
        axis_length: float = 50.0,
    ) -> np.ndarray:
        """Draw 3D head pose axes on the frame.

        Args:
            frame: BGR image.
            nose_point: (x, y) of the nose tip on the image.
            rvec: Rotation vector from solvePnP.
            tvec: Translation vector from solvePnP.
            camera_matrix: Camera intrinsic matrix.
            axis_length: Length of drawn axes in pixels.

        Returns:
            Annotated frame.
        """
        if rvec is None or tvec is None or camera_matrix is None:
            return frame

        dist_coeffs = np.zeros((4, 1))
        axis_points = np.array([
            [axis_length, 0, 0],
            [0, axis_length, 0],
            [0, 0, axis_length],
        ], dtype=np.float64)

        img_pts, _ = cv2.projectPoints(
            axis_points, rvec, tvec, camera_matrix, dist_coeffs
        )
        origin = tuple(int(c) for c in nose_point)
        x_end = tuple(int(c) for c in img_pts[0].ravel())
        y_end = tuple(int(c) for c in img_pts[1].ravel())
        z_end = tuple(int(c) for c in img_pts[2].ravel())

        cv2.line(frame, origin, x_end, COLOR_RED, 2)    # X = Red
        cv2.line(frame, origin, y_end, COLOR_GREEN, 2)   # Y = Green
        cv2.line(frame, origin, z_end, COLOR_BLUE, 2)    # Z = Blue

        return frame

    def draw_warning_overlay(
        self, frame: np.ndarray, state: str, border_width: int = 15,
    ) -> np.ndarray:
        """Draw red border flash for emergency / danger states.

        Args:
            frame: BGR image.
            state: Current driver state string.
            border_width: Width of the warning border.

        Returns:
            Annotated frame.
        """
        danger_states = {"FATIGUE", "DISTRACTED_HEAD", "DISTRACTED_NO_FACE", "EMERGENCY"}
        if state not in danger_states:
            self._flash_counter = 0
            return frame

        self._flash_counter += 1
        if self._flash_counter % 4 < 2:  # Flash effect
            h, w = frame.shape[:2]
            color = COLOR_RED if state == "EMERGENCY" else COLOR_ORANGE
            cv2.rectangle(frame, (0, 0), (w, h), color, border_width)

        return frame

    def draw_dashboard(
        self,
        frame: np.ndarray,
        ear: float = 0.0,
        perclos: float = 0.0,
        mar: float = 0.0,
        state: str = "ATTENTIVE",
        fps: float = 0.0,
        yaw: float = 0.0,
        pitch: float = 0.0,
        warning_level: int = 0,
    ) -> np.ndarray:
        """Attach an info dashboard panel to the right side of the frame.

        Args:
            frame: BGR image.
            ear: Current EAR value.
            perclos: Current PERCLOS percentage.
            mar: Current MAR value.
            state: Current driver state string.
            fps: Current FPS.
            yaw: Head yaw angle (degrees).
            pitch: Head pitch angle (degrees).
            warning_level: Current warning level (0-3).

        Returns:
            Frame with dashboard panel appended.
        """
        if not self.show_dashboard:
            return frame

        h, w = frame.shape[:2]
        dw = self.dashboard_width
        dashboard = np.zeros((h, dw, 3), dtype=np.uint8)
        dashboard[:] = (40, 40, 40)

        color = STATE_COLORS.get(state, COLOR_WHITE)
        lines = [
            ("DMS Dashboard", COLOR_WHITE, 0.7),
            ("", COLOR_WHITE, 0.5),
            (f"State: {state}", color, 0.5),
            (f"Warning Lv: {warning_level}", color, 0.5),
            ("", COLOR_WHITE, 0.5),
            (f"EAR:     {ear:.3f}", COLOR_WHITE, 0.5),
            (f"PERCLOS: {perclos:.1%}", COLOR_WHITE, 0.5),
            (f"MAR:     {mar:.3f}", COLOR_WHITE, 0.5),
            ("", COLOR_WHITE, 0.5),
            (f"Yaw:   {yaw:.1f} deg", COLOR_WHITE, 0.5),
            (f"Pitch: {pitch:.1f} deg", COLOR_WHITE, 0.5),
            ("", COLOR_WHITE, 0.5),
            (f"FPS: {fps:.1f}", COLOR_WHITE, 0.5),
        ]

        y_pos = 30
        for text, clr, scale in lines:
            if text:
                cv2.putText(dashboard, text, (10, y_pos), FONT, scale, clr, 1)
            y_pos += 28

        # PERCLOS bar
        bar_y = y_pos + 10
        bar_h = 20
        cv2.rectangle(dashboard, (10, bar_y), (dw - 10, bar_y + bar_h), COLOR_WHITE, 1)
        fill_w = int((dw - 20) * min(perclos, 1.0))
        bar_color = COLOR_GREEN if perclos < 0.2 else COLOR_YELLOW if perclos < 0.35 else COLOR_RED
        cv2.rectangle(dashboard, (10, bar_y), (10 + fill_w, bar_y + bar_h), bar_color, -1)
        cv2.putText(dashboard, "PERCLOS", (10, bar_y - 5), FONT, 0.4, COLOR_WHITE, 1)

        return np.hstack([frame, dashboard])

    def render(
        self,
        frame: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
        eye_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
        landmarks: Optional[List[Tuple[float, float]]] = None,
        ear: float = 0.0,
        perclos: float = 0.0,
        mar: float = 0.0,
        state: str = "ATTENTIVE",
        fps: float = 0.0,
        yaw: float = 0.0,
        pitch: float = 0.0,
        warning_level: int = 0,
        rvec: Optional[np.ndarray] = None,
        tvec: Optional[np.ndarray] = None,
        camera_matrix: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Full render pipeline: all overlays and dashboard.

        Args:
            frame: BGR image.
            face_bbox: Face bounding box (x, y, w, h).
            eye_bboxes: Eye bounding boxes.
            landmarks: MediaPipe landmarks.
            ear: Current EAR.
            perclos: Current PERCLOS.
            mar: Current MAR.
            state: Current driver state string.
            fps: Current FPS.
            yaw: Head yaw angle.
            pitch: Head pitch angle.
            warning_level: Warning level (0-3).
            rvec: Rotation vector.
            tvec: Translation vector.
            camera_matrix: Camera intrinsic matrix.

        Returns:
            Fully annotated frame with dashboard.
        """
        out = frame.copy()
        state_color = STATE_COLORS.get(state, COLOR_GREEN)

        out = self.draw_face_bbox(out, face_bbox, color=state_color)
        if eye_bboxes:
            out = self.draw_eye_bboxes(out, eye_bboxes)
        if landmarks:
            out = self.draw_landmarks(out, landmarks)
            if rvec is not None and len(landmarks) > 1:
                nose = (int(landmarks[1][0]), int(landmarks[1][1]))
                out = self.draw_head_pose_axes(out, nose, rvec, tvec, camera_matrix)

        out = self.draw_text_overlay(out, ear, perclos, state, fps, mar)
        out = self.draw_warning_overlay(out, state)
        out = self.draw_dashboard(out, ear, perclos, mar, state, fps, yaw, pitch, warning_level)

        return out
