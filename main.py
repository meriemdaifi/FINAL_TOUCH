"""
main.py
=======
Main entry point for the Driver Monitoring System.

Usage:
    python main.py [--mode mediapipe|haar] [--camera 0] [--use_cnn]

Author: Daifi Meriem, Intern at Expleo Group Maroc for Stellantis
"""

import argparse
import logging
import sys
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

from src.video_acquisition import VideoAcquisition, AcquisitionConfig
from src.face_eye_detection import FaceEyeDetector, DetectionMode
from src.head_pose import HeadPoseEstimator
from src.behavioral_analysis import (
    compute_ear,
    compute_perclos,
    blink_rate,
    compute_mar,
    combine_indicators,
    EAR_CLOSED_THRESHOLD,
    EAR_BLINK_THRESHOLD,
)
from src.ecu_decision import ECUDecision, DriverIndicators
from src.visualization import Visualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EAR_HISTORY_MAXLEN: int = 150
BLINK_HISTORY_MAXLEN: int = 200


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="DMS - Driver Monitoring System")
    parser.add_argument(
        "--mode", type=str, default="mediapipe",
        choices=["mediapipe", "haar"],
        help="Detection mode (default: mediapipe)",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--use_cnn", action="store_true", help="Enable CNN inference")
    parser.add_argument(
        "--cnn_model_path", type=str, default="models/saved_model/best_model.pth",
        help="Path to CNN model weights",
    )
    parser.add_argument("--perclos_threshold", type=float, default=0.35)
    parser.add_argument("--ear_threshold", type=float, default=0.25)
    parser.add_argument("--no_face_timeout", type=float, default=3.0)
    parser.add_argument("--head_turn_timeout", type=float, default=6.0)
    return parser.parse_args()


def main() -> None:
    """Main DMS loop."""
    args = parse_args()

    # Detection mode
    det_mode = DetectionMode.MEDIAPIPE if args.mode == "mediapipe" else DetectionMode.HAAR
    use_mediapipe = det_mode == DetectionMode.MEDIAPIPE

    # Initialise modules
    acq_config = AcquisitionConfig(
        camera_index=args.camera, width=args.width, height=args.height,
    )
    detector = FaceEyeDetector(mode=det_mode)
    pose_estimator = HeadPoseEstimator(
        use_mediapipe=use_mediapipe,
        frame_width=args.width, frame_height=args.height,
    )
    ecu = ECUDecision(
        no_face_timeout=args.no_face_timeout,
        head_turn_timeout=args.head_turn_timeout,
        perclos_fatigue=args.perclos_threshold,
        ear_warning=args.ear_threshold,
    )
    visualizer = Visualizer()

    # CNN model (optional)
    cnn_model = None
    if args.use_cnn:
        try:
            from models.cnn_model import load_model
            cnn_model = load_model(args.cnn_model_path)
            logger.info("CNN model loaded from %s", args.cnn_model_path)
        except Exception as exc:
            logger.warning("CNN model not loaded: %s", exc)

    # State variables
    ear_history: deque = deque(maxlen=EAR_HISTORY_MAXLEN)
    blink_timestamps: list = []
    prev_ear: float = 0.3
    blink_in_progress: bool = False

    logger.info("Starting DMS with mode=%s, camera=%d", args.mode, args.camera)

    cam = VideoAcquisition(acq_config)
    try:
        cam.open()
        while True:
            frame, metrics = cam.read_frame()
            if frame is None:
                continue

            # 1. Detect face / eyes
            detection = detector.detect(frame)

            # 2. Compute indicators
            ear = 0.0
            mar = 0.0
            cnn_class: Optional[int] = None

            if detection.face_detected:
                if detection.left_eye_landmarks and detection.right_eye_landmarks:
                    left_ear = compute_ear(detection.left_eye_landmarks)
                    right_ear = compute_ear(detection.right_eye_landmarks)
                    ear = (left_ear + right_ear) / 2.0
                if detection.mouth_landmarks:
                    mar = compute_mar(detection.mouth_landmarks)

                # CNN inference
                if cnn_model is not None and detection.face_bbox is not None:
                    bb = detection.face_bbox
                    roi = frame[bb.y:bb.y + bb.h, bb.x:bb.x + bb.w]
                    if roi.size > 0:
                        cnn_class, _, _ = cnn_model.predict(roi)

            ear_history.append(ear)

            # Blink detection
            if ear < EAR_BLINK_THRESHOLD and not blink_in_progress:
                blink_in_progress = True
            elif ear >= EAR_BLINK_THRESHOLD and blink_in_progress:
                blink_in_progress = False
                blink_timestamps.append(time.time())

            perclos = compute_perclos(ear_history)
            br = blink_rate(blink_timestamps)

            # 3. Head pose
            head_result = pose_estimator.estimate(detection.landmarks, frame)

            # 4. ECU decision
            indicators = DriverIndicators(
                ear=ear, perclos=perclos, mar=mar,
                face_detected=detection.face_detected,
                head_distracted=head_result.is_distracted,
                cnn_class=cnn_class, blink_rate=br,
            )
            ecu.update(indicators)

            # 5. Visualize
            state_name = ecu.state.name
            face_bb = detection.face_bbox.as_tuple if detection.face_bbox else None
            eye_bbs = [eb.as_tuple for eb in detection.eye_bboxes]

            rendered = visualizer.render(
                frame,
                face_bbox=face_bb,
                eye_bboxes=eye_bbs,
                landmarks=detection.landmarks if use_mediapipe else None,
                ear=ear, perclos=perclos, mar=mar,
                state=state_name, fps=metrics.fps,
                yaw=head_result.yaw, pitch=head_result.pitch,
                warning_level=ecu.warning_level.value,
                rvec=head_result.rotation_vector,
                tvec=head_result.translation_vector,
                camera_matrix=pose_estimator._camera_matrix,
            )

            # 6. Display
            cv2.imshow("DMS - Driver Monitoring System", rendered)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("User requested quit.")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        cam.close()
        detector.close()
        cv2.destroyAllWindows()
        logger.info("DMS shutdown complete.")


if __name__ == "__main__":
    main()
