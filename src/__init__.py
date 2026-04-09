"""
Driver Monitoring System (DMS) - Source Package
================================================
Industrial-grade DMS for Stellantis / Expleo Group Maroc.
Author: Daifi Meriem, Intern at Expleo Group Maroc
"""

from .video_acquisition import VideoAcquisition
from .face_eye_detection import FaceEyeDetector, DetectionResult, DetectionMode
from .head_pose import HeadPoseEstimator, HeadPoseResult
from .behavioral_analysis import (
    compute_ear,
    compute_perclos,
    blink_rate,
    compute_mar,
    combine_indicators,
)
from .ecu_decision import ECUDecision, DriverState, WarningLevel
from .scenarios import (
    NormalDrivingScenario,
    GradualDrowsinessScenario,
    SuddenDistractionScenario,
    FaceLostScenario,
    YawningScenario,
    CombinedFatigueScenario,
    EmergencyScenario,
)
from .visualization import Visualizer

__all__ = [
    "VideoAcquisition",
    "FaceEyeDetector",
    "DetectionResult",
    "DetectionMode",
    "HeadPoseEstimator",
    "HeadPoseResult",
    "compute_ear",
    "compute_perclos",
    "blink_rate",
    "compute_mar",
    "combine_indicators",
    "ECUDecision",
    "DriverState",
    "WarningLevel",
    "NormalDrivingScenario",
    "GradualDrowsinessScenario",
    "SuddenDistractionScenario",
    "FaceLostScenario",
    "YawningScenario",
    "CombinedFatigueScenario",
    "EmergencyScenario",
    "Visualizer",
]
