"""
test_dms.py
===========
Comprehensive unit tests for the Driver Monitoring System.

30+ test functions covering all DMS modules.

Author: Daifi Meriem, Intern at Expleo Group Maroc for Stellantis
"""

import time
from collections import deque
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# ── src modules ──────────────────────────────────────────────────────────
from src.behavioral_analysis import (
    EAR_CLOSED_THRESHOLD,
    MAR_YAWN_THRESHOLD,
    PERCLOS_FATIGUE_THRESHOLD,
    blink_rate,
    combine_indicators,
    compute_ear,
    compute_mar,
    compute_perclos,
)
from src.ecu_decision import (
    ActionCommand,
    DriverIndicators,
    DriverState,
    ECUDecision,
    WarningLevel,
)
from src.face_eye_detection import (
    BoundingBox,
    DetectionMode,
    DetectionResult,
)
from src.head_pose import HeadPoseEstimator, HeadPoseResult
from src.scenarios import (
    CombinedFatigueScenario,
    EmergencyScenario,
    FaceLostScenario,
    GradualDrowsinessScenario,
    NormalDrivingScenario,
    SuddenDistractionScenario,
    YawningScenario,
)
from src.video_acquisition import AcquisitionConfig, FrameMetrics, VideoAcquisition
from src.visualization import Visualizer

# ── models ───────────────────────────────────────────────────────────────
from models.cnn_model import DrowsinessCNN, INPUT_SIZE, NUM_CLASSES


# =========================================================================
# EAR computation tests
# =========================================================================

class TestComputeEAR:
    """Tests for compute_ear function."""

    def test_ear_open_eye(self) -> None:
        """Open eye landmarks should give EAR ~ 0.3."""
        landmarks = [
            (0.0, 0.0),    # outer corner
            (1.0, 1.0),    # upper-outer
            (2.0, 1.0),    # upper-inner
            (3.0, 0.0),    # inner corner
            (2.0, -1.0),   # lower-inner
            (1.0, -1.0),   # lower-outer
        ]
        ear = compute_ear(landmarks)
        assert ear > 0.2, f"EAR for open eye should be > 0.2, got {ear}"

    def test_ear_closed_eye(self) -> None:
        """Closed eye landmarks should give EAR ~ 0."""
        landmarks = [
            (0.0, 0.0),
            (1.0, 0.01),
            (2.0, 0.01),
            (3.0, 0.0),
            (2.0, -0.01),
            (1.0, -0.01),
        ]
        ear = compute_ear(landmarks)
        assert ear < 0.1, f"EAR for closed eye should be < 0.1, got {ear}"

    def test_ear_insufficient_landmarks(self) -> None:
        """Less than 6 landmarks should return 0.0."""
        assert compute_ear([(0, 0), (1, 1)]) == 0.0

    def test_ear_empty_landmarks(self) -> None:
        """Empty list should return 0.0."""
        assert compute_ear([]) == 0.0

    def test_ear_horizontal_zero(self) -> None:
        """Same outer/inner corners should return 0.0 (no division error)."""
        landmarks = [(0, 0)] * 6
        ear = compute_ear(landmarks)
        assert ear == 0.0

    def test_ear_value_range(self) -> None:
        """EAR should be non-negative."""
        landmarks = [
            (10, 20), (15, 25), (20, 25),
            (25, 20), (20, 15), (15, 15),
        ]
        ear = compute_ear(landmarks)
        assert ear >= 0.0


# =========================================================================
# PERCLOS computation tests
# =========================================================================

class TestComputePERCLOS:
    """Tests for compute_perclos function."""

    def test_all_open(self) -> None:
        """All-open EAR history should give PERCLOS = 0."""
        history = deque([0.30] * 100, maxlen=100)
        assert compute_perclos(history) == 0.0

    def test_all_closed(self) -> None:
        """All-closed EAR history should give PERCLOS = 1."""
        history = deque([0.10] * 100, maxlen=100)
        assert compute_perclos(history) == 1.0

    def test_half_closed(self) -> None:
        """50% closed should give PERCLOS = 0.5."""
        data = [0.30] * 50 + [0.10] * 50
        history = deque(data, maxlen=100)
        assert abs(compute_perclos(history) - 0.5) < 0.01

    def test_empty_history(self) -> None:
        """Empty history should return 0.0."""
        assert compute_perclos(deque()) == 0.0

    def test_custom_threshold(self) -> None:
        """Custom threshold should affect PERCLOS calculation."""
        history = deque([0.20] * 100, maxlen=100)
        # All 0.20 are below 0.25 threshold => PERCLOS = 1.0
        assert compute_perclos(history, threshold=0.25) == 1.0
        # All 0.20 are above 0.15 threshold => PERCLOS = 0.0
        assert compute_perclos(history, threshold=0.15) == 0.0


# =========================================================================
# MAR computation tests
# =========================================================================

class TestComputeMAR:
    """Tests for compute_mar function."""

    def test_closed_mouth(self) -> None:
        """Closed mouth should give low MAR."""
        landmarks = [
            (0, 0), (1, 0.1), (2, 0.1), (3, 0.1),
            (4, 0), (3, -0.1), (2, -0.1), (1, -0.1),
        ]
        mar = compute_mar(landmarks)
        assert mar < 0.3

    def test_open_mouth_yawning(self) -> None:
        """Wide open mouth should give high MAR."""
        landmarks = [
            (0, 0), (1, 3), (2, 4), (3, 3),
            (4, 0), (3, -3), (2, -4), (1, -3),
        ]
        mar = compute_mar(landmarks)
        assert mar > 0.5

    def test_insufficient_landmarks(self) -> None:
        """Fewer than 8 landmarks should return 0.0."""
        assert compute_mar([(0, 0)] * 5) == 0.0


# =========================================================================
# Blink rate tests
# =========================================================================

class TestBlinkRate:
    """Tests for blink_rate function."""

    def test_no_blinks(self) -> None:
        """No blinks should return 0."""
        assert blink_rate([]) == 0.0

    def test_blinks_in_window(self) -> None:
        """Recent blinks should be counted."""
        now = time.time()
        timestamps = [now - i for i in range(10)]
        rate = blink_rate(timestamps, window_seconds=60.0)
        assert rate > 0

    def test_old_blinks_excluded(self) -> None:
        """Blinks older than window should be excluded."""
        now = time.time()
        timestamps = [now - 120]  # 2 minutes ago
        rate = blink_rate(timestamps, window_seconds=60.0)
        assert rate == 0.0


# =========================================================================
# Combine indicators tests
# =========================================================================

class TestCombineIndicators:
    """Tests for combine_indicators function."""

    def test_all_normal(self) -> None:
        """Normal indicators: not drowsy."""
        result = combine_indicators(ear=0.30, cnn_prediction=0, perclos=0.05, mar=0.2)
        assert result["drowsy"] is False
        assert result["eye_closed"] is False

    def test_drowsy_combination(self) -> None:
        """High PERCLOS should flag drowsy."""
        result = combine_indicators(ear=0.15, cnn_prediction=1, perclos=0.40, mar=0.3)
        assert result["drowsy"] is True
        assert result["eye_closed"] is True

    def test_yawning_detected(self) -> None:
        """High MAR should flag yawning."""
        result = combine_indicators(ear=0.30, cnn_prediction=2, perclos=0.05, mar=0.7)
        assert result["yawning"] is True

    def test_cnn_none(self) -> None:
        """CNN=None should still work with geometric indicators."""
        result = combine_indicators(ear=0.15, cnn_prediction=None, perclos=0.05, mar=0.2)
        assert result["eye_closed"] is True


# =========================================================================
# CNN model tests
# =========================================================================

class TestCNNModel:
    """Tests for DrowsinessCNN model."""

    def test_model_output_shape(self) -> None:
        """Output should be (batch, NUM_CLASSES)."""
        model = DrowsinessCNN()
        x = torch.randn(4, 1, INPUT_SIZE, INPUT_SIZE)
        out = model(x)
        assert out.shape == (4, NUM_CLASSES)

    def test_model_single_image(self) -> None:
        """Single image forward pass."""
        model = DrowsinessCNN()
        x = torch.randn(1, 1, INPUT_SIZE, INPUT_SIZE)
        out = model(x)
        assert out.shape == (1, NUM_CLASSES)

    def test_model_predict(self) -> None:
        """predict() should return valid class index and name."""
        model = DrowsinessCNN()
        img = np.random.randint(0, 255, (INPUT_SIZE, INPUT_SIZE), dtype=np.uint8)
        idx, conf, name = model.predict(img)
        assert 0 <= idx < NUM_CLASSES
        assert 0.0 <= conf <= 1.0
        assert name in ("OPEN", "CLOSED", "YAWNING")

    def test_model_summary(self) -> None:
        """summary() should return a non-empty string."""
        model = DrowsinessCNN()
        s = model.summary()
        assert len(s) > 0
        assert "DrowsinessCNN" in s

    def test_model_different_input_size(self) -> None:
        """Model should work with different input sizes."""
        model = DrowsinessCNN(input_size=64)
        x = torch.randn(2, 1, 64, 64)
        out = model(x)
        assert out.shape == (2, NUM_CLASSES)


# =========================================================================
# ECU Decision tests
# =========================================================================

class TestECUDecision:
    """Tests for ECU FSM."""

    def test_initial_state_attentive(self) -> None:
        """FSM starts in ATTENTIVE state."""
        ecu = ECUDecision()
        assert ecu.state == DriverState.ATTENTIVE

    def test_transition_to_warning_drowsiness(self) -> None:
        """Low EAR should trigger WARNING_DROWSINESS after hysteresis."""
        ecu = ECUDecision(hysteresis_frames=2)
        ind = DriverIndicators(ear=0.15, perclos=0.0, mar=0.0)
        for _ in range(5):
            ecu.update(ind)
        assert ecu.state == DriverState.WARNING_DROWSINESS

    def test_transition_to_fatigue(self) -> None:
        """High PERCLOS should trigger FATIGUE after hysteresis."""
        ecu = ECUDecision(hysteresis_frames=2)
        ind = DriverIndicators(ear=0.30, perclos=0.40, mar=0.0)
        for _ in range(5):
            ecu.update(ind)
        assert ecu.state == DriverState.FATIGUE

    def test_no_face_distraction(self) -> None:
        """No face for > timeout should trigger DISTRACTED_NO_FACE."""
        ecu = ECUDecision(no_face_timeout=0.1, hysteresis_frames=1)
        ind = DriverIndicators(face_detected=False)
        time.sleep(0.15)
        for _ in range(3):
            ecu.update(ind)
        assert ecu.state == DriverState.DISTRACTED_NO_FACE

    def test_head_distraction(self) -> None:
        """Head turned > timeout triggers DISTRACTED_HEAD."""
        ecu = ECUDecision(head_turn_timeout=0.1, hysteresis_frames=1)
        ind = DriverIndicators(head_distracted=True)
        time.sleep(0.15)
        for _ in range(3):
            ecu.update(ind)
        assert ecu.state == DriverState.DISTRACTED_HEAD

    def test_return_to_attentive(self) -> None:
        """Normal indicators should eventually return to ATTENTIVE."""
        ecu = ECUDecision(hysteresis_frames=2)
        # First go to warning
        warn_ind = DriverIndicators(ear=0.15)
        for _ in range(5):
            ecu.update(warn_ind)
        assert ecu.state != DriverState.ATTENTIVE

        # Then recover
        normal_ind = DriverIndicators(ear=0.30, perclos=0.0, mar=0.0)
        for _ in range(10):
            ecu.update(normal_ind)
        assert ecu.state == DriverState.ATTENTIVE

    def test_warning_level_1_action(self) -> None:
        """LEVEL_1 should trigger audio + visual alerts."""
        ecu = ECUDecision(hysteresis_frames=1)
        ind = DriverIndicators(ear=0.15)
        for _ in range(3):
            ecu.update(ind)
        action = ecu.action
        assert action.audio_alert is True
        assert action.visual_alert is True
        assert action.emergency_braking is False

    def test_warning_level_2_action(self) -> None:
        """LEVEL_2 should trigger vibrations."""
        ecu = ECUDecision(hysteresis_frames=1)
        ind = DriverIndicators(perclos=0.40)
        for _ in range(3):
            ecu.update(ind)
        action = ecu.action
        assert action.seat_vibration is True
        assert action.steering_vibration is True

    def test_reset(self) -> None:
        """reset() should return to ATTENTIVE."""
        ecu = ECUDecision(hysteresis_frames=1)
        ind = DriverIndicators(ear=0.15)
        for _ in range(3):
            ecu.update(ind)
        ecu.reset()
        assert ecu.state == DriverState.ATTENTIVE
        assert ecu.warning_level == WarningLevel.NONE

    def test_hysteresis_prevents_flicker(self) -> None:
        """State should not change until hysteresis threshold is met."""
        ecu = ECUDecision(hysteresis_frames=5)
        # Send 3 warning frames then 1 normal => should still be attentive
        warn = DriverIndicators(ear=0.15)
        normal = DriverIndicators(ear=0.30)
        for _ in range(3):
            ecu.update(warn)
        ecu.update(normal)
        assert ecu.state == DriverState.ATTENTIVE


# =========================================================================
# Head pose tests
# =========================================================================

class TestHeadPose:
    """Tests for HeadPoseEstimator."""

    def test_haar_proxy_no_frame(self) -> None:
        """No frame and no landmarks should return defaults."""
        est = HeadPoseEstimator(use_mediapipe=False)
        result = est.estimate([], frame=None)
        assert result.yaw == 0.0
        assert result.is_distracted is False

    def test_pnp_with_landmarks(self) -> None:
        """solvePnP with synthetic landmarks should return valid angles."""
        est = HeadPoseEstimator(use_mediapipe=True, frame_width=640, frame_height=480)
        # Create 300 dummy landmarks (enough for max index 287)
        landmarks = [(320 + i * 0.1, 240 + i * 0.1) for i in range(300)]
        result = est.estimate(landmarks)
        # Should return some angle values
        assert isinstance(result.yaw, float)
        assert isinstance(result.pitch, float)

    def test_update_frame_size(self) -> None:
        """update_frame_size should update the camera matrix."""
        est = HeadPoseEstimator()
        est.update_frame_size(1920, 1080)
        assert est._camera_matrix[0, 0] == 1920


# =========================================================================
# Detection result tests
# =========================================================================

class TestDetectionResult:
    """Tests for DetectionResult dataclass."""

    def test_default_no_face(self) -> None:
        """Default DetectionResult should have no face."""
        result = DetectionResult()
        assert result.face_detected is False
        assert result.face_bbox is None
        assert len(result.landmarks) == 0

    def test_bounding_box(self) -> None:
        """BoundingBox properties should work correctly."""
        bb = BoundingBox(10, 20, 100, 200)
        assert bb.as_tuple == (10, 20, 100, 200)
        assert bb.center == (60, 120)

    def test_mode_switching(self) -> None:
        """DetectionResult should track its mode."""
        r1 = DetectionResult(mode=DetectionMode.HAAR)
        assert r1.mode == DetectionMode.HAAR
        r2 = DetectionResult(mode=DetectionMode.MEDIAPIPE)
        assert r2.mode == DetectionMode.MEDIAPIPE


# =========================================================================
# Video acquisition config tests
# =========================================================================

class TestVideoAcquisition:
    """Tests for VideoAcquisition."""

    def test_default_config(self) -> None:
        """Default config should have expected values."""
        cfg = AcquisitionConfig()
        assert cfg.camera_index == 0
        assert cfg.width == 640
        assert cfg.height == 480

    def test_custom_config(self) -> None:
        """Custom config values should be stored."""
        cfg = AcquisitionConfig(camera_index=1, width=1280, height=720, fps=60)
        assert cfg.camera_index == 1
        assert cfg.width == 1280

    def test_frame_metrics(self) -> None:
        """FrameMetrics should store values correctly."""
        m = FrameMetrics(timestamp=100.0, capture_latency_ms=5.0, fps=30.0, frame_index=42)
        assert m.fps == 30.0
        assert m.frame_index == 42


# =========================================================================
# Scenario tests
# =========================================================================

class TestScenarios:
    """Tests for scenario generators."""

    def test_normal_scenario_length(self) -> None:
        """Normal scenario should generate 30 steps."""
        s = NormalDrivingScenario()
        assert len(s.steps) == 30

    def test_gradual_drowsiness_perclos_rises(self) -> None:
        """PERCLOS should increase over time in drowsiness scenario."""
        s = GradualDrowsinessScenario()
        steps = s.steps
        assert steps[-1].indicators.perclos > steps[0].indicators.perclos

    def test_sudden_distraction_head_turn(self) -> None:
        """Distraction scenario should have head_distracted=True in middle."""
        s = SuddenDistractionScenario()
        steps = s.steps
        assert steps[7].indicators.head_distracted is True
        assert steps[0].indicators.head_distracted is False

    def test_face_lost_scenario(self) -> None:
        """Face lost scenario should have face_detected=False in middle."""
        s = FaceLostScenario()
        steps = s.steps
        assert steps[5].indicators.face_detected is False

    def test_yawning_scenario_mar(self) -> None:
        """Yawning scenario should have high MAR during yawn periods."""
        s = YawningScenario()
        steps = s.steps
        assert steps[7].indicators.mar > 0.5

    def test_emergency_scenario_phases(self) -> None:
        """Emergency scenario should have different phases."""
        s = EmergencyScenario()
        steps = s.steps
        # First phase: normal
        assert steps[0].indicators.face_detected is True
        # Critical phase: no face
        assert steps[20].indicators.face_detected is False

    def test_combined_fatigue_escalation(self) -> None:
        """Combined fatigue should show increasing indicators."""
        s = CombinedFatigueScenario()
        steps = s.steps
        assert steps[-1].indicators.perclos > steps[0].indicators.perclos
        assert steps[-1].indicators.ear < steps[0].indicators.ear


# =========================================================================
# Visualization tests
# =========================================================================

class TestVisualization:
    """Tests for Visualizer."""

    def test_render_empty_frame(self) -> None:
        """Render with blank frame should not crash."""
        viz = Visualizer(show_dashboard=True)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = viz.render(frame, state="ATTENTIVE", fps=30.0)
        assert result.shape[0] == 480
        assert result.shape[2] == 3

    def test_draw_face_bbox(self) -> None:
        """Drawing face bbox should modify frame."""
        viz = Visualizer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = viz.draw_face_bbox(frame, (100, 100, 200, 200))
        # Frame should have non-zero pixels from the rectangle
        assert np.any(result > 0)

    def test_warning_overlay_emergency(self) -> None:
        """Emergency state should trigger warning overlay."""
        viz = Visualizer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = viz.draw_warning_overlay(frame, "EMERGENCY")
        # May or may not flash depending on counter, but should not crash
        assert result.shape == frame.shape


# =========================================================================
# Integration test
# =========================================================================

class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline_synthetic(self) -> None:
        """Run full pipeline with synthetic data (no camera)."""
        ecu = ECUDecision(hysteresis_frames=1)
        viz = Visualizer(show_dashboard=False)

        # Simulate 10 frames
        for i in range(10):
            ear = 0.30 - i * 0.02
            perclos = i * 0.04
            ind = DriverIndicators(ear=ear, perclos=perclos)
            ecu.update(ind)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            rendered = viz.render(
                frame, state=ecu.state.name, ear=ear, perclos=perclos,
            )
            assert rendered is not None

    def test_ecu_with_scenario(self) -> None:
        """Run ECU through a complete scenario."""
        ecu = ECUDecision(hysteresis_frames=1, no_face_timeout=0.01, head_turn_timeout=0.01)
        scenario = EmergencyScenario()
        states_seen = set()
        for step in scenario.steps:
            time.sleep(0.02)
            ecu.update(step.indicators)
            states_seen.add(ecu.state)
        # Should have seen at least ATTENTIVE and some warning state
        assert DriverState.ATTENTIVE in states_seen
        assert len(states_seen) > 1
