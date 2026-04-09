"""
ecu_decision.py
===============
ECU-like Finite State Machine for the Driver Monitoring System.

Implements priority-based state transitions with hysteresis and warning
escalation: visual alert -> vibration -> emergency braking + hazard lights.

States (priority order):
  ATTENTIVE -> WARNING_DROWSINESS / WARNING_DISTRACTION -> FATIGUE ->
  DISTRACTED_HEAD -> DISTRACTED_NO_FACE -> EMERGENCY

Author: Daifi Meriem, Intern at Expleo Group Maroc for Stellantis
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants / thresholds
# ---------------------------------------------------------------------------
DEFAULT_NO_FACE_TIMEOUT: float = 3.0
DEFAULT_HEAD_TURN_TIMEOUT: float = 6.0
DEFAULT_PERCLOS_FATIGUE: float = 0.35
DEFAULT_PERCLOS_WARNING: float = 0.20
DEFAULT_EAR_WARNING: float = 0.25
DEFAULT_MAR_YAWN: float = 0.6
DEFAULT_HYSTERESIS_FRAMES: int = 5
DEFAULT_ESCALATION_TIMEOUT: float = 5.0


class DriverState(Enum):
    """All possible FSM driver states."""
    ATTENTIVE = auto()
    WARNING_DROWSINESS = auto()
    WARNING_DISTRACTION = auto()
    FATIGUE = auto()
    DISTRACTED_HEAD = auto()
    DISTRACTED_NO_FACE = auto()
    EMERGENCY = auto()


class WarningLevel(Enum):
    """Warning escalation levels."""
    NONE = 0
    LEVEL_1 = 1  # Visual alert + audio beep
    LEVEL_2 = 2  # Seat / steering wheel vibration
    LEVEL_3 = 3  # Emergency braking + pull to side + hazard lights


@dataclass
class ActionCommand:
    """Commands issued to vehicle actuators.

    Args:
        audio_alert: Play audio beep/warning.
        visual_alert: Show on-screen warning.
        seat_vibration: Activate seat vibration.
        steering_vibration: Activate steering wheel vibration.
        emergency_braking: Trigger emergency braking.
        pull_to_side: Command autonomous pull-to-side manoeuvre.
        hazard_lights: Activate hazard (4-way) lights.
        warning_level: Current WarningLevel enum value.
    """
    audio_alert: bool = False
    visual_alert: bool = False
    seat_vibration: bool = False
    steering_vibration: bool = False
    emergency_braking: bool = False
    pull_to_side: bool = False
    hazard_lights: bool = False
    warning_level: WarningLevel = WarningLevel.NONE


@dataclass
class DriverIndicators:
    """Snapshot of behavioural indicators for a single frame.

    Args:
        ear: Eye Aspect Ratio.
        perclos: PERCLOS value [0, 1].
        mar: Mouth Aspect Ratio.
        face_detected: Whether a face was found.
        head_distracted: Whether head pose signals distraction.
        cnn_class: CNN predicted class (0=OPEN, 1=CLOSED, 2=YAWNING).
        blink_rate: Blinks per minute.
    """
    ear: float = 0.3
    perclos: float = 0.0
    mar: float = 0.0
    face_detected: bool = True
    head_distracted: bool = False
    cnn_class: Optional[int] = None
    blink_rate: float = 15.0


class ECUDecision:
    """Priority-based ECU Finite State Machine.

    Args:
        no_face_timeout: Seconds without face before DISTRACTED_NO_FACE.
        head_turn_timeout: Seconds with head turned before DISTRACTED_HEAD.
        perclos_fatigue: PERCLOS threshold for FATIGUE state.
        perclos_warning: PERCLOS threshold for WARNING_DROWSINESS.
        ear_warning: EAR below this triggers WARNING_DROWSINESS.
        mar_yawn: MAR above this contributes to drowsiness warning.
        hysteresis_frames: Consecutive confirming frames for state change.
        escalation_timeout: Seconds in WARNING before escalating.
    """

    def __init__(
        self,
        no_face_timeout: float = DEFAULT_NO_FACE_TIMEOUT,
        head_turn_timeout: float = DEFAULT_HEAD_TURN_TIMEOUT,
        perclos_fatigue: float = DEFAULT_PERCLOS_FATIGUE,
        perclos_warning: float = DEFAULT_PERCLOS_WARNING,
        ear_warning: float = DEFAULT_EAR_WARNING,
        mar_yawn: float = DEFAULT_MAR_YAWN,
        hysteresis_frames: int = DEFAULT_HYSTERESIS_FRAMES,
        escalation_timeout: float = DEFAULT_ESCALATION_TIMEOUT,
    ) -> None:
        self.no_face_timeout = no_face_timeout
        self.head_turn_timeout = head_turn_timeout
        self.perclos_fatigue = perclos_fatigue
        self.perclos_warning = perclos_warning
        self.ear_warning = ear_warning
        self.mar_yawn = mar_yawn
        self.hysteresis_frames = hysteresis_frames
        self.escalation_timeout = escalation_timeout

        self._state: DriverState = DriverState.ATTENTIVE
        self._candidate_state: DriverState = DriverState.ATTENTIVE
        self._candidate_count: int = 0

        self._no_face_since: Optional[float] = None
        self._head_turned_since: Optional[float] = None
        self._warning_since: Optional[float] = None
        self._state_entered_at: float = time.time()
        self._warning_level: WarningLevel = WarningLevel.NONE
        self._state_history: List[Dict] = []

    def update(self, indicators: DriverIndicators) -> "ECUDecision":
        """Process one frame of indicators and update FSM state.

        Args:
            indicators: DriverIndicators snapshot for this frame.

        Returns:
            self (fluent interface).
        """
        now = time.time()
        target = self._compute_target_state(indicators, now)
        self._apply_hysteresis(target, now)
        self._update_warning_escalation(now)
        return self

    @property
    def state(self) -> DriverState:
        """Current confirmed FSM state."""
        return self._state

    @property
    def warning_level(self) -> WarningLevel:
        """Current warning escalation level."""
        return self._warning_level

    @property
    def action(self) -> ActionCommand:
        """Action command for the current state and warning level."""
        return self._build_action()

    def reset(self) -> None:
        """Reset the FSM to ATTENTIVE state."""
        self._state = DriverState.ATTENTIVE
        self._candidate_state = DriverState.ATTENTIVE
        self._candidate_count = 0
        self._no_face_since = None
        self._head_turned_since = None
        self._warning_since = None
        self._warning_level = WarningLevel.NONE
        self._state_entered_at = time.time()
        logger.info("ECU FSM reset to ATTENTIVE.")

    def _compute_target_state(
        self, ind: DriverIndicators, now: float
    ) -> DriverState:
        """Determine desired next state based on priority rules.

        Priority (highest first):
          1. No face > timeout -> DISTRACTED_NO_FACE
          2. Head turned > timeout -> DISTRACTED_HEAD
          3. PERCLOS > fatigue threshold -> FATIGUE
          4. Warning indicators -> WARNING_DROWSINESS
          5. Head distracted (within timeout) -> WARNING_DISTRACTION
          6. Default -> ATTENTIVE

        Args:
            ind: Current frame indicators.
            now: Current epoch time.

        Returns:
            Target DriverState.
        """
        # Track face absence timer
        if not ind.face_detected:
            if self._no_face_since is None:
                self._no_face_since = now
        else:
            self._no_face_since = None

        # Track head turn timer
        if ind.head_distracted:
            if self._head_turned_since is None:
                self._head_turned_since = now
        else:
            self._head_turned_since = None

        # Priority 1: No face
        if (
            self._no_face_since is not None
            and (now - self._no_face_since) >= self.no_face_timeout
        ):
            return DriverState.DISTRACTED_NO_FACE

        # Priority 2: Head turned too long
        if (
            self._head_turned_since is not None
            and (now - self._head_turned_since) >= self.head_turn_timeout
        ):
            return DriverState.DISTRACTED_HEAD

        # Priority 3: PERCLOS fatigue
        if ind.perclos >= self.perclos_fatigue:
            return DriverState.FATIGUE

        # Priority 4: Drowsiness warning
        ear_low = ind.ear < self.ear_warning
        perclos_warn = ind.perclos >= self.perclos_warning
        yawning = ind.mar >= self.mar_yawn
        cnn_closed = ind.cnn_class == 1
        cnn_yawn = ind.cnn_class == 2

        if ear_low or perclos_warn or yawning or cnn_closed or cnn_yawn:
            return DriverState.WARNING_DROWSINESS

        # Priority 5: Head distraction warning
        if ind.head_distracted:
            return DriverState.WARNING_DISTRACTION

        return DriverState.ATTENTIVE

    def _apply_hysteresis(self, target: DriverState, now: float) -> None:
        """Confirm state change only after hysteresis_frames consecutive hits.

        Args:
            target: Desired next state.
            now: Current epoch time.
        """
        if target == self._candidate_state:
            self._candidate_count += 1
        else:
            self._candidate_state = target
            self._candidate_count = 1

        if self._candidate_count >= self.hysteresis_frames:
            if target != self._state:
                self._transition_to(target, now)

    def _transition_to(self, new_state: DriverState, now: float) -> None:
        """Execute state transition.

        Args:
            new_state: State to transition into.
            now: Current epoch time.
        """
        old_state = self._state
        self._state = new_state
        self._state_entered_at = now

        if new_state == DriverState.ATTENTIVE:
            self._warning_since = None
            self._warning_level = WarningLevel.NONE
        elif new_state in (DriverState.WARNING_DROWSINESS, DriverState.WARNING_DISTRACTION):
            if self._warning_since is None:
                self._warning_since = now
            self._warning_level = WarningLevel.LEVEL_1
        elif new_state in (DriverState.FATIGUE, DriverState.DISTRACTED_HEAD, DriverState.DISTRACTED_NO_FACE):
            self._warning_level = WarningLevel.LEVEL_2
        elif new_state == DriverState.EMERGENCY:
            self._warning_level = WarningLevel.LEVEL_3

        self._state_history.append({
            "from": old_state.name, "to": new_state.name, "time": now,
        })
        logger.info("ECU state: %s -> %s (level=%s)", old_state.name, new_state.name, self._warning_level.name)

    def _update_warning_escalation(self, now: float) -> None:
        """Escalate warning level if stuck in WARNING state too long.

        Args:
            now: Current epoch time.
        """
        if self._state in (DriverState.WARNING_DROWSINESS, DriverState.WARNING_DISTRACTION):
            if self._warning_since and (now - self._warning_since) >= self.escalation_timeout:
                if self._warning_level == WarningLevel.LEVEL_1:
                    self._warning_level = WarningLevel.LEVEL_2
                    logger.info("ECU escalated to LEVEL_2")
                elif self._warning_level == WarningLevel.LEVEL_2:
                    self._warning_level = WarningLevel.LEVEL_3
                    self._state = DriverState.EMERGENCY
                    self._state_entered_at = now
                    logger.info("ECU escalated to EMERGENCY")

        if self._state in (DriverState.FATIGUE, DriverState.DISTRACTED_HEAD, DriverState.DISTRACTED_NO_FACE):
            time_in = now - self._state_entered_at
            if time_in >= self.escalation_timeout and self._warning_level.value < WarningLevel.LEVEL_3.value:
                self._warning_level = WarningLevel.LEVEL_3
                self._state = DriverState.EMERGENCY
                self._state_entered_at = now
                logger.info("ECU escalated critical state to EMERGENCY")

    def _build_action(self) -> ActionCommand:
        """Build action command from current state and warning level.

        Returns:
            ActionCommand for actuators.
        """
        wl = self._warning_level
        if wl == WarningLevel.NONE:
            return ActionCommand(warning_level=WarningLevel.NONE)

        if wl == WarningLevel.LEVEL_1:
            return ActionCommand(
                audio_alert=True, visual_alert=True,
                warning_level=WarningLevel.LEVEL_1,
            )
        if wl == WarningLevel.LEVEL_2:
            return ActionCommand(
                audio_alert=True, visual_alert=True,
                seat_vibration=True, steering_vibration=True,
                warning_level=WarningLevel.LEVEL_2,
            )
        # LEVEL_3: Emergency
        return ActionCommand(
            audio_alert=True, visual_alert=True,
            seat_vibration=True, steering_vibration=True,
            emergency_braking=True, pull_to_side=True,
            hazard_lights=True, warning_level=WarningLevel.LEVEL_3,
        )
