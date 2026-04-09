"""
scenarios.py
============
Test scenarios for the Driver Monitoring System.

Each scenario generates synthetic indicator values over time for testing
the full DMS pipeline and ECU state machine.

Author: Daifi Meriem, Intern at Expleo Group Maroc for Stellantis
"""

import logging
from dataclasses import dataclass
from typing import List

from .ecu_decision import DriverIndicators

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

@dataclass
class ScenarioStep:
    """Single step in a scenario timeline.

    Args:
        time_offset: Seconds from scenario start.
        indicators: DriverIndicators for this step.
        description: Human-readable description.
    """
    time_offset: float
    indicators: DriverIndicators
    description: str = ""


class BaseScenario:
    """Base class for all scenarios.

    Args:
        name: Scenario name.
        description: Scenario description.
    """

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description
        self._steps: List[ScenarioStep] = []

    def generate(self) -> List[ScenarioStep]:
        """Generate the scenario timeline.

        Returns:
            List of ScenarioStep instances.
        """
        raise NotImplementedError

    @property
    def steps(self) -> List[ScenarioStep]:
        """Lazy-generate and cache steps."""
        if not self._steps:
            self._steps = self.generate()
        return self._steps


# ---------------------------------------------------------------------------
# Concrete scenarios
# ---------------------------------------------------------------------------

class NormalDrivingScenario(BaseScenario):
    """Normal attentive driving scenario."""

    def __init__(self) -> None:
        super().__init__("Normal Driving", "Driver is attentive throughout.")

    def generate(self) -> List[ScenarioStep]:
        """Generate normal driving indicators.

        Returns:
            List of ScenarioStep for 30 seconds of attentive driving.
        """
        steps = []
        for i in range(30):
            steps.append(ScenarioStep(
                time_offset=float(i),
                indicators=DriverIndicators(
                    ear=0.30, perclos=0.05, mar=0.2,
                    face_detected=True, head_distracted=False,
                ),
                description=f"t={i}s: Attentive driving",
            ))
        return steps


class GradualDrowsinessScenario(BaseScenario):
    """Gradual drowsiness with increasing PERCLOS."""

    def __init__(self) -> None:
        super().__init__("Gradual Drowsiness", "PERCLOS increases gradually over 60s.")

    def generate(self) -> List[ScenarioStep]:
        """Generate gradual drowsiness indicators.

        Returns:
            List of ScenarioStep with rising PERCLOS.
        """
        steps = []
        for i in range(60):
            perclos = min(i / 60.0 * 0.5, 0.5)
            ear = max(0.30 - (i / 60.0 * 0.15), 0.15)
            steps.append(ScenarioStep(
                time_offset=float(i),
                indicators=DriverIndicators(
                    ear=ear, perclos=perclos, mar=0.2,
                    face_detected=True, head_distracted=False,
                ),
                description=f"t={i}s: PERCLOS={perclos:.2f}, EAR={ear:.2f}",
            ))
        return steps


class SuddenDistractionScenario(BaseScenario):
    """Sudden head turn distraction."""

    def __init__(self) -> None:
        super().__init__("Sudden Distraction", "Driver suddenly turns head for >6 seconds.")

    def generate(self) -> List[ScenarioStep]:
        """Generate sudden distraction indicators.

        Returns:
            List of ScenarioStep with head turn event.
        """
        steps = []
        for i in range(20):
            head_distracted = 5 <= i <= 15
            steps.append(ScenarioStep(
                time_offset=float(i),
                indicators=DriverIndicators(
                    ear=0.30, perclos=0.05, mar=0.2,
                    face_detected=True, head_distracted=head_distracted,
                ),
                description=f"t={i}s: head_distracted={head_distracted}",
            ))
        return steps


class FaceLostScenario(BaseScenario):
    """Driver face lost (looking completely away)."""

    def __init__(self) -> None:
        super().__init__("Face Lost", "Face disappears for extended period.")

    def generate(self) -> List[ScenarioStep]:
        """Generate face-lost indicators.

        Returns:
            List of ScenarioStep with face_detected=False.
        """
        steps = []
        for i in range(15):
            face_detected = i < 3 or i > 12
            steps.append(ScenarioStep(
                time_offset=float(i),
                indicators=DriverIndicators(
                    ear=0.30 if face_detected else 0.0,
                    perclos=0.05, mar=0.2,
                    face_detected=face_detected,
                    head_distracted=False,
                ),
                description=f"t={i}s: face_detected={face_detected}",
            ))
        return steps


class YawningScenario(BaseScenario):
    """Yawning detection scenario."""

    def __init__(self) -> None:
        super().__init__("Yawning", "Driver yawns multiple times.")

    def generate(self) -> List[ScenarioStep]:
        """Generate yawning indicators.

        Returns:
            List of ScenarioStep with elevated MAR.
        """
        steps = []
        for i in range(30):
            yawning = i in range(5, 10) or i in range(18, 23)
            mar = 0.75 if yawning else 0.2
            steps.append(ScenarioStep(
                time_offset=float(i),
                indicators=DriverIndicators(
                    ear=0.28, perclos=0.10, mar=mar,
                    face_detected=True, head_distracted=False,
                    cnn_class=2 if yawning else 0,
                ),
                description=f"t={i}s: yawning={yawning}, MAR={mar:.2f}",
            ))
        return steps


class CombinedFatigueScenario(BaseScenario):
    """Combined fatigue indicators (low EAR + high PERCLOS + yawning)."""

    def __init__(self) -> None:
        super().__init__("Combined Fatigue", "Multiple fatigue indicators active.")

    def generate(self) -> List[ScenarioStep]:
        """Generate combined fatigue indicators.

        Returns:
            List of ScenarioStep with multiple fatigue cues.
        """
        steps = []
        for i in range(40):
            factor = min(i / 40.0, 1.0)
            ear = max(0.30 - factor * 0.18, 0.12)
            perclos = min(factor * 0.45, 0.45)
            mar = 0.3 + factor * 0.4
            steps.append(ScenarioStep(
                time_offset=float(i),
                indicators=DriverIndicators(
                    ear=ear, perclos=perclos, mar=mar,
                    face_detected=True, head_distracted=False,
                    cnn_class=1 if ear < 0.2 else 0,
                ),
                description=f"t={i}s: EAR={ear:.2f} PERCLOS={perclos:.2f} MAR={mar:.2f}",
            ))
        return steps


class EmergencyScenario(BaseScenario):
    """Worst-case emergency: full distraction leading to emergency stop."""

    def __init__(self) -> None:
        super().__init__(
            "Emergency",
            "Full distraction -> emergency braking + pull to side + hazard lights.",
        )

    def generate(self) -> List[ScenarioStep]:
        """Generate emergency scenario indicators.

        Returns:
            List of ScenarioStep escalating to emergency.
        """
        steps = []
        # Phase 1: Normal (0-5s)
        for i in range(5):
            steps.append(ScenarioStep(
                time_offset=float(i),
                indicators=DriverIndicators(
                    ear=0.30, perclos=0.05, mar=0.2,
                    face_detected=True, head_distracted=False,
                ),
                description=f"t={i}s: Normal driving",
            ))
        # Phase 2: Warning signs (5-15s)
        for i in range(5, 15):
            steps.append(ScenarioStep(
                time_offset=float(i),
                indicators=DriverIndicators(
                    ear=0.18, perclos=0.30, mar=0.5,
                    face_detected=True, head_distracted=True,
                    cnn_class=1,
                ),
                description=f"t={i}s: Warning signs",
            ))
        # Phase 3: Critical (15-25s)
        for i in range(15, 25):
            steps.append(ScenarioStep(
                time_offset=float(i),
                indicators=DriverIndicators(
                    ear=0.10, perclos=0.50, mar=0.8,
                    face_detected=False, head_distracted=True,
                    cnn_class=1,
                ),
                description=f"t={i}s: Critical - face lost + fatigue",
            ))
        # Phase 4: Recovery (25-30s)
        for i in range(25, 30):
            steps.append(ScenarioStep(
                time_offset=float(i),
                indicators=DriverIndicators(
                    ear=0.28, perclos=0.10, mar=0.2,
                    face_detected=True, head_distracted=False,
                ),
                description=f"t={i}s: Recovery",
            ))
        return steps
