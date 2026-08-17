from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flightrescue import FlightRescueService


def test_service_reports_missing_artifacts_without_crashing(tmp_path):
    service = FlightRescueService(tmp_path)
    status = service.status()
    assert status["ready"] is False
    assert status["any_model"] is False
    assert status["severe_model"] is False


def test_risk_labels_are_monotonic():
    assert FlightRescueService._risk_label(0.10) == "low"
    assert FlightRescueService._risk_label(0.40) == "moderate"
    assert FlightRescueService._risk_label(0.60) == "high"
    assert FlightRescueService._risk_label(0.80) == "very_high"
