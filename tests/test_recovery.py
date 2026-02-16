"""Tests for the recovery / escalation chain module."""

from __future__ import annotations

import time

import pytest

from bleak_retry_connector.recovery import (
    PROFILE_BATTERY,
    PROFILE_ON_DEMAND,
    PROFILE_SENSOR,
    EscalationAction,
    EscalationConfig,
    EscalationPolicy,
)


# ---------------------------------------------------------------------------
# EscalationConfig tests
# ---------------------------------------------------------------------------


class TestEscalationConfig:
    """Tests for EscalationConfig defaults and profiles."""

    def test_defaults(self):
        """Default config has safe defaults."""
        cfg = EscalationConfig()
        assert cfg.diagnose_and_fix is True
        assert cfg.clear_bluez_on_inprogress_dominance is True
        assert cfg.rotate_adapter is True
        assert cfg.reset_adapter is False
        assert cfg.rotate_after == 2
        assert cfg.clear_after == 4
        assert cfg.reset_after == 6
        assert cfg.reset_cooldown == 300.0
        assert cfg.max_escalation == EscalationAction.RESET_ADAPTER

    def test_profile_battery(self):
        """Battery profile enables reset."""
        assert PROFILE_BATTERY.reset_adapter is True
        assert PROFILE_BATTERY.reset_after == 6
        assert PROFILE_BATTERY.reset_cooldown == 300.0

    def test_profile_sensor(self):
        """Sensor profile caps at rotation, no reset."""
        assert PROFILE_SENSOR.reset_adapter is False
        assert PROFILE_SENSOR.max_escalation == EscalationAction.ROTATE_ADAPTER

    def test_profile_on_demand(self):
        """On-demand profile rotates fast, skips InProgress dominance."""
        assert PROFILE_ON_DEMAND.clear_bluez_on_inprogress_dominance is False
        assert PROFILE_ON_DEMAND.reset_adapter is False
        assert PROFILE_ON_DEMAND.rotate_after == 1
        assert PROFILE_ON_DEMAND.max_escalation == EscalationAction.ROTATE_ADAPTER


# ---------------------------------------------------------------------------
# EscalationAction tests
# ---------------------------------------------------------------------------


class TestEscalationAction:
    """Tests for the EscalationAction enum."""

    def test_values(self):
        vals = {
            EscalationAction.RETRY: "retry",
            EscalationAction.DIAGNOSE: "diagnose",
            EscalationAction.CLEAR_BLUEZ: "clear_bluez",
            EscalationAction.ROTATE_ADAPTER: "rotate",
            EscalationAction.RESET_ADAPTER: "reset",
        }
        for action, expected in vals.items():
            assert action.value == expected

    def test_is_str_subclass(self):
        """EscalationAction should be usable as a string."""
        assert isinstance(EscalationAction.RETRY, str)


# ---------------------------------------------------------------------------
# EscalationPolicy tests
# ---------------------------------------------------------------------------


class TestEscalationPolicy:
    """Tests for the EscalationPolicy decision-making logic."""

    def test_first_failure_returns_diagnose(self):
        """First failure with default config should suggest DIAGNOSE."""
        policy = EscalationPolicy(["hci0"])
        action = policy.on_failure("hci0")
        assert action == EscalationAction.DIAGNOSE

    def test_rotate_after_threshold(self):
        """After rotate_after failures, should suggest ROTATE_ADAPTER."""
        policy = EscalationPolicy(["hci0", "hci1"])
        policy.on_failure("hci0")  # 1 → DIAGNOSE
        action = policy.on_failure("hci0")  # 2 → ROTATE
        assert action == EscalationAction.ROTATE_ADAPTER

    def test_clear_after_threshold(self):
        """After clear_after failures, should suggest CLEAR_BLUEZ."""
        policy = EscalationPolicy(["hci0"])
        for _ in range(3):
            policy.on_failure("hci0")
        action = policy.on_failure("hci0")  # 4 → CLEAR
        assert action == EscalationAction.CLEAR_BLUEZ

    def test_reset_after_threshold(self):
        """After reset_after failures, should suggest RESET_ADAPTER if enabled."""
        config = EscalationConfig(reset_adapter=True)
        policy = EscalationPolicy(["hci0"], config=config)
        for _ in range(5):
            policy.on_failure("hci0")
        action = policy.on_failure("hci0")  # 6 → RESET
        assert action == EscalationAction.RESET_ADAPTER

    def test_reset_disabled_by_default(self):
        """Default config does not enable reset — should fall back to CLEAR_BLUEZ."""
        policy = EscalationPolicy(["hci0"])
        for _ in range(10):
            action = policy.on_failure("hci0")
        # Even after many failures, should not suggest RESET
        assert action != EscalationAction.RESET_ADAPTER

    def test_on_success_resets_counter(self):
        """on_success should reset the failure counter."""
        policy = EscalationPolicy(["hci0"])
        policy.on_failure("hci0")
        policy.on_failure("hci0")
        policy.on_success("hci0")
        action = policy.on_failure("hci0")  # back to 1 → DIAGNOSE
        assert action == EscalationAction.DIAGNOSE

    def test_max_escalation_caps_actions(self):
        """max_escalation should cap the returned action."""
        config = EscalationConfig(
            reset_adapter=True,
            max_escalation=EscalationAction.ROTATE_ADAPTER,
        )
        policy = EscalationPolicy(["hci0"], config=config)
        for _ in range(20):
            action = policy.on_failure("hci0")
        # Even after many failures, should not exceed ROTATE
        assert action in (
            EscalationAction.ROTATE_ADAPTER,
            EscalationAction.CLEAR_BLUEZ,
        )
        assert action != EscalationAction.RESET_ADAPTER

    def test_diagnose_disabled(self):
        """If diagnose_and_fix is disabled, skip to next enabled level."""
        config = EscalationConfig(diagnose_and_fix=False)
        policy = EscalationPolicy(["hci0"], config=config)
        action = policy.on_failure("hci0")
        # diagnose disabled, failure count=1 (< rotate_after=2), so RETRY
        assert action == EscalationAction.RETRY

    def test_per_adapter_tracking(self):
        """Failures are tracked per adapter."""
        policy = EscalationPolicy(["hci0", "hci1"])
        policy.on_failure("hci0")
        policy.on_failure("hci0")  # hci0 at 2
        action_hci1 = policy.on_failure("hci1")  # hci1 at 1
        assert action_hci1 == EscalationAction.DIAGNOSE

    def test_record_reset_clears_counter(self):
        """record_reset should clear the failure counter and record time."""
        config = EscalationConfig(reset_adapter=True)
        policy = EscalationPolicy(["hci0"], config=config)
        for _ in range(6):
            policy.on_failure("hci0")
        policy.record_reset("hci0")
        action = policy.on_failure("hci0")  # back to 1
        assert action == EscalationAction.DIAGNOSE

    def test_reset_cooldown(self):
        """Reset should be blocked during cooldown period."""
        config = EscalationConfig(reset_adapter=True, reset_cooldown=300.0)
        policy = EscalationPolicy(["hci0"], config=config)

        # Simulate a recent reset
        policy._last_reset["hci0"] = time.monotonic()

        for _ in range(10):
            action = policy.on_failure("hci0")
        # Should not suggest RESET because cooldown hasn't elapsed
        assert action != EscalationAction.RESET_ADAPTER

    def test_reset_cooldown_expired(self):
        """Reset should be allowed after cooldown expires."""
        config = EscalationConfig(reset_adapter=True, reset_cooldown=0.0)
        policy = EscalationPolicy(["hci0"], config=config)
        for _ in range(5):
            policy.on_failure("hci0")
        action = policy.on_failure("hci0")  # 6 → RESET (cooldown=0)
        assert action == EscalationAction.RESET_ADAPTER

    def test_config_property(self):
        """config property should return the current config."""
        config = EscalationConfig(reset_adapter=True)
        policy = EscalationPolicy(["hci0"], config=config)
        assert policy.config is config

    def test_default_config_if_none(self):
        """If no config provided, should use default EscalationConfig."""
        policy = EscalationPolicy(["hci0"])
        assert isinstance(policy.config, EscalationConfig)
        assert policy.config.reset_adapter is False

    def test_unknown_adapter_on_failure(self):
        """on_failure with an unknown adapter should still work."""
        policy = EscalationPolicy(["hci0"])
        # hci1 not in initial list but should be handled gracefully
        action = policy.on_failure("hci1")
        assert action == EscalationAction.DIAGNOSE

    def test_sensor_profile_never_resets(self):
        """Sensor profile should never suggest RESET_ADAPTER."""
        policy = EscalationPolicy(["hci0", "hci1"], config=PROFILE_SENSOR)
        for _ in range(20):
            action = policy.on_failure("hci0")
        assert action != EscalationAction.RESET_ADAPTER

    def test_on_demand_profile_rotates_fast(self):
        """On-demand profile should rotate after 1 failure."""
        policy = EscalationPolicy(["hci0", "hci1"], config=PROFILE_ON_DEMAND)
        action = policy.on_failure("hci0")  # 1 → ROTATE (rotate_after=1)
        assert action == EscalationAction.ROTATE_ADAPTER


# ---------------------------------------------------------------------------
# Integration: importability from top-level
# ---------------------------------------------------------------------------


def test_importable_from_top_level():
    """All recovery symbols should be importable from bleak_retry_connector."""
    from bleak_retry_connector import (  # noqa: F401
        PROFILE_BATTERY,
        PROFILE_ON_DEMAND,
        PROFILE_SENSOR,
        EscalationAction,
        EscalationConfig,
        EscalationPolicy,
    )
