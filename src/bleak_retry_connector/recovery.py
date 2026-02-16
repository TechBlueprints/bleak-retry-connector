"""Adapter recovery and escalation chain for BLE connection failures.

Provides a configurable escalation policy that tracks consecutive
failures per adapter and recommends increasingly aggressive recovery
actions.  The policy respects caller configuration — it never suggests
an action the caller has disabled.

Escalation levels (least to most disruptive)::

    1. RETRY          — simple backoff retry
    2. DIAGNOSE       — diagnose stuck state + targeted fix
    3. CLEAR_BLUEZ    — clear InProgress-dominant stale BlueZ state
    4. ROTATE_ADAPTER — switch to a different adapter
    5. RESET_ADAPTER  — power-cycle adapter (disrupts ALL connections)

For adapter reset, callers should use ``bluetooth-auto-recovery``
(``bluetooth_auto_recovery.recover_adapter()``) which handles adapter
recovery via the BlueZ management socket, kernel ioctl, USB device
reset, and rfkill — rather than shelling out to ``hciconfig``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Escalation configuration
# ---------------------------------------------------------------------------


class EscalationAction(str, Enum):
    """Actions the escalation policy can recommend."""

    RETRY = "retry"
    DIAGNOSE = "diagnose"
    CLEAR_BLUEZ = "clear_bluez"
    ROTATE_ADAPTER = "rotate"
    RESET_ADAPTER = "reset"


# Ordered from least to most disruptive
_LEVELS = list(EscalationAction)


@dataclass
class EscalationConfig:
    """Configuration for the recovery escalation chain.

    Each escalation level can be individually enabled or disabled.
    Thresholds control when each level triggers.

    Parameters
    ----------
    diagnose_and_fix:
        Enable stuck-state diagnosis + targeted fix.  Almost always
        ``True``.
    clear_bluez_on_inprogress_dominance:
        Enable BlueZ state cleanup when ``InProgress`` errors dominate
        all adapters.
    rotate_adapter:
        Enable adapter rotation on failure.  Requires multiple adapters.
    reset_adapter:
        Enable adapter reset as last resort.  **WARNING:** disrupts ALL
        connections on the adapter.  Only enable if this service "owns"
        the adapter or coordinates with others.  Default ``False``.
    rotate_after:
        Consecutive failures before rotating adapter.
    clear_after:
        Consecutive ``InProgress`` failures before BlueZ cleanup.
    reset_after:
        Consecutive failures before adapter reset.
    reset_cooldown:
        Minimum seconds between adapter resets.
    max_escalation:
        Hard ceiling on escalation.  Even if ``reset_adapter`` is
        ``True``, setting ``max_escalation`` to
        :attr:`EscalationAction.ROTATE_ADAPTER` prevents reset.
    """

    diagnose_and_fix: bool = True
    clear_bluez_on_inprogress_dominance: bool = True
    rotate_adapter: bool = True
    reset_adapter: bool = False
    rotate_after: int = 2
    clear_after: int = 4
    reset_after: int = 6
    reset_cooldown: float = 300.0
    max_escalation: EscalationAction = EscalationAction.RESET_ADAPTER


# Pre-built profiles for common service types
PROFILE_BATTERY = EscalationConfig(
    reset_adapter=True,
    reset_after=6,
    reset_cooldown=300.0,
)

PROFILE_SENSOR = EscalationConfig(
    reset_adapter=False,
    max_escalation=EscalationAction.ROTATE_ADAPTER,
)

PROFILE_ON_DEMAND = EscalationConfig(
    clear_bluez_on_inprogress_dominance=False,
    reset_adapter=False,
    rotate_after=1,
    max_escalation=EscalationAction.ROTATE_ADAPTER,
)


# ---------------------------------------------------------------------------
# Escalation policy
# ---------------------------------------------------------------------------


class EscalationPolicy:
    """Track consecutive failures per adapter and decide escalation level.

    The policy respects the caller's :class:`EscalationConfig` — it will
    never suggest an action the caller has disabled.

    Example::

        config = EscalationConfig(reset_adapter=False)  # sensor service
        policy = EscalationPolicy(["hci0", "hci1"], config=config)

        action = policy.on_failure("hci0")
        # action will never be RESET_ADAPTER because config disabled it

        policy.on_success("hci0")  # resets failure counter
    """

    def __init__(
        self,
        adapters: list[str],
        config: EscalationConfig | None = None,
    ) -> None:
        self._config = config or EscalationConfig()
        self._adapters = adapters
        self._max_level_idx = _LEVELS.index(self._config.max_escalation)
        self._failures: dict[str, int] = {a: 0 for a in adapters}
        self._last_reset: dict[str, float] = {a: 0.0 for a in adapters}

    @property
    def config(self) -> EscalationConfig:
        """Return the current escalation configuration."""
        return self._config

    def on_failure(self, adapter: str) -> EscalationAction:
        """Record a failure and return the next escalation action.

        The returned action will never exceed *max_escalation* or
        suggest a disabled level.
        """
        self._failures[adapter] = self._failures.get(adapter, 0) + 1
        count = self._failures[adapter]

        if (
            count >= self._config.reset_after
            and self._is_level_enabled(EscalationAction.RESET_ADAPTER)
            and self._can_reset(adapter)
        ):
            return EscalationAction.RESET_ADAPTER

        if count >= self._config.clear_after and self._is_level_enabled(
            EscalationAction.CLEAR_BLUEZ
        ):
            return EscalationAction.CLEAR_BLUEZ

        if count >= self._config.rotate_after and self._is_level_enabled(
            EscalationAction.ROTATE_ADAPTER
        ):
            return EscalationAction.ROTATE_ADAPTER

        if count >= 1 and self._is_level_enabled(EscalationAction.DIAGNOSE):
            return EscalationAction.DIAGNOSE

        return EscalationAction.RETRY

    def on_success(self, adapter: str) -> None:
        """Record a success — resets the failure counter for *adapter*."""
        self._failures[adapter] = 0

    def record_reset(self, adapter: str) -> None:
        """Record that an adapter reset was performed."""
        self._last_reset[adapter] = time.monotonic()
        self._failures[adapter] = 0

    def _is_level_enabled(self, level: EscalationAction) -> bool:
        """Check if a given escalation level is enabled in config."""
        if _LEVELS.index(level) > self._max_level_idx:
            return False
        level_config_map = {
            EscalationAction.DIAGNOSE: self._config.diagnose_and_fix,
            EscalationAction.CLEAR_BLUEZ: (
                self._config.clear_bluez_on_inprogress_dominance
            ),
            EscalationAction.ROTATE_ADAPTER: self._config.rotate_adapter,
            EscalationAction.RESET_ADAPTER: self._config.reset_adapter,
        }
        return level_config_map.get(level, True)

    def _can_reset(self, adapter: str) -> bool:
        """Check if enough time has passed since the last reset."""
        last = self._last_reset.get(adapter, 0.0)
        return (time.monotonic() - last) >= self._config.reset_cooldown
