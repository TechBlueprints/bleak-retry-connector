"""Connection watchdog for monitoring BLE notification activity.

Detects "zombie" connections where BlueZ still reports Connected=True
but no notifications are being received — the radio link is effectively
dead without a disconnect callback ever firing.

The caller must specify the expected timeout because only the caller
knows the device's notification cadence.  There is no sensible default
— a battery BMS sends every 1-5 s while a temperature sensor may send
every 60 s.  Making the timeout explicit forces the caller to think
about what "dead" means for their specific device.

Usage::

    # Battery BMS: expects data every ~5 s, dead after 30 s
    watchdog = ConnectionWatchdog(
        timeout=30.0,
        on_timeout=my_reconnect_callback,
    )
    watchdog.start()

    # In your notification callback:
    watchdog.notify_activity()

    # When done:
    watchdog.stop()

When *client* and *device* are provided, the watchdog automatically
tears down the connection at the BlueZ level before invoking the
callback.  This ensures the next ``establish_connection()`` call
starts fresh instead of adopting stale state::

    from bleak_retry_connector import ConnectionWatchdog

    watchdog = ConnectionWatchdog(
        timeout=30.0,
        on_timeout=my_reconnect_callback,
        client=client,
        device=device,
    )

Important: avoid ``async with BleakClient`` for long-lived connections.
Its ``__aexit__`` calls ``disconnect()`` without a timeout, which hangs
indefinitely in phantom states.  Always use explicit ``connect()`` /
``disconnect()`` with ``asyncio.wait_for(client.disconnect(), timeout=5.0)``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from .bluez import clear_cache

if TYPE_CHECKING:
    from bleak import BleakClient
    from bleak.backends.device import BLEDevice

_LOGGER = logging.getLogger(__name__)

DISCONNECT_TIMEOUT = 5.0


class ConnectionWatchdog:
    """Monitor a BLE connection for notification activity.

    Tracks the time since the last :meth:`notify_activity` call.
    When the timeout is exceeded the optional *on_timeout* callback
    is invoked so the caller can trigger reconnection or cleanup.

    When *client* and *device* are both provided, the watchdog
    performs BlueZ-level cleanup before invoking the callback:

    1. ``client.disconnect()`` with a 5 s timeout (prevents hang
       on phantom connections — Stuck State 8).
    2. ``clear_cache(device.address)`` to remove the device from
       BlueZ so the next ``establish_connection()`` starts fresh.
    3. The *on_timeout* callback, where the caller can reconnect.

    The monitoring loop runs as an ``asyncio.Task`` — no threads are
    needed for the normal case.

    Parameters
    ----------
    timeout:
        Seconds of inactivity before the watchdog fires.  Required —
        there is no default because only the caller knows the device's
        expected notification cadence.
    on_timeout:
        Async callback invoked when the timeout expires.  If ``None``,
        the watchdog only logs a warning.
    client:
        The connected ``BleakClient``.  When provided together with
        *device*, the watchdog disconnects and clears the BlueZ cache
        on timeout before invoking *on_timeout*.
    device:
        The ``BLEDevice`` for the connection.  Required together with
        *client* for BlueZ-level cleanup.
    """

    def __init__(
        self,
        timeout: float,
        on_timeout: Callable[[], Awaitable[None]] | None = None,
        client: BleakClient | None = None,
        device: BLEDevice | None = None,
    ) -> None:
        self._timeout = timeout
        self._on_timeout = on_timeout
        self._client = client
        self._device = device
        self._last_activity: float = 0.0
        self._task: asyncio.Task[None] | None = None
        self._started = False

    @property
    def is_running(self) -> bool:
        """Return whether the watchdog is actively monitoring."""
        return self._started and self._task is not None and not self._task.done()

    @property
    def last_activity(self) -> float:
        """Return the monotonic timestamp of the last activity."""
        return self._last_activity

    def notify_activity(self) -> None:
        """Record that a notification or other activity was received.

        Call this from your BLE notification callback to reset the
        watchdog timer.
        """
        self._last_activity = time.monotonic()

    def start(self) -> None:
        """Start the watchdog monitoring loop.

        Records the current time as the initial activity timestamp and
        creates an asyncio task for the monitoring loop.  Calling
        ``start()`` on an already-running watchdog is a no-op.
        """
        if self._started:
            return
        self._last_activity = time.monotonic()
        self._started = True
        self._task = asyncio.ensure_future(self._monitor())

    def stop(self) -> None:
        """Stop the watchdog.

        Cancels the monitoring task.  Safe to call multiple times or
        before ``start()``.
        """
        self._started = False
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def _cleanup_connection(self) -> None:
        """Disconnect the client and clear BlueZ cache.

        Called when *client* and *device* were provided and the
        inactivity timeout has fired.  Each step is wrapped in
        exception handling so a failure in one does not prevent the
        next step or the caller's *on_timeout* callback.
        """
        if self._client is None or self._device is None:
            return
        address = self._device.address

        # Step 1: disconnect with timeout (prevents State 8 hang)
        try:
            await asyncio.wait_for(
                self._client.disconnect(), timeout=DISCONNECT_TIMEOUT
            )
        except asyncio.TimeoutError:
            _LOGGER.debug(
                "ConnectionWatchdog: disconnect timed out for %s,"
                " proceeding to cache clear",
                address,
            )
        except Exception:
            _LOGGER.debug(
                "ConnectionWatchdog: disconnect failed for %s,"
                " proceeding to cache clear",
                address,
                exc_info=True,
            )

        # Step 2: remove device from BlueZ so next connect starts fresh
        try:
            await clear_cache(address)
        except Exception:
            _LOGGER.debug(
                "ConnectionWatchdog: clear_cache failed for %s",
                address,
                exc_info=True,
            )

    async def _monitor(self) -> None:
        """Internal monitoring loop.

        Wakes up periodically and checks whether the inactivity timeout
        has been exceeded.  Uses a check interval of half the timeout
        (clamped to 1–30 s) so the actual fire time is at most one
        interval late.
        """
        check_interval = min(self._timeout / 2, 30.0)
        try:
            while self._started:
                await asyncio.sleep(check_interval)
                elapsed = time.monotonic() - self._last_activity
                if elapsed < self._timeout:
                    continue

                _LOGGER.warning(
                    "ConnectionWatchdog: no activity for %.1f s" " (timeout %.1f s)",
                    elapsed,
                    self._timeout,
                )

                if self._client is not None and self._device is not None:
                    await self._cleanup_connection()

                if self._on_timeout is not None:
                    try:
                        await self._on_timeout()
                    except Exception:
                        _LOGGER.exception(
                            "ConnectionWatchdog: on_timeout callback failed"
                        )
                break
        except asyncio.CancelledError:
            pass
        finally:
            self._started = False
