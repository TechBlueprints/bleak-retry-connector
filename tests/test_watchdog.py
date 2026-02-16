"""Tests for ConnectionWatchdog."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bleak_retry_connector import ConnectionWatchdog


@pytest.mark.asyncio
async def test_watchdog_fires_on_inactivity():
    """Watchdog fires the callback after the timeout expires."""
    callback = AsyncMock()
    wd = ConnectionWatchdog(timeout=0.2, on_timeout=callback)
    wd.start()

    # Don't call notify_activity — let it expire
    await asyncio.sleep(0.5)

    callback.assert_awaited_once()
    assert not wd.is_running


@pytest.mark.asyncio
async def test_watchdog_activity_resets_timer():
    """notify_activity() resets the watchdog timer."""
    callback = AsyncMock()
    wd = ConnectionWatchdog(timeout=0.3, on_timeout=callback)
    wd.start()

    # Keep feeding activity for a while
    for _ in range(5):
        await asyncio.sleep(0.1)
        wd.notify_activity()

    # Should NOT have fired yet
    callback.assert_not_awaited()

    # Now let it expire
    await asyncio.sleep(0.6)
    callback.assert_awaited_once()


@pytest.mark.asyncio
async def test_watchdog_stop_cancels():
    """stop() prevents the callback from firing."""
    callback = AsyncMock()
    wd = ConnectionWatchdog(timeout=0.2, on_timeout=callback)
    wd.start()
    running_before_stop = wd.is_running
    assert running_before_stop

    wd.stop()
    running_after_stop = wd.is_running
    assert not running_after_stop

    await asyncio.sleep(0.5)
    callback.assert_not_awaited()


@pytest.mark.asyncio
async def test_watchdog_stop_before_start():
    """stop() before start() is a no-op."""
    wd = ConnectionWatchdog(timeout=1.0)
    wd.stop()
    assert not wd.is_running


@pytest.mark.asyncio
async def test_watchdog_double_start():
    """Calling start() twice is a no-op (doesn't create duplicate tasks)."""
    callback = AsyncMock()
    wd = ConnectionWatchdog(timeout=0.2, on_timeout=callback)
    wd.start()
    task1 = wd._task
    wd.start()
    task2 = wd._task

    assert task1 is task2
    wd.stop()


@pytest.mark.asyncio
async def test_watchdog_no_callback():
    """Watchdog with no callback just logs and stops."""
    wd = ConnectionWatchdog(timeout=0.2, on_timeout=None)
    wd.start()

    await asyncio.sleep(0.5)

    # Should have stopped naturally after timeout
    assert not wd.is_running


@pytest.mark.asyncio
async def test_watchdog_callback_exception():
    """If the callback raises, the watchdog still stops cleanly."""

    async def bad_callback():
        raise RuntimeError("oops")

    wd = ConnectionWatchdog(timeout=0.2, on_timeout=bad_callback)
    wd.start()

    await asyncio.sleep(0.5)

    # Watchdog should have stopped even though callback failed
    assert not wd.is_running


@pytest.mark.asyncio
async def test_watchdog_last_activity_property():
    """last_activity is updated by notify_activity()."""
    wd = ConnectionWatchdog(timeout=10.0)
    assert wd.last_activity == 0.0

    wd.start()
    start_time = wd.last_activity
    assert start_time > 0.0

    await asyncio.sleep(0.05)
    wd.notify_activity()
    assert wd.last_activity > start_time

    wd.stop()


@pytest.mark.asyncio
async def test_watchdog_restart():
    """Watchdog can be stopped and restarted."""
    callback = AsyncMock()
    wd = ConnectionWatchdog(timeout=0.2, on_timeout=callback)

    wd.start()
    running_1 = wd.is_running
    assert running_1
    wd.stop()
    running_2 = wd.is_running
    assert not running_2

    # Restart
    wd.start()
    running_3 = wd.is_running
    assert running_3

    await asyncio.sleep(0.5)
    callback.assert_awaited_once()


def test_watchdog_is_importable_from_top_level():
    """ConnectionWatchdog is accessible from the top-level package."""
    from bleak_retry_connector import ConnectionWatchdog as CW

    assert CW is ConnectionWatchdog


@pytest.mark.asyncio
async def test_watchdog_cleanup_disconnects_and_clears_cache():
    """When client/device are provided, timeout disconnects and clears cache."""
    on_timeout = AsyncMock()
    mock_client = MagicMock()
    mock_client.disconnect = AsyncMock()
    mock_device = MagicMock()
    mock_device.address = "AA:BB:CC:DD:EE:FF"

    with patch(
        "bleak_retry_connector.watchdog.clear_cache", new_callable=AsyncMock
    ) as mock_clear:
        wd = ConnectionWatchdog(
            timeout=0.2,
            on_timeout=on_timeout,
            client=mock_client,
            device=mock_device,
        )
        wd.start()
        await asyncio.sleep(0.5)

    mock_client.disconnect.assert_awaited_once()
    mock_clear.assert_awaited_once_with("AA:BB:CC:DD:EE:FF")
    on_timeout.assert_awaited_once()
    assert not wd.is_running


@pytest.mark.asyncio
async def test_watchdog_cleanup_disconnect_timeout():
    """If disconnect hangs past 5s, watchdog proceeds to clear_cache."""
    on_timeout = AsyncMock()
    mock_client = MagicMock()

    async def slow_disconnect():
        await asyncio.sleep(60)

    mock_client.disconnect = slow_disconnect
    mock_device = MagicMock()
    mock_device.address = "AA:BB:CC:DD:EE:FF"

    with (
        patch(
            "bleak_retry_connector.watchdog.clear_cache", new_callable=AsyncMock
        ) as mock_clear,
        patch("bleak_retry_connector.watchdog.DISCONNECT_TIMEOUT", 0.1),
    ):
        wd = ConnectionWatchdog(
            timeout=0.2,
            on_timeout=on_timeout,
            client=mock_client,
            device=mock_device,
        )
        wd.start()
        await asyncio.sleep(1.0)

    mock_clear.assert_awaited_once_with("AA:BB:CC:DD:EE:FF")
    on_timeout.assert_awaited_once()
    assert not wd.is_running


@pytest.mark.asyncio
async def test_watchdog_cleanup_disconnect_exception():
    """If disconnect raises, watchdog proceeds to clear_cache and callback."""
    on_timeout = AsyncMock()
    mock_client = MagicMock()
    mock_client.disconnect = AsyncMock(side_effect=RuntimeError("dead"))
    mock_device = MagicMock()
    mock_device.address = "AA:BB:CC:DD:EE:FF"

    with patch(
        "bleak_retry_connector.watchdog.clear_cache", new_callable=AsyncMock
    ) as mock_clear:
        wd = ConnectionWatchdog(
            timeout=0.2,
            on_timeout=on_timeout,
            client=mock_client,
            device=mock_device,
        )
        wd.start()
        await asyncio.sleep(0.5)

    mock_clear.assert_awaited_once_with("AA:BB:CC:DD:EE:FF")
    on_timeout.assert_awaited_once()
    assert not wd.is_running


@pytest.mark.asyncio
async def test_watchdog_no_client_no_cleanup():
    """Without client/device, no cleanup happens (backward compat)."""
    on_timeout = AsyncMock()

    with patch(
        "bleak_retry_connector.watchdog.clear_cache", new_callable=AsyncMock
    ) as mock_clear:
        wd = ConnectionWatchdog(timeout=0.2, on_timeout=on_timeout)
        wd.start()
        await asyncio.sleep(0.5)

    mock_clear.assert_not_awaited()
    on_timeout.assert_awaited_once()
    assert not wd.is_running
