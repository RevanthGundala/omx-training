"""Modal-Dict-based rendezvous for QUIC peers.

Both endpoints write their public ``(ip, port)`` tuple (as discovered
via STUN) into a shared :class:`modal.Dict` keyed by session id and
role. The other side polls until it sees a fresh entry, reads it, and
proceeds to hole punching.

The dict is a one-shot bootstrap channel — once both sides have each
other's address, no further reads or writes happen. Real traffic flows
peer-to-peer over the punched UDP path (QUIC, in later phases).
"""

from __future__ import annotations

import json
import time
from typing import Literal

import modal

Role = Literal["client", "server"]
_OTHER: dict[Role, Role] = {"client": "server", "server": "client"}

# Stale entries from previous runs are ignored if older than this.
DEFAULT_FRESHNESS_S = 60.0
# Total time we'll poll for a peer entry before giving up.
DEFAULT_TIMEOUT_S = 60.0
# Polling interval.
_POLL_INTERVAL_S = 0.5


def _dict_for(session_id: str) -> modal.Dict:
    """Return (creating if necessary) the Modal Dict used as a rendezvous
    point for ``session_id``. Per-session dicts keep concurrent sessions
    isolated and let stale entries be garbage-collected by Modal's TTL."""
    return modal.Dict.from_name(
        f"omx-quic-rendezvous-{session_id}", create_if_missing=True
    )


def publish(
    session_id: str,
    role: Role,
    ip: str,
    port: int,
) -> None:
    """Write our public address into the rendezvous dict."""
    d = _dict_for(session_id)
    d[role] = json.dumps({"ip": ip, "port": int(port), "ts": time.time()})


def wait_for_peer(
    session_id: str,
    role: Role,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    freshness_s: float = DEFAULT_FRESHNESS_S,
) -> tuple[str, int]:
    """Block until the peer publishes a fresh entry, then return its
    ``(ip, port)``. Raises :class:`TimeoutError` if no fresh entry
    appears within ``timeout_s``."""
    peer_role = _OTHER[role]
    d = _dict_for(session_id)
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        try:
            raw = d[peer_role]
        except KeyError:
            raw = None
        if raw is not None:
            info = json.loads(raw)
            if (time.time() - float(info["ts"])) <= freshness_s:
                return str(info["ip"]), int(info["port"])
        time.sleep(_POLL_INTERVAL_S)

    raise TimeoutError(
        f"rendezvous: peer {peer_role!r} did not publish a fresh address "
        f"for session {session_id!r} within {timeout_s:.0f}s"
    )


def clear(session_id: str, role: Role) -> None:
    """Remove our role's entry — call after handshake completes so a
    later session reusing the same id doesn't pick up stale data."""
    d = _dict_for(session_id)
    try:
        del d[role]
    except KeyError:
        pass
