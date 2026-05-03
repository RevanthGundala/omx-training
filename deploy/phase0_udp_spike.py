"""
Phase 0 — UDP feasibility kill-switch for the QUIC inference plan.

Validates three things:
  1. A Modal container can do outbound UDP to a public STUN server and
     discover its own public address.
  2. A local laptop can do the same.
  3. After exchanging addresses via a shared Modal Dict, two-way UDP
     packets can flow directly between the laptop and the Modal
     container (i.e. NAT hole punching works in this network topology).

If any of these fail, the QUIC + hole-punching plan is infeasible and
we should pivot to "Modal Tunnel + websocket" instead.

Usage:
  Terminal A (Modal):  modal run deploy/phase0_udp_spike.py::server
  Terminal B (laptop): python deploy/phase0_udp_spike.py client

Both sides should be started within ~60s of each other.
"""

from __future__ import annotations

import json
import secrets
import socket
import struct
import sys
import time

import modal

app = modal.App("omx-phase0-udp-spike")

# Shared key/value store used as a one-time rendezvous point so the two
# sides can swap public addresses. Created on first use.
shared_dict = modal.Dict.from_name("omx-phase0-spike", create_if_missing=True)

STUN_HOST = "stun.l.google.com"
STUN_PORT = 19302
STUN_MAGIC_COOKIE = 0x2112A442

PROBE_DURATION_S = 8.0
SEND_INTERVAL_S = 0.05
HANDSHAKE_TIMEOUT_S = 60.0
PEER_FRESHNESS_S = 60.0


def stun_discover(sock: socket.socket) -> tuple[str, int]:
    """Send a STUN binding request and return the public (ip, port)."""
    txn_id = secrets.token_bytes(12)
    request = struct.pack("!HHI", 0x0001, 0x0000, STUN_MAGIC_COOKIE) + txn_id
    sock.sendto(request, (STUN_HOST, STUN_PORT))

    sock.settimeout(5.0)
    try:
        data, _ = sock.recvfrom(2048)
    finally:
        sock.settimeout(None)

    msg_type, msg_len, magic = struct.unpack("!HHI", data[:8])
    if msg_type != 0x0101:
        raise RuntimeError(
            f"STUN: expected binding response 0x0101, got 0x{msg_type:04x}"
        )
    if magic != STUN_MAGIC_COOKIE:
        raise RuntimeError("STUN: magic cookie mismatch")
    if data[8:20] != txn_id:
        raise RuntimeError("STUN: transaction ID mismatch")

    pos = 20
    end = 20 + msg_len
    while pos < end:
        attr_type, attr_len = struct.unpack("!HH", data[pos:pos + 4])
        attr_val = data[pos + 4:pos + 4 + attr_len]

        # XOR-MAPPED-ADDRESS — the de-facto modern attribute. Anchor STUN
        # servers (Google, Cloudflare) always include it.
        if attr_type == 0x0020:
            family = attr_val[1]
            xport = struct.unpack("!H", attr_val[2:4])[0]
            port = xport ^ (STUN_MAGIC_COOKIE >> 16)
            if family == 0x01:
                xaddr = struct.unpack("!I", attr_val[4:8])[0]
                addr_int = xaddr ^ STUN_MAGIC_COOKIE
                ip = socket.inet_ntoa(struct.pack("!I", addr_int))
                return ip, port
            raise RuntimeError("STUN: only IPv4 supported in this spike")

        pos += 4 + ((attr_len + 3) & ~3)

    raise RuntimeError("STUN: no XOR-MAPPED-ADDRESS attribute in response")


def _wait_for_peer(role: str, peer_role: str) -> tuple[str, int] | None:
    deadline = time.time() + HANDSHAKE_TIMEOUT_S
    while time.time() < deadline:
        try:
            raw = shared_dict[peer_role]
        except KeyError:
            raw = None
        if raw is not None:
            info = json.loads(raw)
            if (time.time() - float(info["ts"])) <= PEER_FRESHNESS_S:
                return info["ip"], int(info["port"])
        time.sleep(0.5)
    return None


def run_probe(role: str) -> bool:
    assert role in ("server", "client")
    peer_role = "client" if role == "server" else "server"

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", 0))
    local_port = sock.getsockname()[1]
    print(f"[{role}] Bound local UDP socket on 0.0.0.0:{local_port}")

    print(f"[{role}] Discovering public address via STUN...")
    try:
        public_ip, public_port = stun_discover(sock)
    except Exception as e:
        print(f"[{role}] STUN FAILED: {e!r}")
        sock.close()
        return False
    print(f"[{role}] Public address: {public_ip}:{public_port}")

    # Verify we get the same port if we ask STUN twice (= cone NAT).
    # If the port differs, we're behind a symmetric NAT and hole punching
    # to a third-party peer will not work.
    try:
        ip2, port2 = stun_discover(sock)
        if (ip2, port2) != (public_ip, public_port):
            print(
                f"[{role}] WARNING: symmetric-NAT-like behavior detected "
                f"(2nd STUN saw {ip2}:{port2}). Hole punching to a "
                f"different peer may not work."
            )
    except Exception as e:
        print(f"[{role}] 2nd STUN check skipped: {e!r}")

    shared_dict[role] = json.dumps({
        "ip": public_ip,
        "port": public_port,
        "ts": time.time(),
    })
    print(f"[{role}] Published address to Modal Dict; waiting for {peer_role}...")

    peer = _wait_for_peer(role, peer_role)
    if peer is None:
        print(f"[{role}] FAIL: peer never published a fresh address within "
              f"{HANDSHAKE_TIMEOUT_S:.0f}s")
        sock.close()
        return False
    peer_ip, peer_port = peer
    print(f"[{role}] Peer address: {peer_ip}:{peer_port}")

    print(f"[{role}] Punching: send/recv UDP for {PROBE_DURATION_S:.1f}s...")
    sock.settimeout(SEND_INTERVAL_S)
    sent = 0
    received = 0
    first_recv_t: float | None = None
    last_send_t = 0.0
    start = time.time()

    while time.time() - start < PROBE_DURATION_S:
        now = time.time()
        if now - last_send_t >= SEND_INTERVAL_S:
            try:
                sock.sendto(f"{role}:{sent}".encode(), (peer_ip, peer_port))
                sent += 1
            except OSError as e:
                print(f"[{role}] sendto error: {e!r}")
            last_send_t = now

        try:
            data, src = sock.recvfrom(2048)
        except socket.timeout:
            continue
        except OSError as e:
            print(f"[{role}] recvfrom error: {e!r}")
            continue

        if first_recv_t is None:
            first_recv_t = now - start
            print(f"[{role}] FIRST PACKET at t+{first_recv_t:.3f}s "
                  f"from {src!r}: {data!r}")
        received += 1

    sock.close()
    print(f"[{role}] Done. sent={sent} received={received} "
          f"first_recv={'%.3fs' % first_recv_t if first_recv_t is not None else 'NEVER'}")

    try:
        del shared_dict[role]
    except KeyError:
        pass

    return received > 0


image = modal.Image.debian_slim(python_version="3.12")


@app.function(image=image, timeout=180)
def server() -> bool:
    ok = run_probe("server")
    print(f"\n[server] RESULT: {'PASS' if ok else 'FAIL'}\n")
    return ok


@app.local_entrypoint()
def main():
    """Default entrypoint when invoked as `modal run deploy/phase0_udp_spike.py`."""
    result = server.remote()
    print(f"[driver] server returned: {result}")


if __name__ == "__main__":
    # Local laptop side: `python deploy/phase0_udp_spike.py client`
    if len(sys.argv) >= 2 and sys.argv[1] == "client":
        ok = run_probe("client")
        print(f"\n[client] RESULT: {'PASS' if ok else 'FAIL'}\n")
        sys.exit(0 if ok else 1)
    print(__doc__)
    sys.exit(1)
