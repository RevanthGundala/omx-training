"""Phase 4 end-to-end test — STUN + rendezvous + punch + QUIC echo.

Drives the full Phase 0–4 flow on real Modal+laptop networking and
verifies a bidirectional QUIC stream round-trips bytes correctly.

Usage:
  Terminal A (Modal):  modal run deploy/phase4_quic_echo.py::server
  Terminal B (laptop): python deploy/phase4_quic_echo.py client
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
CRATE_DIR = REPO_ROOT / "omx_quic"

SESSION_ID = "phase4-echo"

app = modal.App("omx-phase4-quic-echo")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "build-essential", "pkg-config")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | "
        "sh -s -- -y --default-toolchain stable --profile minimal",
    )
    .env({"PATH": "/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin"})
    .pip_install("maturin>=1.7,<2.0")
    .add_local_dir(str(CRATE_DIR), remote_path="/build/omx_quic", copy=True)
    .run_commands(
        "cd /build/omx_quic && maturin build --release --out /build/wheels",
        "pip install /build/wheels/*.whl",
    )
)


def run_server() -> dict:
    import omx_quic
    from omx_quic import rendezvous

    print(f"[server] omx_quic version: {omx_quic.version()}")
    s = omx_quic.QuicServer(SESSION_ID)
    print(f"[server] local UDP port: {s.local_port()}")

    pub_ip, pub_port = s.discover_public_address()
    print(f"[server] public address: {pub_ip}:{pub_port}")

    rendezvous.publish(SESSION_ID, "server", pub_ip, pub_port)
    peer_ip, peer_port = rendezvous.wait_for_peer(SESSION_ID, "server", timeout_s=60.0)
    print(f"[server] peer address: {peer_ip}:{peer_port}")
    s.set_peer_address(peer_ip, peer_port)

    print("[server] punching...")
    sent, received, elapsed = s.punch(timeout_s=10.0)
    print(f"[server] punch: sent={sent} received={received} elapsed={elapsed:.3f}s")

    print("[server] accepting QUIC connection...")
    t0 = time.perf_counter()
    s.listen(timeout_s=20.0)
    handshake_s = time.perf_counter() - t0
    print(f"[server] QUIC connection accepted in {handshake_s*1000:.1f}ms")

    print("[server] echoing one bidi stream...")
    n = s.echo_once()
    print(f"[server] echoed {n} bytes")

    rendezvous.clear(SESSION_ID, "server")
    return {
        "punch_elapsed_s": elapsed,
        "handshake_s": handshake_s,
        "echoed_bytes": int(n),
    }


def run_client() -> dict:
    import omx_quic
    from omx_quic import rendezvous

    print(f"[client] omx_quic version: {omx_quic.version()}")
    c = omx_quic.QuicClient(SESSION_ID)
    print(f"[client] local UDP port: {c.local_port()}")

    pub_ip, pub_port = c.discover_public_address()
    print(f"[client] public address: {pub_ip}:{pub_port}")

    rendezvous.publish(SESSION_ID, "client", pub_ip, pub_port)
    peer_ip, peer_port = rendezvous.wait_for_peer(SESSION_ID, "client", timeout_s=60.0)
    print(f"[client] peer address: {peer_ip}:{peer_port}")
    c.set_peer_address(peer_ip, peer_port)

    print("[client] punching...")
    sent, received, elapsed = c.punch(timeout_s=10.0)
    print(f"[client] punch: sent={sent} received={received} elapsed={elapsed:.3f}s")

    print("[client] connecting QUIC...")
    t0 = time.perf_counter()
    c.connect(timeout_s=20.0)
    handshake_s = time.perf_counter() - t0
    print(f"[client] QUIC connected in {handshake_s*1000:.1f}ms")

    payload = b"the quick brown fox jumps over the lazy dog" * 50  # ~2 KB
    print(f"[client] echoing {len(payload)} bytes...")
    t0 = time.perf_counter()
    echoed = c.echo(payload)
    rtt = time.perf_counter() - t0
    print(f"[client] echo round-trip {rtt*1000:.1f}ms, "
          f"echoed={len(echoed)} bytes, match={echoed == payload}")

    rendezvous.clear(SESSION_ID, "client")
    return {
        "punch_elapsed_s": elapsed,
        "handshake_s": handshake_s,
        "echo_rtt_s": rtt,
        "payload_bytes": len(payload),
        "match": echoed == payload,
    }


@app.function(image=image, timeout=300)
def server() -> dict:
    return run_server()


@app.local_entrypoint()
def main():
    result = server.remote()
    print("\n=== Modal server result ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print("\nPASS" if result["echoed_bytes"] > 0 else "\nFAIL")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "client":
        result = run_client()
        print("\n=== Local client result ===")
        for k, v in result.items():
            print(f"  {k}: {v}")
        ok = bool(result["match"])
        print("\n" + ("PASS" if ok else "FAIL"))
        sys.exit(0 if ok else 1)
    print(__doc__)
    sys.exit(1)
