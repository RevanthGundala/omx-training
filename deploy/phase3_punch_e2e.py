"""Phase 3 end-to-end test — STUN + rendezvous + Rust hole punch.

Drives the real ``omx_quic.QuicClient`` and ``QuicServer`` through the
full Phase 0–3 flow: discover public addresses via STUN, swap them via
``omx_quic.rendezvous`` (Modal Dict), then run the punch loop on both
sides simultaneously and confirm packets flowed in both directions.

Usage:
  Terminal A (Modal):  modal run deploy/phase3_punch_e2e.py::server
  Terminal B (laptop): python deploy/phase3_punch_e2e.py client
"""

from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
CRATE_DIR = REPO_ROOT / "omx_quic"

# A fixed session id so client and server find each other in the dict.
# In production each connection gets a fresh uuid.
SESSION_ID = "phase3-e2e"

app = modal.App("omx-phase3-punch-e2e")

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


def run_role(role: str) -> dict:
    import omx_quic
    from omx_quic import rendezvous

    print(f"[{role}] omx_quic version: {omx_quic.version()}")

    if role == "client":
        endpoint = omx_quic.QuicClient(SESSION_ID)
    else:
        endpoint = omx_quic.QuicServer(SESSION_ID)

    print(f"[{role}] local UDP port: {endpoint.local_port()}")

    pub_ip, pub_port = endpoint.discover_public_address()
    print(f"[{role}] public address (STUN): {pub_ip}:{pub_port}")

    # Verify cone-NAT-like behavior with a 2nd STUN query.
    pub_ip2, pub_port2 = endpoint.discover_public_address()
    if (pub_ip2, pub_port2) != (pub_ip, pub_port):
        print(
            f"[{role}] WARNING: 2nd STUN saw {pub_ip2}:{pub_port2} (symmetric NAT?). "
            f"Hole punching may fail."
        )

    rendezvous.publish(SESSION_ID, role, pub_ip, pub_port)
    print(f"[{role}] published; waiting for peer...")

    peer_ip, peer_port = rendezvous.wait_for_peer(SESSION_ID, role, timeout_s=60.0)
    print(f"[{role}] peer address: {peer_ip}:{peer_port}")

    endpoint.set_peer_address(peer_ip, peer_port)

    print(f"[{role}] punching...")
    t0 = time.perf_counter()
    sent, received, elapsed = endpoint.punch(timeout_s=10.0)
    wallclock = time.perf_counter() - t0
    print(
        f"[{role}] punch RESULT: sent={sent} received={received} "
        f"elapsed={elapsed:.3f}s wallclock={wallclock:.3f}s"
    )

    rendezvous.clear(SESSION_ID, role)

    return {
        "role": role,
        "public": f"{pub_ip}:{pub_port}",
        "peer": f"{peer_ip}:{peer_port}",
        "sent": sent,
        "received": received,
        "elapsed_s": elapsed,
    }


@app.function(image=image, timeout=300)
def server() -> dict:
    return run_role("server")


@app.local_entrypoint()
def main():
    """`modal run deploy/phase3_punch_e2e.py::server` entrypoint."""
    result = server.remote()
    print("\n=== Modal server result ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
    if result["received"] > 0:
        print("\nPASS")
    else:
        print("\nFAIL")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "client":
        result = run_role("client")
        print("\n=== Local client result ===")
        for k, v in result.items():
            print(f"  {k}: {v}")
        ok = result["received"] > 0
        print("\n" + ("PASS" if ok else "FAIL"))
        sys.exit(0 if ok else 1)
    print(__doc__)
    sys.exit(1)
