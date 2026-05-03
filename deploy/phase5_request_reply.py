"""Phase 5 smoke test — request/reply protocol without PI0.5.

Runs ``QuicServer.serve_forever`` against a dummy reverse-bytes handler
and verifies ``QuicClient.request`` round-trips correctly. This is the
last test before wiring up the real PI0.5 inference handler.

Usage:
  Modal:  modal run --detach deploy/phase5_request_reply.py::server
  Local:  python deploy/phase5_request_reply.py client
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
CRATE_DIR = REPO_ROOT / "omx_quic"
SESSION_ID = "phase5-reqrep"

app = modal.App("omx-phase5-reqrep")

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


@app.function(image=image, timeout=300)
def server() -> dict:
    import omx_quic
    from omx_quic import rendezvous
    s = omx_quic.QuicServer(SESSION_ID)
    pub = s.discover_public_address()
    print(f"[server] public {pub}")
    rendezvous.publish(SESSION_ID, "server", *pub)
    peer = rendezvous.wait_for_peer(SESSION_ID, "server", 60.0)
    print(f"[server] peer {peer}")
    s.set_peer_address(*peer)
    s.punch(timeout_s=10.0)
    s.listen(timeout_s=20.0)
    print("[server] connected; serve_forever()")

    def handler(req: bytes) -> bytes:
        return req[::-1]

    n = s.serve_forever(handler)
    rendezvous.clear(SESSION_ID, "server")
    return {"requests": int(n)}


def run_client() -> dict:
    import omx_quic
    from omx_quic import rendezvous
    c = omx_quic.QuicClient(SESSION_ID)
    pub = c.discover_public_address()
    print(f"[client] public {pub}")
    rendezvous.publish(SESSION_ID, "client", *pub)
    peer = rendezvous.wait_for_peer(SESSION_ID, "client", 60.0)
    print(f"[client] peer {peer}")
    c.set_peer_address(*peer)
    c.punch(timeout_s=10.0)
    c.connect(timeout_s=20.0)
    print("[client] connected")

    rtts = []
    for i in range(5):
        payload = f"hello-{i}".encode() * 100
        t0 = time.perf_counter()
        reply = c.request(payload, 10.0)
        rtt = (time.perf_counter() - t0) * 1000
        ok = reply == payload[::-1]
        rtts.append(rtt)
        print(f"[client] req {i}: {len(payload)}B rtt={rtt:.1f}ms match={ok}")
        if not ok:
            return {"ok": False}

    c.close()
    rendezvous.clear(SESSION_ID, "client")
    return {"ok": True, "rtt_ms_avg": sum(rtts) / len(rtts), "rtt_ms_min": min(rtts)}


@app.local_entrypoint()
def main():
    res = server.remote()
    print(f"\nServer: {res}")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "client":
        r = run_client()
        print(f"\n{r}")
        sys.exit(0 if r.get("ok") else 1)
    print(__doc__)
    sys.exit(1)
