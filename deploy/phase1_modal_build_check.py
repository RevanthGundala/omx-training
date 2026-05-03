"""Phase 1 sanity check — verify the omx_quic Rust extension builds and
imports inside a Modal container image (linux/amd64).

Usage: modal run deploy/phase1_modal_build_check.py
"""
from pathlib import Path

import modal

app = modal.App("omx-phase1-build-check")

REPO_ROOT = Path(__file__).resolve().parent.parent
CRATE_DIR = REPO_ROOT / "omx_quic"

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


@app.function(image=image, timeout=600)
def check() -> dict:
    import omx_quic
    c = omx_quic.QuicClient("modal-session")
    c.connect()
    s = omx_quic.QuicServer("modal-session")
    s.listen()
    return {
        "version": omx_quic.version(),
        "client_repr": repr(c),
        "server_repr": repr(s),
        "client_connected": c.is_connected(),
        "server_listening": s.is_listening(),
    }


@app.local_entrypoint()
def main():
    result = check.remote()
    print("Phase 1 Modal build check:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print("PASS")
