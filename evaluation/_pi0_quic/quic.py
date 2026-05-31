"""QUIC client setup: STUN discovery, rendezvous, hole-punch, handshake."""

from __future__ import annotations

import time

import omx_quic
from omx_quic import rendezvous

from .colors import DIM, GREEN, color


def connect_quic(session_id: str, stun_server: str) -> omx_quic.QuicClient:
    client = omx_quic.QuicClient(session_id)
    print(color(f"[client] local UDP port: {client.local_port()}", DIM))
    pub_ip, pub_port = client.discover_public_address(stun_server)
    print(color(f"[client] public address: {pub_ip}:{pub_port}", DIM))
    pub_ip2, pub_port2 = client.discover_public_address(stun_server)
    if (pub_ip2, pub_port2) != (pub_ip, pub_port):
        raise RuntimeError(
            f"[client] symmetric NAT detected: STUN1={pub_ip}:{pub_port} "
            f"STUN2={pub_ip2}:{pub_port2}. Hole punching cannot work from "
            "this network. Try a different network or a TURN relay."
        )
    rendezvous.publish(session_id, "client", pub_ip, pub_port)
    try:
        print(color("[client] waiting for server peer in rendezvous dict ...", DIM))
        peer_ip, peer_port = rendezvous.wait_for_peer(session_id, "client", timeout_s=300.0)
        print(color(f"[client] peer: {peer_ip}:{peer_port}", DIM))
        client.set_peer_address(peer_ip, peer_port)
        sent, received, elapsed = client.punch(timeout_s=15.0)
        print(color(f"[client] punch ok: sent={sent} received={received} elapsed={elapsed:.3f}s", GREEN))
        print(color("[client] QUIC handshake ...", DIM))
        t0 = time.perf_counter()
        client.connect(timeout_s=30.0)
        print(color(f"[client] QUIC connected in {(time.perf_counter()-t0)*1000:.1f}ms", GREEN))
        return client
    except Exception:
        try:
            client.close()
        except Exception:
            pass
        raise
    finally:
        rendezvous.clear(session_id, "client")
