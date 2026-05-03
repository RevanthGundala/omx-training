"""omx_quic — QUIC + UDP hole-punched transport for OMX policy inference.

Re-exports the Rust extension module so users can ``import omx_quic`` and
get the ``QuicClient`` / ``QuicServer`` classes directly.
"""

from ._omx_quic import QuicClient, QuicServer, version, DEFAULT_STUN_SERVER
from . import rendezvous

__all__ = ["QuicClient", "QuicServer", "version", "DEFAULT_STUN_SERVER", "rendezvous"]

