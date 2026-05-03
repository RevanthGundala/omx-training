//! Simultaneous-send UDP hole punching.
//!
//! Once both peers know each other's public address (via STUN +
//! rendezvous), they each begin spamming small UDP packets at the
//! other. Every outbound packet opens a temporary mapping in the
//! sender's NAT for return traffic from the destination address. As
//! soon as both sides have sent at least one packet, both NATs have a
//! return-path mapping and packets flow in both directions.
//!
//! This implementation:
//!   - Sends a small marker payload at a fixed cadence.
//!   - Polls for incoming packets on the same socket.
//!   - Returns success the moment we receive any packet from the peer
//!     address (we ignore packets from anywhere else — could be late
//!     STUN replies or unrelated traffic).
//!   - Returns failure if no peer packet arrives within the timeout.

use std::io;
use std::net::{SocketAddr, UdpSocket};
use std::time::{Duration, Instant};

/// Magic prefix so we can distinguish punch packets from any stray UDP
/// noise that happens to arrive on the same socket.
const PUNCH_MAGIC: &[u8; 8] = b"OMXPUNCH";

const SEND_INTERVAL: Duration = Duration::from_millis(50);
const RECV_POLL: Duration = Duration::from_millis(20);

#[derive(Debug, Clone, Copy)]
pub struct PunchStats {
    pub sent: u32,
    pub received: u32,
    pub elapsed: Duration,
}

/// Run the punch loop on `socket` against `peer`. Blocks until either
/// a packet is received from `peer` (success) or `timeout` elapses
/// (failure).
pub fn punch(
    socket: &UdpSocket,
    peer: SocketAddr,
    timeout: Duration,
) -> io::Result<PunchStats> {
    let prev_read_timeout = socket.read_timeout()?;
    socket.set_read_timeout(Some(RECV_POLL))?;

    let mut buf = [0u8; 64];
    let mut sent: u32 = 0;
    let mut received: u32 = 0;
    let mut last_send = Instant::now() - SEND_INTERVAL;
    let start = Instant::now();
    let mut got_peer = false;

    while start.elapsed() < timeout {
        if last_send.elapsed() >= SEND_INTERVAL {
            let mut payload = Vec::with_capacity(PUNCH_MAGIC.len() + 4);
            payload.extend_from_slice(PUNCH_MAGIC);
            payload.extend_from_slice(&sent.to_be_bytes());
            // Best-effort send; transient errors during punching are
            // expected (the peer's NAT is likely still closed).
            let _ = socket.send_to(&payload, peer);
            sent = sent.wrapping_add(1);
            last_send = Instant::now();
        }

        match socket.recv_from(&mut buf) {
            Ok((n, src)) => {
                if src == peer && n >= PUNCH_MAGIC.len() && &buf[..PUNCH_MAGIC.len()] == PUNCH_MAGIC
                {
                    received = received.saturating_add(1);
                    got_peer = true;
                }
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock || e.kind() == io::ErrorKind::TimedOut => {
                // No packet this poll; keep looping.
            }
            Err(e) => {
                socket.set_read_timeout(prev_read_timeout)?;
                return Err(e);
            }
        }

        // Drain any additional packets already queued so we don't have
        // to wait another RECV_POLL for them on the next iteration.
        if got_peer {
            // Brief grace period so the peer also receives at least one
            // packet from us before we hand the socket off to QUIC.
            // 200ms is well under any sane higher-level timeout but
            // long enough for ~4 of our 50ms-spaced packets.
            let grace_end = Instant::now() + Duration::from_millis(200);
            while Instant::now() < grace_end {
                if last_send.elapsed() >= SEND_INTERVAL {
                    let mut payload = Vec::with_capacity(PUNCH_MAGIC.len() + 4);
                    payload.extend_from_slice(PUNCH_MAGIC);
                    payload.extend_from_slice(&sent.to_be_bytes());
                    let _ = socket.send_to(&payload, peer);
                    sent = sent.wrapping_add(1);
                    last_send = Instant::now();
                }
                match socket.recv_from(&mut buf) {
                    Ok((n, src)) if src == peer
                        && n >= PUNCH_MAGIC.len()
                        && &buf[..PUNCH_MAGIC.len()] == PUNCH_MAGIC =>
                    {
                        received = received.saturating_add(1);
                    }
                    _ => {}
                }
            }
            socket.set_read_timeout(prev_read_timeout)?;
            return Ok(PunchStats {
                sent,
                received,
                elapsed: start.elapsed(),
            });
        }
    }

    socket.set_read_timeout(prev_read_timeout)?;
    Err(io::Error::new(
        io::ErrorKind::TimedOut,
        format!(
            "hole punch to {peer} timed out after {:.1}s (sent {sent} packets, received 0 from peer)",
            timeout.as_secs_f32()
        ),
    ))
}
