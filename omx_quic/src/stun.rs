//! STUN binding request implementation for public-address discovery.
//!
//! Implements RFC 5389/8489 well enough to talk to public STUN servers
//! (Google, Cloudflare). Only IPv4 + XOR-MAPPED-ADDRESS is supported,
//! which is all every modern STUN server returns.

use std::io;
use std::net::{Ipv4Addr, SocketAddr, ToSocketAddrs, UdpSocket};
use std::time::Duration;

use rand::RngCore;

const STUN_BINDING_REQUEST: u16 = 0x0001;
const STUN_BINDING_RESPONSE: u16 = 0x0101;
const STUN_MAGIC_COOKIE: u32 = 0x2112_A442;
const STUN_ATTR_XOR_MAPPED_ADDRESS: u16 = 0x0020;
const STUN_HEADER_LEN: usize = 20;

/// Send a STUN binding request through `socket` to `server` and return
/// the public `(ip, port)` the server saw the packet arrive from.
///
/// `socket` is left bound and unmodified so the same NAT mapping can be
/// reused for subsequent hole punching and QUIC traffic.
pub fn discover_public_address<A: ToSocketAddrs>(
    socket: &UdpSocket,
    server: A,
    timeout: Duration,
) -> io::Result<SocketAddr> {
    let server_addr = server
        .to_socket_addrs()?
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "STUN: empty server addr"))?;

    let mut txn_id = [0u8; 12];
    rand::rng().fill_bytes(&mut txn_id);

    let mut request = Vec::with_capacity(STUN_HEADER_LEN);
    request.extend_from_slice(&STUN_BINDING_REQUEST.to_be_bytes());
    request.extend_from_slice(&0u16.to_be_bytes()); // length: 0 attributes
    request.extend_from_slice(&STUN_MAGIC_COOKIE.to_be_bytes());
    request.extend_from_slice(&txn_id);

    socket.send_to(&request, server_addr)?;

    let prev_timeout = socket.read_timeout()?;
    socket.set_read_timeout(Some(timeout))?;

    let mut buf = [0u8; 2048];
    let result = recv_and_parse(socket, &mut buf, &txn_id);

    socket.set_read_timeout(prev_timeout)?;
    result
}

fn recv_and_parse(
    socket: &UdpSocket,
    buf: &mut [u8],
    expected_txn: &[u8; 12],
) -> io::Result<SocketAddr> {
    let (n, _) = socket.recv_from(buf)?;
    let data = &buf[..n];

    if data.len() < STUN_HEADER_LEN {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("STUN: response too short ({} bytes)", data.len()),
        ));
    }

    let msg_type = u16::from_be_bytes([data[0], data[1]]);
    let msg_len = u16::from_be_bytes([data[2], data[3]]) as usize;
    let magic = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);

    if msg_type != STUN_BINDING_RESPONSE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("STUN: expected 0x{STUN_BINDING_RESPONSE:04x}, got 0x{msg_type:04x}"),
        ));
    }
    if magic != STUN_MAGIC_COOKIE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "STUN: magic cookie mismatch",
        ));
    }
    if &data[8..STUN_HEADER_LEN] != expected_txn {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "STUN: transaction id mismatch",
        ));
    }
    if data.len() < STUN_HEADER_LEN + msg_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "STUN: declared length exceeds buffer",
        ));
    }

    let mut pos = STUN_HEADER_LEN;
    let end = STUN_HEADER_LEN + msg_len;
    while pos + 4 <= end {
        let attr_type = u16::from_be_bytes([data[pos], data[pos + 1]]);
        let attr_len = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
        let val_start = pos + 4;
        let val_end = val_start + attr_len;
        if val_end > end {
            break;
        }

        if attr_type == STUN_ATTR_XOR_MAPPED_ADDRESS && attr_len >= 8 {
            let family = data[val_start + 1];
            let xport = u16::from_be_bytes([data[val_start + 2], data[val_start + 3]]);
            let port = xport ^ ((STUN_MAGIC_COOKIE >> 16) as u16);

            if family == 0x01 {
                let xaddr = u32::from_be_bytes([
                    data[val_start + 4],
                    data[val_start + 5],
                    data[val_start + 6],
                    data[val_start + 7],
                ]);
                let addr_bits = xaddr ^ STUN_MAGIC_COOKIE;
                let ip = Ipv4Addr::from(addr_bits);
                return Ok(SocketAddr::new(ip.into(), port));
            }
        }

        // Pad attribute to 4-byte boundary
        pos = val_start + ((attr_len + 3) & !3);
    }

    Err(io::Error::new(
        io::ErrorKind::InvalidData,
        "STUN: no XOR-MAPPED-ADDRESS attribute in response",
    ))
}
