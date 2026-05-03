//! QUIC transport built on top of a hole-punched UDP socket using quinn.
//!
//! Phase 4 scope: bring up a QUIC connection between client and server
//! over the punched socket and demonstrate one bidirectional stream
//! roundtripping bytes (echo test). The three-channel split
//! (observation datagrams, action stream, control stream) is layered
//! on top of this in Phase 5.

use std::net::{SocketAddr, UdpSocket};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use quinn::crypto::rustls::{QuicClientConfig, QuicServerConfig};
use quinn::{ClientConfig, Connection, Endpoint, EndpointConfig, ServerConfig, TokioRuntime};
use rustls::pki_types::{CertificateDer, PrivatePkcs8KeyDer, ServerName, UnixTime};

const ALPN: &[u8] = b"omx-quic/1";

/// Ensure rustls's default crypto provider is installed exactly once.
/// Required for quinn 0.11 + rustls 0.23.
fn install_crypto_provider() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
}

/// Build a `quinn::Endpoint` from a pre-existing (hole-punched)
/// `std::net::UdpSocket`. The socket is consumed.
fn make_endpoint(
    socket: UdpSocket,
    server_config: Option<ServerConfig>,
) -> Result<Endpoint> {
    socket
        .set_nonblocking(true)
        .context("failed to set socket non-blocking")?;
    let endpoint = Endpoint::new(
        EndpointConfig::default(),
        server_config,
        socket,
        Arc::new(TokioRuntime),
    )
    .context("quinn::Endpoint::new failed")?;
    Ok(endpoint)
}

/// Generate a fresh self-signed cert and the matching server config.
fn make_server_config() -> Result<ServerConfig> {
    install_crypto_provider();

    let cert = rcgen::generate_simple_self_signed(vec!["omx-quic".to_string()])
        .context("rcgen self-signed cert generation failed")?;
    let cert_der = CertificateDer::from(cert.cert.der().to_vec());
    let key_der = PrivatePkcs8KeyDer::from(cert.key_pair.serialize_der());

    let mut crypto = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(vec![cert_der], key_der.into())
        .context("rustls server config")?;
    crypto.alpn_protocols = vec![ALPN.to_vec()];

    let quic_crypto = QuicServerConfig::try_from(crypto)
        .context("rustls -> quinn server crypto")?;
    let mut server_config = ServerConfig::with_crypto(Arc::new(quic_crypto));

    // Tight idle timeout so a dead peer is detected quickly.
    let mut transport = quinn::TransportConfig::default();
    transport.max_idle_timeout(Some(
        Duration::from_secs(30).try_into().unwrap(),
    ));
    transport.keep_alive_interval(Some(Duration::from_secs(5)));
    server_config.transport_config(Arc::new(transport));

    Ok(server_config)
}

/// Build a client config that accepts ANY server certificate. This is
/// safe-ish here because both endpoints authenticate each other out of
/// band via the Modal Dict rendezvous (only the holder of the session
/// secret can publish a peer address). A pre-shared session token will
/// be added in Phase 5 for stronger guarantees.
fn make_client_config() -> Result<ClientConfig> {
    install_crypto_provider();

    let mut crypto = rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(SkipServerVerification))
        .with_no_client_auth();
    crypto.alpn_protocols = vec![ALPN.to_vec()];

    let quic_crypto = QuicClientConfig::try_from(crypto)
        .context("rustls -> quinn client crypto")?;
    let mut client_config = ClientConfig::new(Arc::new(quic_crypto));

    let mut transport = quinn::TransportConfig::default();
    transport.max_idle_timeout(Some(
        Duration::from_secs(30).try_into().unwrap(),
    ));
    transport.keep_alive_interval(Some(Duration::from_secs(5)));
    client_config.transport_config(Arc::new(transport));

    Ok(client_config)
}

#[derive(Debug)]
struct SkipServerVerification;

impl rustls::client::danger::ServerCertVerifier for SkipServerVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> std::result::Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> std::result::Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> std::result::Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        // Match the algorithms the default ring provider supports.
        vec![
            rustls::SignatureScheme::RSA_PKCS1_SHA256,
            rustls::SignatureScheme::RSA_PKCS1_SHA384,
            rustls::SignatureScheme::RSA_PKCS1_SHA512,
            rustls::SignatureScheme::ECDSA_NISTP256_SHA256,
            rustls::SignatureScheme::ECDSA_NISTP384_SHA384,
            rustls::SignatureScheme::ED25519,
            rustls::SignatureScheme::RSA_PSS_SHA256,
            rustls::SignatureScheme::RSA_PSS_SHA384,
            rustls::SignatureScheme::RSA_PSS_SHA512,
        ]
    }
}

/// Server: bring up an Endpoint configured to accept incoming
/// connections, then wait for and finalize one connection from `peer`.
pub async fn server_accept(socket: UdpSocket, peer: SocketAddr) -> Result<Connection> {
    let server_config = make_server_config()?;
    let endpoint = make_endpoint(socket, Some(server_config))?;

    loop {
        let incoming = endpoint
            .accept()
            .await
            .ok_or_else(|| anyhow!("endpoint closed before any incoming connection"))?;
        let remote = incoming.remote_address();
        if remote.ip() != peer.ip() {
            // Drop unrelated traffic. In a real deployment we'd log this.
            incoming.ignore();
            continue;
        }
        let connection = incoming.await.context("accepting incoming connection")?;
        return Ok(connection);
    }
}

/// Client: bring up an Endpoint and complete the handshake to `peer`.
pub async fn client_connect(socket: UdpSocket, peer: SocketAddr) -> Result<Connection> {
    let client_config = make_client_config()?;
    let mut endpoint = make_endpoint(socket, None)?;
    endpoint.set_default_client_config(client_config);

    let connecting = endpoint
        .connect(peer, "omx-quic")
        .context("endpoint.connect failed")?;
    let connection = connecting.await.context("client handshake")?;
    Ok(connection)
}

/// Server-side echo loop: accept one bidirectional stream, read all
/// data sent on it, write it back, then return.
pub async fn server_echo_once(connection: &Connection) -> Result<u64> {
    let (mut send, mut recv) = connection
        .accept_bi()
        .await
        .context("accept_bi failed")?;
    let data = recv
        .read_to_end(64 * 1024)
        .await
        .context("read_to_end on echo stream")?;
    let n = data.len() as u64;
    send.write_all(&data).await.context("echo write_all")?;
    send.finish().context("echo finish")?;
    Ok(n)
}

/// Client-side echo: open a bidirectional stream, send `payload`, read
/// the echoed bytes back, return them.
pub async fn client_echo(connection: &Connection, payload: &[u8]) -> Result<Vec<u8>> {
    let (mut send, mut recv) = connection
        .open_bi()
        .await
        .context("open_bi failed")?;
    send.write_all(payload).await.context("client echo send")?;
    send.finish().context("client echo finish")?;
    let echoed = recv
        .read_to_end(64 * 1024)
        .await
        .context("client echo recv")?;
    Ok(echoed)
}

/// Maximum request/response size for the inference protocol (4 MB).
/// Covers JPEG-compressed observations + action chunks comfortably.
const MAX_MESSAGE_BYTES: usize = 4 * 1024 * 1024;

/// Client-side request: open a bidi stream, write `request` bytes,
/// finish, read response, return it. One round-trip per call.
pub async fn client_request(connection: &Connection, request: &[u8]) -> Result<Vec<u8>> {
    let (mut send, mut recv) = connection
        .open_bi()
        .await
        .context("open_bi failed")?;
    send.write_all(request).await.context("write request")?;
    send.finish().context("finish request")?;
    let response = recv
        .read_to_end(MAX_MESSAGE_BYTES)
        .await
        .context("read response")?;
    Ok(response)
}

/// Server-side: accept one bidi stream, read the request bytes, return
/// them along with the send half (so the caller can write the reply).
/// Returns `Ok(None)` when the connection is closing.
pub async fn server_accept_request(
    connection: &Connection,
) -> Result<Option<(Vec<u8>, quinn::SendStream)>> {
    let (send, mut recv) = match connection.accept_bi().await {
        Ok(pair) => pair,
        Err(quinn::ConnectionError::ApplicationClosed(_))
        | Err(quinn::ConnectionError::ConnectionClosed(_))
        | Err(quinn::ConnectionError::LocallyClosed) => return Ok(None),
        Err(e) => return Err(anyhow!("accept_bi failed: {e}")),
    };
    let request = recv
        .read_to_end(MAX_MESSAGE_BYTES)
        .await
        .context("read request")?;
    Ok(Some((request, send)))
}

/// Server-side: write reply bytes on the send half and finish the
/// stream.
pub async fn server_send_reply(mut send: quinn::SendStream, reply: &[u8]) -> Result<()> {
    send.write_all(reply).await.context("write reply")?;
    send.finish().context("finish reply")?;
    Ok(())
}
