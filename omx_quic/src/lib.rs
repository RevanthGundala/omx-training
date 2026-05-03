//! omx_quic — QUIC + UDP hole-punched transport for OMX policy inference.
//!
//! Phase 4 status: STUN discovery, UDP hole punching, and a QUIC
//! transport built on top of the punched socket are all implemented.
//! Demonstrated by an echo round-trip on a single bidirectional QUIC
//! stream. Three-channel split (datagrams, action stream, control
//! stream) is layered on top in Phase 5.

mod punch;
mod quic;
mod stun;

use std::net::{SocketAddr, UdpSocket};
use std::sync::Arc;
use std::time::Duration;

use pyo3::exceptions::{PyConnectionError, PyRuntimeError, PyTimeoutError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use tokio::runtime::Runtime;

const DEFAULT_STUN_SERVER: &str = "stun.l.google.com:19302";
const STUN_TIMEOUT: Duration = Duration::from_secs(5);
const DEFAULT_PUNCH_TIMEOUT: Duration = Duration::from_secs(8);

fn map_quic_err(e: anyhow::Error) -> PyErr {
    PyConnectionError::new_err(format!("{e:#}"))
}

/// Common state shared by both endpoint roles.
struct Endpoint {
    socket: Option<UdpSocket>,
    public_addr: Option<SocketAddr>,
    peer_addr: Option<SocketAddr>,
    runtime: Option<Arc<Runtime>>,
    connection: Option<quinn::Connection>,
}

impl Endpoint {
    fn new() -> PyResult<Self> {
        let socket = UdpSocket::bind("0.0.0.0:0")
            .map_err(|e| PyRuntimeError::new_err(format!("UDP bind failed: {e}")))?;
        Ok(Self {
            socket: Some(socket),
            public_addr: None,
            peer_addr: None,
            runtime: None,
            connection: None,
        })
    }

    fn socket_ref(&self) -> PyResult<&UdpSocket> {
        self.socket.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err(
                "socket has already been handed to the QUIC transport; \
                 STUN/punch operations are no longer permitted",
            )
        })
    }

    fn local_port(&self) -> PyResult<u16> {
        self.socket_ref()?
            .local_addr()
            .map(|a| a.port())
            .map_err(|e| PyConnectionError::new_err(format!("local_addr failed: {e}")))
    }

    fn discover(&mut self, server: &str) -> PyResult<(String, u16)> {
        let socket = self.socket_ref()?;
        let addr = stun::discover_public_address(socket, server, STUN_TIMEOUT)
            .map_err(|e| PyConnectionError::new_err(format!("STUN failed: {e}")))?;
        self.public_addr = Some(addr);
        Ok((addr.ip().to_string(), addr.port()))
    }

    fn set_peer(&mut self, ip: &str, port: u16) -> PyResult<()> {
        let parsed: SocketAddr = format!("{ip}:{port}")
            .parse()
            .map_err(|e| PyValueError::new_err(format!("invalid peer address {ip}:{port}: {e}")))?;
        self.peer_addr = Some(parsed);
        Ok(())
    }

    fn punch(&self, timeout: Duration) -> PyResult<(u32, u32, f64)> {
        let socket = self.socket_ref()?;
        let peer = self.peer_addr.ok_or_else(|| {
            PyRuntimeError::new_err("set_peer_address must be called before punch()")
        })?;
        let stats = Python::with_gil(|py| {
            py.allow_threads(|| punch::punch(socket, peer, timeout))
        })
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::TimedOut {
                PyTimeoutError::new_err(format!("{e}"))
            } else {
                PyConnectionError::new_err(format!("punch failed: {e}"))
            }
        })?;
        Ok((stats.sent, stats.received, stats.elapsed.as_secs_f64()))
    }

    fn ensure_runtime(&mut self) -> PyResult<Arc<Runtime>> {
        if let Some(rt) = &self.runtime {
            return Ok(rt.clone());
        }
        let rt = Runtime::new().map_err(|e| {
            PyRuntimeError::new_err(format!("failed to start tokio runtime: {e}"))
        })?;
        let rt = Arc::new(rt);
        self.runtime = Some(rt.clone());
        Ok(rt)
    }

    fn take_socket(&mut self) -> PyResult<UdpSocket> {
        self.socket
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("socket already taken by QUIC transport"))
    }

    fn require_peer(&self) -> PyResult<SocketAddr> {
        self.peer_addr
            .ok_or_else(|| PyRuntimeError::new_err("peer address not set"))
    }

    fn require_connection(&self) -> PyResult<&quinn::Connection> {
        self.connection
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("QUIC connection not established"))
    }
}

/// Client-side handle (the robot/laptop end).
#[pyclass]
struct QuicClient {
    session_id: String,
    endpoint: Endpoint,
    connected: bool,
}

#[pymethods]
impl QuicClient {
    #[new]
    fn new(session_id: String) -> PyResult<Self> {
        Ok(Self {
            session_id,
            endpoint: Endpoint::new()?,
            connected: false,
        })
    }

    fn local_port(&self) -> PyResult<u16> {
        self.endpoint.local_port()
    }

    /// Run a STUN binding request to learn the public (ip, port) of
    /// our local socket. Returns the discovered tuple as a Python
    /// `(str, int)`. The socket remains bound and the NAT mapping is
    /// retained for subsequent steps.
    #[pyo3(signature = (server = None))]
    fn discover_public_address(&mut self, server: Option<&str>) -> PyResult<(String, u16)> {
        self.endpoint.discover(server.unwrap_or(DEFAULT_STUN_SERVER))
    }

    fn public_address(&self) -> Option<(String, u16)> {
        self.endpoint
            .public_addr
            .map(|a| (a.ip().to_string(), a.port()))
    }

    fn set_peer_address(&mut self, ip: &str, port: u16) -> PyResult<()> {
        self.endpoint.set_peer(ip, port)
    }

    fn peer_address(&self) -> Option<(String, u16)> {
        self.endpoint
            .peer_addr
            .map(|a| (a.ip().to_string(), a.port()))
    }

    /// Run the simultaneous-send hole-punch loop against the peer.
    /// Returns ``(sent, received, elapsed_seconds)``. Raises
    /// ``TimeoutError`` if no packet from the peer arrives in
    /// ``timeout_s`` seconds (default 8s).
    #[pyo3(signature = (timeout_s = 8.0))]
    fn punch(&self, timeout_s: f64) -> PyResult<(u32, u32, f64)> {
        self.endpoint.punch(Duration::from_secs_f64(timeout_s))
    }

    /// Hand the punched socket to quinn and complete the QUIC handshake
    /// with the peer. Must be called after ``set_peer_address`` and
    /// (typically) after ``punch``.
    #[pyo3(signature = (timeout_s = 20.0))]
    fn connect(&mut self, timeout_s: f64) -> PyResult<()> {
        let peer = self.endpoint.require_peer()?;
        let runtime = self.endpoint.ensure_runtime()?;
        let socket = self.endpoint.take_socket()?;

        let connection = Python::with_gil(|py| {
            py.allow_threads(|| {
                runtime.block_on(async move {
                    tokio::time::timeout(
                        Duration::from_secs_f64(timeout_s),
                        quic::client_connect(socket, peer),
                    )
                    .await
                })
            })
        });

        let connection = match connection {
            Ok(Ok(c)) => c,
            Ok(Err(e)) => return Err(map_quic_err(e)),
            Err(_) => {
                return Err(PyTimeoutError::new_err(format!(
                    "QUIC handshake to {peer} timed out after {timeout_s:.1}s"
                )))
            }
        };

        self.endpoint.connection = Some(connection);
        self.connected = true;
        Ok(())
    }

    /// Open a bidirectional stream, send `payload`, finish, then read
    /// the echoed bytes back. Used by the Phase 4 echo test.
    fn echo<'py>(&self, py: Python<'py>, payload: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
        let runtime = self
            .endpoint
            .runtime
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("connect() must be called first"))?
            .clone();
        let conn = self.endpoint.require_connection()?.clone();
        let payload = payload.to_vec();
        let result = py.allow_threads(|| {
            runtime.block_on(async move { quic::client_echo(&conn, &payload).await })
        });
        let bytes = result.map_err(map_quic_err)?;
        Ok(PyBytes::new_bound(py, &bytes))
    }

    /// Send a request payload to the server and block until a reply is
    /// received. One bidi stream per call.
    fn request<'py>(
        &self,
        py: Python<'py>,
        payload: &[u8],
        timeout_s: f64,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let runtime = self
            .endpoint
            .runtime
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("connect() must be called first"))?
            .clone();
        let conn = self.endpoint.require_connection()?.clone();
        let payload = payload.to_vec();
        let result = py.allow_threads(|| {
            runtime.block_on(async move {
                tokio::time::timeout(
                    Duration::from_secs_f64(timeout_s),
                    quic::client_request(&conn, &payload),
                )
                .await
            })
        });
        match result {
            Ok(Ok(bytes)) => Ok(PyBytes::new_bound(py, &bytes)),
            Ok(Err(e)) => Err(map_quic_err(e)),
            Err(_) => Err(PyTimeoutError::new_err(format!(
                "QUIC request timed out after {timeout_s:.1}s"
            ))),
        }
    }

    fn is_connected(&self) -> bool {
        self.connected
    }

    fn session_id(&self) -> &str {
        &self.session_id
    }

    fn close(&mut self) {
        self.connected = false;
    }

    fn __repr__(&self) -> String {
        format!(
            "QuicClient(session_id={:?}, public={:?}, peer={:?}, connected={})",
            self.session_id,
            self.public_address(),
            self.peer_address(),
            self.connected
        )
    }
}

/// Server-side handle (the Modal GPU container end).
#[pyclass]
struct QuicServer {
    session_id: String,
    endpoint: Endpoint,
    listening: bool,
}

#[pymethods]
impl QuicServer {
    #[new]
    fn new(session_id: String) -> PyResult<Self> {
        Ok(Self {
            session_id,
            endpoint: Endpoint::new()?,
            listening: false,
        })
    }

    fn local_port(&self) -> PyResult<u16> {
        self.endpoint.local_port()
    }

    #[pyo3(signature = (server = None))]
    fn discover_public_address(&mut self, server: Option<&str>) -> PyResult<(String, u16)> {
        self.endpoint.discover(server.unwrap_or(DEFAULT_STUN_SERVER))
    }

    fn public_address(&self) -> Option<(String, u16)> {
        self.endpoint
            .public_addr
            .map(|a| (a.ip().to_string(), a.port()))
    }

    fn set_peer_address(&mut self, ip: &str, port: u16) -> PyResult<()> {
        self.endpoint.set_peer(ip, port)
    }

    fn peer_address(&self) -> Option<(String, u16)> {
        self.endpoint
            .peer_addr
            .map(|a| (a.ip().to_string(), a.port()))
    }

    /// Run the simultaneous-send hole-punch loop against the peer.
    /// Identical to ``QuicClient.punch`` — the protocol is symmetric.
    #[pyo3(signature = (timeout_s = 8.0))]
    fn punch(&self, timeout_s: f64) -> PyResult<(u32, u32, f64)> {
        self.endpoint.punch(Duration::from_secs_f64(timeout_s))
    }

    /// Hand the punched socket to quinn and accept one incoming QUIC
    /// connection from the peer.
    #[pyo3(signature = (timeout_s = 20.0))]
    fn listen(&mut self, timeout_s: f64) -> PyResult<()> {
        let peer = self.endpoint.require_peer()?;
        let runtime = self.endpoint.ensure_runtime()?;
        let socket = self.endpoint.take_socket()?;

        let connection = Python::with_gil(|py| {
            py.allow_threads(|| {
                runtime.block_on(async move {
                    tokio::time::timeout(
                        Duration::from_secs_f64(timeout_s),
                        quic::server_accept(socket, peer),
                    )
                    .await
                })
            })
        });

        let connection = match connection {
            Ok(Ok(c)) => c,
            Ok(Err(e)) => return Err(map_quic_err(e)),
            Err(_) => {
                return Err(PyTimeoutError::new_err(format!(
                    "QUIC accept from {peer} timed out after {timeout_s:.1}s"
                )))
            }
        };

        self.endpoint.connection = Some(connection);
        self.listening = true;
        Ok(())
    }

    /// Accept one bidirectional stream, read all bytes, write them
    /// back. Used by the Phase 4 echo test.
    fn echo_once(&self) -> PyResult<u64> {
        let runtime = self
            .endpoint
            .runtime
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("listen() must be called first"))?
            .clone();
        let conn = self.endpoint.require_connection()?.clone();
        let bytes = Python::with_gil(|py| {
            py.allow_threads(|| {
                runtime.block_on(async move { quic::server_echo_once(&conn).await })
            })
        })
        .map_err(map_quic_err)?;
        Ok(bytes)
    }

    /// Run a request/reply loop. Blocks the calling Python thread.
    /// On each incoming bidi stream, reads request bytes, calls
    /// ``handler(request_bytes) -> reply_bytes``, writes the reply.
    /// Returns when the client closes the connection.
    fn serve_forever(&self, handler: PyObject) -> PyResult<u64> {
        let runtime = self
            .endpoint
            .runtime
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("listen() must be called first"))?
            .clone();
        let conn = self.endpoint.require_connection()?.clone();
        let mut count: u64 = 0u64;
        loop {
            let result = Python::with_gil(|py| -> PyResult<Option<()>> {
                let accept_result = py.allow_threads(|| {
                    runtime.block_on(async { quic::server_accept_request(&conn).await })
                });
                let (request, send) = match accept_result.map_err(map_quic_err)? {
                    Some(pair) => pair,
                    None => return Ok(None),
                };
                let py_bytes = PyBytes::new_bound(py, &request);
                let reply_obj = handler.call1(py, (py_bytes,))?;
                let reply: Vec<u8> = reply_obj.extract::<Vec<u8>>(py)?;
                py.allow_threads(|| {
                    runtime.block_on(async move { quic::server_send_reply(send, &reply).await })
                })
                .map_err(map_quic_err)?;
                Ok(Some(()))
            })?;
            if result.is_none() {
                break;
            }
            count += 1;
        }
        Ok(count)
    }

    fn is_listening(&self) -> bool {
        self.listening
    }

    fn session_id(&self) -> &str {
        &self.session_id
    }

    fn close(&mut self) {
        self.listening = false;
    }

    fn __repr__(&self) -> String {
        format!(
            "QuicServer(session_id={:?}, public={:?}, peer={:?}, listening={})",
            self.session_id,
            self.public_address(),
            self.peer_address(),
            self.listening
        )
    }
}

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn _omx_quic(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<QuicClient>()?;
    m.add_class::<QuicServer>()?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add("DEFAULT_STUN_SERVER", DEFAULT_STUN_SERVER)?;
    m.add("DEFAULT_PUNCH_TIMEOUT_S", DEFAULT_PUNCH_TIMEOUT.as_secs_f64())?;
    Ok(())
}
