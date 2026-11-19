//! Localhost HTTP segmentation client (`127.0.0.1:17890` by default, or
//! `PHOTOCRAFT_AI_SEGMENT_ADDR`) implementing [`photocraft_engine::ai_segment::SegmentationBackend`].

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::Arc;
use std::time::Duration;

use photocraft_engine::ai_segment::{SegmentInput, SegmentationBackend};
use serde_json::{Value, json};

const DEFAULT_ADDR: &str = "127.0.0.1:17890";
const CONNECT_TIMEOUT: Duration = Duration::from_millis(120);
const WRITE_TIMEOUT: Duration = Duration::from_secs(15);
const READ_TIMEOUT: Duration = Duration::from_secs(60);
const MAX_HEADER_BYTES: usize = 8192;

/// Local loopback client talking to `ai_segment_server.py` (IS-Net + SAM 2.1 Tiny).
#[derive(Clone, Debug)]
pub struct LocalSegmentationClient {
    addr: SocketAddr,
}

impl LocalSegmentationClient {
    /// Creates a client targeting `addr`.
    pub fn new(addr: SocketAddr) -> Self {
        Self { addr }
    }

    fn request(&self, meta: &Value, image: SegmentInput<'_>) -> Option<Vec<u8>> {
        let expected = (image.width as usize).checked_mul(image.height as usize)?;
        if expected == 0 || image.rgb.len() != expected.checked_mul(3)? {
            return None;
        }
        let meta_str = serde_json::to_string(meta).ok()?;
        let mut stream = TcpStream::connect_timeout(&self.addr, CONNECT_TIMEOUT).ok()?;
        let _ = stream.set_nodelay(true);
        let _ = stream.set_write_timeout(Some(WRITE_TIMEOUT));
        let _ = stream.set_read_timeout(Some(READ_TIMEOUT));

        let header = format!(
            "POST /segment HTTP/1.1\r\nHost: {}\r\nConnection: close\r\nContent-Type: application/octet-stream\r\nX-Photocraft-Meta: {}\r\nContent-Length: {}\r\n\r\n",
            self.addr,
            meta_str,
            image.rgb.len()
        );
        stream.write_all(header.as_bytes()).ok()?;
        stream.write_all(image.rgb).ok()?;
        stream.flush().ok()?;

        let cap = expected.checked_add(MAX_HEADER_BYTES)?;
        let mut raw = Vec::new();
        stream.take(cap as u64).read_to_end(&mut raw).ok()?;
        parse_http_mask(&raw, expected)
    }
}

fn parse_http_mask(raw: &[u8], expected: usize) -> Option<Vec<u8>> {
    let ok_status = raw.starts_with(b"HTTP/1.1 200") || raw.starts_with(b"HTTP/1.0 200");
    if !ok_status {
        return None;
    }
    let split = raw.windows(4).position(|w| w == b"\r\n\r\n")?;
    let body = raw.get(split + 4..)?;
    (body.len() == expected).then(|| body.to_vec())
}

impl SegmentationBackend for LocalSegmentationClient {
    fn segment_subject(&self, image: SegmentInput<'_>) -> Option<Vec<u8>> {
        let meta = json!({"op": "subject", "width": image.width, "height": image.height});
        self.request(&meta, image)
    }

    fn segment_box(&self, image: SegmentInput<'_>, box_xyxy: [f32; 4]) -> Option<Vec<u8>> {
        let meta = json!({"op": "box", "width": image.width, "height": image.height, "box": box_xyxy});
        self.request(&meta, image)
    }

    fn segment_point(&self, image: SegmentInput<'_>, point_xy: [f32; 2]) -> Option<Vec<u8>> {
        let meta = json!({"op": "point", "width": image.width, "height": image.height, "point": point_xy});
        self.request(&meta, image)
    }
}

/// Resolves the configured loopback AI segmentation provider (`None` when disabled with
/// `PHOTOCRAFT_AI_SEGMENT_ADDR=off`).
pub fn default_backend() -> Option<Arc<dyn SegmentationBackend>> {
    let raw = std::env::var("PHOTOCRAFT_AI_SEGMENT_ADDR").unwrap_or_else(|_| DEFAULT_ADDR.to_string());
    let trimmed = raw.trim();
    if trimmed.is_empty() || matches!(trimmed.to_ascii_lowercase().as_str(), "off" | "0" | "false" | "none") {
        return None;
    }
    let addr: SocketAddr = trimmed.parse().ok()?;
    if !addr.ip().is_loopback() {
        return None;
    }
    Some(Arc::new(LocalSegmentationClient::new(addr)))
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use photocraft_engine::ai_segment::{SegmentInput, SegmentationBackend};
    use super::LocalSegmentationClient;

    #[test]
    fn round_trips_binary_mask_over_loopback_http() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            let mut buf = [0u8; 4096];
            let _ = sock.read(&mut buf).unwrap();
            let mask = vec![200u8; 16];
            let hdr = format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n", mask.len());
            sock.write_all(hdr.as_bytes()).unwrap();
            sock.write_all(&mask).unwrap();
        });

        let client = LocalSegmentationClient::new(addr);
        let rgb = vec![128u8; 4 * 4 * 3];
        let out = client.segment_subject(SegmentInput { width: 4, height: 4, rgb: &rgb }).unwrap();
        assert_eq!(out, vec![200u8; 16]);
        server.join().unwrap();
    }
}
