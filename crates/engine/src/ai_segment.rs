//! Pluggable AI segmentation seam for `Select › Subject`, `Object Selection` and
//! `Remove Background` (#2788).
//!
//! `photocraft-engine` stays free of network and filesystem access: native hosts (`photocraft`,
//! `photocraft-cli`, `photocraft-automation`) inject a [`SegmentationBackend`] on [`crate::Session`],
//! while web and offline sessions leave it `None` and run the classical algorithms unchanged.

use photocraft_algo::segment::Sampler;
use photocraft_algo::selection::Region;
use photocraft_geom::Rect;

/// Maximum pixel count sent to an external segmentation backend (`~64 MP`).
const MAX_MODEL_PIXELS: usize = 67_108_864;

/// Row-major RGB8 image buffer passed to a [`SegmentationBackend`].
#[derive(Clone, Copy, Debug)]
pub struct SegmentInput<'a> {
    pub width: u32,
    pub height: u32,
    /// Interleaved RGB8 samples (`width * height * 3` bytes).
    pub rgb: &'a [u8],
}

/// Pluggable neural segmentation provider for smart-selection commands.
pub trait SegmentationBackend: Send + Sync {
    /// Foreground alpha mask (`width * height` bytes) for `Select › Subject` and `Remove Background`.
    fn segment_subject(&self, image: SegmentInput<'_>) -> Option<Vec<u8>>;

    /// Prompted object mask (`width * height` bytes) inside `[x0, y0, x1, y1]` (image-local pixels).
    fn segment_box(&self, image: SegmentInput<'_>, box_xyxy: [f32; 4]) -> Option<Vec<u8>>;

    /// Prompted object mask (`width * height` bytes) at `[x, y]` (image-local pixels).
    fn segment_point(&self, image: SegmentInput<'_>, point_xy: [f32; 2]) -> Option<Vec<u8>>;
}

/// Requested execution mode for smart-selection commands (`"backend": "auto|classical|model"`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendChoice {
    Auto,
    Classical,
    Model,
}

impl BackendChoice {
    /// Parses `"auto"` (default), `"classical"`, or `"model"`.
    pub fn parse(raw: Option<&str>) -> Option<Self> {
        match raw.unwrap_or("auto") {
            "auto" => Some(Self::Auto),
            "classical" | "classic" => Some(Self::Classical),
            "model" => Some(Self::Model),
            _ => None,
        }
    }
}

/// Samples `area` into straight RGB8 over mid-gray (matching `segment::RgbImage::from_rgba`),
/// returning `(rgb, alpha_present)` where `alpha_present[i]` is `false` for fully transparent pixels.
fn sample_rgb8(smp: &dyn Sampler, area: Rect) -> Option<(Vec<u8>, Vec<bool>)> {
    if area.is_empty() {
        return None;
    }
    let (w, h) = (area.width() as usize, area.height() as usize);
    let count = w.checked_mul(h)?;
    if count == 0 || count > MAX_MODEL_PIXELS {
        return None;
    }
    let rgba = smp.rgba(area);
    if rgba.len() != count {
        return None;
    }
    let mut rgb = Vec::with_capacity(count.checked_mul(3)?);
    let mut opaque = Vec::with_capacity(count);
    for px in &rgba {
        let a = px[3].clamp(0.0, 1.0);
        opaque.push(a > 0.002);
        for &c in &px[..3] {
            let v = c.clamp(0.0, 1.0) * a + 0.5 * (1.0 - a);
            rgb.push((v * 255.0).round() as u8);
        }
    }
    Some((rgb, opaque))
}

/// Trims a `w × h` mask in `area` down to its non-zero bounding box (`Region` invariant).
fn trim_mask_region(area: Rect, mask: &[u8], opaque: &[bool]) -> Option<Region> {
    let (w, h) = (area.width() as usize, area.height() as usize);
    if w == 0 || h == 0 || mask.len() != w.saturating_mul(h) || opaque.len() != mask.len() {
        return None;
    }
    let gated: Vec<u8> = mask.iter().zip(opaque).map(|(&m, &ok)| if ok { m } else { 0 }).collect();
    let rows: Vec<&[u8]> = gated.chunks_exact(w).collect();
    let used_rows: Vec<usize> = rows.iter().enumerate().filter(|(_, r)| r.iter().any(|&v| v > 0)).map(|(y, _)| y).collect();
    let used_cols: Vec<usize> = (0..w).filter(|&x| rows.iter().any(|r| r.get(x).copied().unwrap_or(0) > 0)).collect();
    let (&y0, &y1, &x0, &x1) = (used_rows.first()?, used_rows.last()?, used_cols.first()?, used_cols.last()?);
    let trimmed_px: Vec<u8> = rows.get(y0..=y1)?.iter().filter_map(|r| r.get(x0..=x1)).flat_map(|s| s.iter().copied()).collect();
    let off = |n: usize| i32::try_from(n).unwrap_or(i32::MAX);
    let bbox = Rect::new(
        area.x0.saturating_add(off(x0)),
        area.y0.saturating_add(off(y0)),
        area.x0.saturating_add(off(x1.saturating_add(1))),
        area.y0.saturating_add(off(y1.saturating_add(1))),
    );
    (!bbox.is_empty()).then_some(Region { bbox, mask: trimmed_px })
}

/// Runs `Select › Subject` through `backend`, returning `None` if unavailable or empty.
pub fn try_subject(backend: &dyn SegmentationBackend, smp: &dyn Sampler, area: Rect) -> Option<Region> {
    let (rgb, opaque) = sample_rgb8(smp, area)?;
    let inp = SegmentInput { width: area.width(), height: area.height(), rgb: &rgb };
    let mask = backend.segment_subject(inp)?;
    trim_mask_region(area, &mask, &opaque)
}

/// Runs box-prompted `Object Selection` through `backend`.
pub fn try_box(backend: &dyn SegmentationBackend, smp: &dyn Sampler, canvas: Rect, rect: Rect) -> Option<Region> {
    let clipped = rect.intersect(&canvas);
    if clipped.is_empty() {
        return None;
    }
    let (rgb, opaque) = sample_rgb8(smp, canvas)?;
    let inp = SegmentInput { width: canvas.width(), height: canvas.height(), rgb: &rgb };
    let bx = [
        (clipped.x0 - canvas.x0) as f32,
        (clipped.y0 - canvas.y0) as f32,
        (clipped.x1 - canvas.x0) as f32,
        (clipped.y1 - canvas.y0) as f32,
    ];
    let mask = backend.segment_box(inp, bx)?;
    trim_mask_region(canvas, &mask, &opaque)
}

/// Runs point-prompted `Object Selection` through `backend`.
pub fn try_point(backend: &dyn SegmentationBackend, smp: &dyn Sampler, canvas: Rect, pt: (f32, f32)) -> Option<Region> {
    let (rgb, opaque) = sample_rgb8(smp, canvas)?;
    let inp = SegmentInput { width: canvas.width(), height: canvas.height(), rgb: &rgb };
    let local = [pt.0 - canvas.x0 as f32, pt.1 - canvas.y0 as f32];
    let mask = backend.segment_point(inp, local)?;
    trim_mask_region(canvas, &mask, &opaque)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use serde_json::json;
    use super::{BackendChoice, SegmentInput, SegmentationBackend};
    use crate::Session;

    struct MockSeg {
        available: bool,
    }

    impl SegmentationBackend for MockSeg {
        fn segment_subject(&self, image: SegmentInput<'_>) -> Option<Vec<u8>> {
            if !self.available {
                return None;
            }
            let (w, h) = (image.width as usize, image.height as usize);
            let mut out = vec![0u8; w * h];
            for y in h / 4..(3 * h / 4) {
                for x in w / 4..(3 * w / 4) {
                    out[y * w + x] = 255;
                }
            }
            Some(out)
        }

        fn segment_box(&self, image: SegmentInput<'_>, bx: [f32; 4]) -> Option<Vec<u8>> {
            if !self.available {
                return None;
            }
            let (w, h) = (image.width as usize, image.height as usize);
            let mut out = vec![0u8; w * h];
            let (x0, y0) = ((bx[0].max(0.0) as usize).min(w), (bx[1].max(0.0) as usize).min(h));
            let (x1, y1) = ((bx[2].max(0.0) as usize).min(w), (bx[3].max(0.0) as usize).min(h));
            for y in y0..y1 {
                for x in x0..x1 {
                    out[y * w + x] = 255;
                }
            }
            Some(out)
        }

        fn segment_point(&self, image: SegmentInput<'_>, pt: [f32; 2]) -> Option<Vec<u8>> {
            self.segment_box(image, [pt[0] - 4.0, pt[1] - 4.0, pt[0] + 4.0, pt[1] + 4.0])
        }
    }

    #[test]
    fn backend_choice_parses_known_modes() {
        assert_eq!(BackendChoice::parse(None), Some(BackendChoice::Auto));
        assert_eq!(BackendChoice::parse(Some("auto")), Some(BackendChoice::Auto));
        assert_eq!(BackendChoice::parse(Some("classical")), Some(BackendChoice::Classical));
        assert_eq!(BackendChoice::parse(Some("model")), Some(BackendChoice::Model));
        assert_eq!(BackendChoice::parse(Some("bogus")), None);
    }

    #[test]
    fn select_subject_and_object_use_injected_model_and_report_backend() {
        let mut s = Session::new();
        s.segmentation = Some(Arc::new(MockSeg { available: true }));
        s.execute("file.new", json!({"width": 80, "height": 60})).unwrap();

        let sub = s.execute("select.subject", json!({})).unwrap();
        assert_eq!(sub["selected"], true);
        assert_eq!(sub["backend"], "model");
        assert_eq!(sub["bounds"], json!([20, 15, 40, 30]));

        let obj = s.execute("select.object", json!({"rect": [10, 12, 20, 16], "mode": "add"})).unwrap();
        assert_eq!(obj["selected"], true);
        assert_eq!(obj["backend"], "model");
    }

    #[test]
    fn explicit_model_backend_errors_when_unavailable_while_auto_falls_back() {
        let mut s = Session::new();
        s.segmentation = Some(Arc::new(MockSeg { available: false }));
        s.execute("file.new", json!({"width": 64, "height": 64})).unwrap();

        assert!(s.execute("select.subject", json!({"backend": "model"})).is_err());
        assert!(s.execute("select.subject", json!({"backend": "auto"})).is_ok());
    }
}
