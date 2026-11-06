//! Inline image and video-frame previews over the kitty graphics protocol.
//!
//! `ffmpeg` converts the selected file to a bounded PNG on a background thread, and the event loop
//! writes that PNG straight to the terminal on top of the content pane, right after ratatui's
//! frame. ratatui never knows about it: its text placeholder simply sits under the image. A
//! missing `ffmpeg` or an undecodable file leaves that placeholder visible.
//!
//! Videos and GIFs show their first frame and play on request, with sound through `ffplay`.
//! `ffmpeg` streams PNG frames in real time and the event loop draws them, because only that loop
//! may write to the terminal: a second writer could split ratatui's escape sequences. Each frame
//! goes on top of the last under a new image id, and the older one is erased afterwards, so the
//! pane is never left empty between frames.

use ratatui::layout::Rect;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::time::SystemTime;

/// Playing frames cycle through these ids. Index 0 also holds still images.
const IMAGE_IDS: [u32; 3] = [4242, 4243, 4244];
const MAX_EDGE_PX: u32 = 1600;
const CHUNK_BYTES: usize = 4096;
const PNG_SIGNATURE: &[u8] = b"\x89PNG\r\n\x1a\n";
const EXTENSIONS: &[&str] = &[
    "png", "jpg", "jpeg", "webp", "gif", "bmp", "tif", "tiff", "heic", "avif", "mp4", "mov", "m4v",
    "webm", "mkv",
];
const ANIMATED_EXTENSIONS: &[&str] = &["gif", "mp4", "mov", "m4v", "webm", "mkv"];
/// The largest stream herdr kept up with: 540x960 at 15 fps. 720x1280 at 24 fps fell behind.
const PLAY_MAX_EDGE_PX: u32 = 960;
const PLAY_FPS: u32 = 15;
/// Used when the terminal does not report its pixel size.
const DEFAULT_CELL_PX: (u32, u32) = (10, 20);
/// Homebrew's bin directory is missing from `PATH` when herdr is launched outside a shell.
const FFMPEG_PATHS: &[&str] = &[
    "ffmpeg",
    "/opt/homebrew/bin/ffmpeg",
    "/usr/local/bin/ffmpeg",
];
const FFPLAY_PATHS: &[&str] = &[
    "ffplay",
    "/opt/homebrew/bin/ffplay",
    "/usr/local/bin/ffplay",
];

/// Erases every image and frees its data in the terminal.
pub const CLEAR_SEQUENCE: &[u8] =
    b"\x1b_Ga=d,d=I,i=4242,q=2\x1b\\\x1b_Ga=d,d=I,i=4243,q=2\x1b\\\x1b_Ga=d,d=I,i=4244,q=2\x1b\\";

fn has_extension(path: &Path, list: &[&str]) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| list.iter().any(|x| e.eq_ignore_ascii_case(x)))
}

/// Whether the file's extension is one the preview handles.
pub fn is_previewable(path: &Path) -> bool {
    has_extension(path, EXTENSIONS)
}

/// Read one PNG from a stream of concatenated PNGs, or `None` at the end of the stream.
fn read_png(stream: &mut impl Read) -> Option<Vec<u8>> {
    let mut png = vec![0; PNG_SIGNATURE.len()];
    stream.read_exact(&mut png).ok()?;
    if png != PNG_SIGNATURE {
        return None;
    }
    loop {
        let start = png.len();
        png.resize(start + 8, 0);
        stream.read_exact(&mut png[start..]).ok()?;
        let length = u32::from_be_bytes(png[start..start + 4].try_into().ok()?) as usize;
        let is_end = &png[start + 4..start + 8] == b"IEND";
        let body = png.len();
        png.resize(body + length + 4, 0); // chunk data plus its CRC
        stream.read_exact(&mut png[body..]).ok()?;
        if is_end {
            return Some(png);
        }
    }
}

/// A video playing once. Dropping it stops the picture and the sound.
struct Playback {
    children: Vec<Child>,
    frames: Receiver<Vec<u8>>,
}

impl Drop for Playback {
    fn drop(&mut self) {
        for child in &mut self.children {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn start_playback(path: &Path) -> Option<Playback> {
    let filter = format!(
        "fps={PLAY_FPS},scale=w='min({PLAY_MAX_EDGE_PX},iw)':h='min({PLAY_MAX_EDGE_PX},ih)':force_original_aspect_ratio=decrease"
    );
    let mut child = FFMPEG_PATHS.iter().find_map(|ffmpeg| {
        let mut cmd = Command::new(ffmpeg);
        cmd.args(["-nostdin", "-v", "error", "-protocol_whitelist", "file"])
            .args(["-re", "-i"])
            .arg(path)
            .args(["-an", "-vf", &filter])
            .args([
                "-f",
                "image2pipe",
                "-c:v",
                "png",
                "-compression_level",
                "1",
                "-",
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        crate::proc::in_launch_dir(&mut cmd).spawn().ok()
    })?;
    let mut stdout = child.stdout.take()?;
    let (sender, frames) = mpsc::sync_channel(1);
    std::thread::spawn(move || {
        while let Some(frame) = read_png(&mut stdout) {
            if sender.send(frame).is_err() {
                return;
            }
        }
    });
    let mut children = vec![child];
    children.extend(start_sound(path)); // a video with no sound still plays
    Some(Playback { children, frames })
}

fn start_sound(path: &Path) -> Option<Child> {
    FFPLAY_PATHS.iter().find_map(|ffplay| {
        let mut cmd = Command::new(ffplay);
        cmd.args([
            "-nodisp",
            "-vn",
            "-autoexit",
            "-nostats",
            "-loglevel",
            "quiet",
        ])
        .arg(path)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
        crate::proc::in_launch_dir(&mut cmd).spawn().ok()
    })
}

/// The pixel size stored in a PNG header.
pub fn png_size(png: &[u8]) -> Option<(u32, u32)> {
    if !png.starts_with(PNG_SIGNATURE) || png.len() < 24 {
        return None;
    }
    let read = |at: usize| u32::from_be_bytes([png[at], png[at + 1], png[at + 2], png[at + 3]]);
    Some((read(16), read(20))).filter(|&(w, h)| w > 0 && h > 0)
}

/// One side of the placement. The terminal derives the other side from the image's aspect ratio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fit {
    Cols(u32),
    Rows(u32),
}

/// Fit an image inside `area`, never enlarging it past its own pixel size.
pub fn fit(image: (u32, u32), area: (u16, u16), cell: (u32, u32)) -> Fit {
    let (img_w, img_h) = (u64::from(image.0), u64::from(image.1));
    let (cols, rows) = (u64::from(area.0), u64::from(area.1));
    let (cell_w, cell_h) = (u64::from(cell.0), u64::from(cell.1));
    if rows * cell_h * img_w <= cols * cell_w * img_h {
        Fit::Rows((rows.min((img_h / cell_h).max(1))) as u32)
    } else {
        Fit::Cols((cols.min((img_w / cell_w).max(1))) as u32)
    }
}

/// The bytes that draw `png` with its top-left corner at `rect`. The cursor is left where it was.
pub fn place_sequence(png: &[u8], rect: Rect, fit: Fit, id: u32, z: u32) -> Vec<u8> {
    let size = match fit {
        Fit::Cols(n) => format!("c={n}"),
        Fit::Rows(n) => format!("r={n}"),
    };
    let encoded = encode_base64(png);
    let mut out = format!("\x1b7\x1b[{};{}H", rect.y + 1, rect.x + 1).into_bytes();
    let mut chunks = encoded.chunks(CHUNK_BYTES).peekable();
    let mut head = format!("a=T,f=100,i={id},z={z},q=2,C=1,{size},");
    while let Some(chunk) = chunks.next() {
        let more = u8::from(chunks.peek().is_some());
        out.extend_from_slice(format!("\x1b_G{head}m={more};").as_bytes());
        out.extend_from_slice(chunk);
        out.extend_from_slice(b"\x1b\\");
        head.clear();
    }
    out.extend_from_slice(b"\x1b8");
    out
}

fn encode_base64(data: &[u8]) -> Vec<u8> {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = Vec::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let at = |i: usize| chunk.get(i).copied().unwrap_or(0);
        let bits = u32::from(at(0)) << 16 | u32::from(at(1)) << 8 | u32::from(at(2));
        for i in 0..4 {
            let index = (bits >> (18 - 6 * i)) as usize & 63;
            out.push(if i <= chunk.len() { TABLE[index] } else { b'=' });
        }
    }
    out
}

/// Convert a file to a bounded PNG. Symlinks are refused, as the content pane does for targets
/// outside the viewed root.
fn convert(path: &Path) -> Option<Vec<u8>> {
    if !std::fs::symlink_metadata(path).ok()?.is_file() {
        return None;
    }
    let scale = format!(
        "scale=w='min({MAX_EDGE_PX},iw)':h='min({MAX_EDGE_PX},ih)':force_original_aspect_ratio=decrease"
    );
    let png = FFMPEG_PATHS.iter().find_map(|ffmpeg| {
        let mut cmd = Command::new(ffmpeg);
        cmd.args([
            "-nostdin",
            "-v",
            "error",
            "-protocol_whitelist",
            "file",
            "-i",
        ])
        .arg(path)
        .args(["-frames:v", "1", "-vf", &scale])
        .args(["-f", "image2pipe", "-c:v", "png", "-"])
        .stdin(Stdio::null())
        .stderr(Stdio::null());
        crate::proc::in_launch_dir(&mut cmd).output().ok()
    })?;
    (png.status.success() && png_size(&png.stdout).is_some()).then_some(png.stdout)
}

/// A file version. Overwriting the file changes it, so the preview reloads.
type Source = (PathBuf, Option<SystemTime>);

fn source_of(path: PathBuf) -> Source {
    let modified = std::fs::metadata(&path).and_then(|m| m.modified()).ok();
    (path, modified)
}

/// Where an image should be shown.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Target {
    source: Source,
    rect: Rect,
}

/// The newest conversion: the PNG, or `None` when the file could not be converted.
type Converted = (Source, Option<Vec<u8>>);

pub struct ImagePreview {
    jobs: Sender<Source>,
    results: Receiver<Converted>,
    wanted: Option<Target>,
    placed: Option<Target>,
    requested: Option<Source>,
    converted: Option<Converted>,
    player: fn(&Path) -> Option<Playback>,
    playback: Option<Playback>,
    /// How many frames have been drawn since the last clear, which picks each frame's id and layer.
    frames_drawn: u32,
}

impl ImagePreview {
    pub fn new() -> Self {
        Self::with_converter(convert)
    }

    /// Whether frames are arriving, so the event loop should tick fast enough to show them.
    pub fn playing(&self) -> bool {
        self.playback.is_some()
    }

    /// Play the shown video from the start, or stop it on its current frame.
    pub fn toggle_play(&mut self) {
        if self.playback.take().is_some() {
            return;
        }
        if let Some(target) = &self.placed
            && has_extension(&target.source.0, ANIMATED_EXTENSIONS)
        {
            self.playback = (self.player)(&target.source.0);
        }
    }

    /// One worker that skips to the newest job, so holding an arrow key never queues conversions.
    fn with_converter(converter: impl Fn(&Path) -> Option<Vec<u8>> + Send + 'static) -> Self {
        let (jobs, inbox) = mpsc::channel::<Source>();
        let (outbox, results) = mpsc::channel();
        std::thread::spawn(move || {
            while let Ok(job) = inbox.recv() {
                let source = inbox.try_iter().last().unwrap_or(job);
                let png = converter(&source.0);
                if outbox.send((source, png)).is_err() {
                    return;
                }
            }
        });
        Self {
            jobs,
            results,
            wanted: None,
            placed: None,
            requested: None,
            converted: None,
            player: start_playback,
            playback: None,
            frames_drawn: 0,
        }
    }

    /// Choose what to show. `None` removes the image.
    pub fn set_target(&mut self, target: Option<(PathBuf, Rect)>) {
        self.wanted = target.map(|(path, rect)| Target {
            source: source_of(path),
            rect,
        });
    }

    /// Forget the placement after the screen was wiped, so the next [`sync`](Self::sync) redraws.
    pub fn invalidate(&mut self) {
        self.placed = None;
    }

    /// Advance towards the wanted state, writing to the terminal only when something changes.
    pub fn sync(&mut self, out: &mut impl Write, cell: (u32, u32)) -> io::Result<()> {
        self.converted = self.results.try_iter().last().or(self.converted.take());
        if self.wanted == self.placed {
            return self.play(out, cell);
        }
        self.playback = None;
        let Some(wanted) = self.wanted.clone() else {
            self.placed = None;
            out.write_all(CLEAR_SEQUENCE)?;
            return out.flush();
        };
        let Some(converted) = self.converted.as_ref().filter(|c| c.0 == wanted.source) else {
            if self.requested.as_ref() != Some(&wanted.source) {
                self.requested = Some(wanted.source.clone());
                let _ = self.jobs.send(wanted.source);
            }
            return Ok(()); // the previous image stays until this one is ready
        };
        match converted
            .1
            .as_deref()
            .zip(converted.1.as_deref().and_then(png_size))
        {
            Some((png, size)) => {
                let area = (wanted.rect.width, wanted.rect.height);
                out.write_all(CLEAR_SEQUENCE)?;
                let placement = fit(size, area, cell);
                out.write_all(&place_sequence(
                    png,
                    wanted.rect,
                    placement,
                    IMAGE_IDS[0],
                    0,
                ))?;
                self.frames_drawn = 0;
                self.placed = Some(wanted);
            }
            None => {
                if self.placed.take().is_some() {
                    out.write_all(CLEAR_SEQUENCE)?;
                }
            }
        }
        out.flush()
    }

    /// Draw the newest frame on top of the previous one, then erase the frame before that.
    fn play(&mut self, out: &mut impl Write, cell: (u32, u32)) -> io::Result<()> {
        let Some(playback) = &self.playback else {
            return Ok(());
        };
        let mut newest = None;
        let ended = loop {
            match playback.frames.try_recv() {
                Ok(frame) => newest = Some(frame),
                Err(TryRecvError::Empty) => break false,
                Err(TryRecvError::Disconnected) => break true,
            }
        };
        if ended {
            self.playback = None;
        }
        let Some(target) = &self.placed else {
            return Ok(());
        };
        let Some((frame, size)) = newest.and_then(|f| png_size(&f).map(|size| (f, size))) else {
            return Ok(());
        };
        self.frames_drawn += 1;
        let id_at = |n: u32| IMAGE_IDS[n as usize % IMAGE_IDS.len()];
        let placement = fit(size, (target.rect.width, target.rect.height), cell);
        let (id, oldest) = (id_at(self.frames_drawn), id_at(self.frames_drawn + 1));
        out.write_all(&place_sequence(
            &frame,
            target.rect,
            placement,
            id,
            self.frames_drawn,
        ))?;
        write!(out, "\x1b_Ga=d,d=I,i={oldest},q=2\x1b\\")?;
        out.flush()
    }
}

impl Default for ImagePreview {
    fn default() -> Self {
        Self::new()
    }
}

/// The terminal's cell size in pixels, or a typical size when it does not report one.
pub fn cell_pixels() -> (u32, u32) {
    match crossterm::terminal::window_size() {
        Ok(s) if s.width > 0 && s.height > 0 && s.columns > 0 && s.rows > 0 => (
            (u32::from(s.width) / u32::from(s.columns)).max(1),
            (u32::from(s.height) / u32::from(s.rows)).max(1),
        ),
        _ => DEFAULT_CELL_PX,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn png(w: u32, h: u32) -> Vec<u8> {
        let mut bytes = PNG_SIGNATURE.to_vec();
        bytes.extend_from_slice(&[0, 0, 0, 13]);
        bytes.extend_from_slice(b"IHDR");
        bytes.extend_from_slice(&w.to_be_bytes());
        bytes.extend_from_slice(&h.to_be_bytes());
        bytes
    }

    fn area() -> Rect {
        Rect::new(30, 2, 80, 40)
    }

    /// Sync until `done` holds, because conversion happens on the worker.
    fn sync_until(view: &mut ImagePreview, out: &mut Vec<u8>, done: impl Fn(&[u8]) -> bool) {
        let start = Instant::now();
        while !done(out) {
            assert!(
                start.elapsed() < Duration::from_secs(5),
                "preview never settled"
            );
            view.sync(out, (10, 20)).unwrap();
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    fn drawn(out: &[u8]) -> bool {
        out.windows(4).any(|w| w == b"a=T,")
    }

    /// Two frames are ready. The sender stays alive so the video keeps playing.
    fn fake_player(path: &Path) -> Option<Playback> {
        if path == Path::new("broken.mp4") {
            return None;
        }
        let (sender, frames) = mpsc::sync_channel(2);
        sender.send(png(320, 180)).unwrap();
        sender.send(png(320, 180)).unwrap();
        std::mem::forget(sender);
        Some(Playback {
            children: Vec::new(),
            frames,
        })
    }

    /// One frame, then the end of the video.
    fn short_player(_: &Path) -> Option<Playback> {
        let (sender, frames) = mpsc::sync_channel(1);
        sender.send(png(320, 180)).unwrap();
        Some(Playback {
            children: Vec::new(),
            frames,
        })
    }

    fn video_preview() -> ImagePreview {
        let mut view = ImagePreview::with_converter(|_| Some(png(800, 400)));
        view.player = fake_player;
        view
    }

    #[test]
    fn splits_a_stream_of_concatenated_pngs() {
        let chunk = |name: &[u8; 4], data: &[u8]| {
            let mut bytes = (data.len() as u32).to_be_bytes().to_vec();
            bytes.extend_from_slice(name);
            bytes.extend_from_slice(data);
            bytes.extend_from_slice(&[0; 4]);
            bytes
        };
        let mut frame = PNG_SIGNATURE.to_vec();
        frame.extend(chunk(b"IHDR", &[1; 13]));
        frame.extend(chunk(b"IEND", &[]));

        let mut stream = frame.repeat(2);
        stream.extend_from_slice(&frame[..10]); // a frame cut off mid-way
        let mut reader = stream.as_slice();
        assert_eq!(read_png(&mut reader), Some(frame.clone()));
        assert_eq!(read_png(&mut reader), Some(frame));
        assert_eq!(read_png(&mut reader), None);
    }

    #[test]
    fn a_video_waits_for_the_key_then_stops_when_left() {
        let mut view = video_preview();
        let mut out = Vec::new();
        view.set_target(Some(("v.mp4".into(), area())));
        sync_until(&mut view, &mut out, drawn);
        view.sync(&mut out, (10, 20)).unwrap();
        assert!(!view.playing(), "selecting a video does not play it");

        view.toggle_play();
        assert!(view.playing());
        view.set_target(None);
        view.sync(&mut out, (10, 20)).unwrap();
        assert!(!view.playing());
    }

    #[test]
    fn a_new_frame_goes_on_top_before_the_oldest_is_erased() {
        let mut view = video_preview();
        let mut out = Vec::new();
        view.set_target(Some(("v.mp4".into(), area())));
        sync_until(&mut view, &mut out, drawn);
        view.toggle_play();

        out.clear();
        view.sync(&mut out, (10, 20)).unwrap();
        let text = String::from_utf8_lossy(&out);
        let placed = text
            .find("i=4243,z=1,")
            .expect("the frame uses the next id and layer");
        let erased = text
            .find("a=d,d=I,i=4244")
            .expect("the id two frames back is erased");
        assert!(
            placed < erased,
            "the old frame stays until the new one is drawn"
        );
        assert_eq!(
            text.matches("a=d").count(),
            1,
            "the previous frame stays underneath"
        );
    }

    #[test]
    fn the_key_stops_a_playing_video_and_starts_it_again() {
        let mut view = video_preview();
        let mut out = Vec::new();
        view.set_target(Some(("v.mp4".into(), area())));
        sync_until(&mut view, &mut out, drawn);

        view.toggle_play();
        view.toggle_play();
        assert!(!view.playing());
        view.toggle_play();
        assert!(view.playing());
    }

    #[test]
    fn playback_ends_with_the_video() {
        let mut view = video_preview();
        view.player = short_player;
        let mut out = Vec::new();
        view.set_target(Some(("v.mp4".into(), area())));
        sync_until(&mut view, &mut out, drawn);

        view.toggle_play();
        view.sync(&mut out, (10, 20)).unwrap();
        assert!(!view.playing());
    }

    #[test]
    fn the_key_does_nothing_on_a_still_image() {
        let mut view = video_preview();
        let mut out = Vec::new();
        view.set_target(Some(("a.png".into(), area())));
        sync_until(&mut view, &mut out, drawn);
        view.toggle_play();
        assert!(!view.playing());
    }

    #[test]
    fn a_failed_start_leaves_the_first_frame_in_place() {
        let mut view = video_preview();
        let mut out = Vec::new();
        view.set_target(Some(("broken.mp4".into(), area())));
        sync_until(&mut view, &mut out, drawn);

        view.toggle_play();
        assert!(!view.playing());
        out.clear();
        view.sync(&mut out, (10, 20)).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn recognizes_images_and_videos_by_extension_in_any_case() {
        assert!(is_previewable(Path::new("a/b.PNG")));
        assert!(is_previewable(Path::new("a/b.mp4")));
        assert!(!is_previewable(Path::new("a/b.txt")));
        assert!(!is_previewable(Path::new("a/png")));
    }

    #[test]
    fn reads_the_png_header_and_rejects_anything_else() {
        assert_eq!(png_size(&png(640, 360)), Some((640, 360)));
        assert_eq!(png_size(b"not a png"), None);
        assert_eq!(png_size(&png(0, 10)), None);
    }

    #[test]
    fn a_wide_image_fills_the_width_and_a_tall_one_fills_the_height() {
        assert_eq!(fit((1600, 900), (80, 40), (10, 20)), Fit::Cols(80));
        assert_eq!(fit((900, 1600), (80, 40), (10, 20)), Fit::Rows(40));
    }

    #[test]
    fn a_small_image_is_not_enlarged() {
        assert_eq!(fit((32, 16), (80, 40), (10, 20)), Fit::Cols(3));
        assert_eq!(fit((16, 40), (80, 40), (10, 20)), Fit::Rows(2));
    }

    #[test]
    fn base64_matches_known_values_including_padding() {
        assert_eq!(encode_base64(b"M"), b"TQ==");
        assert_eq!(encode_base64(b"Ma"), b"TWE=");
        assert_eq!(encode_base64(b"Man"), b"TWFu");
        assert_eq!(encode_base64(b""), b"");
    }

    #[test]
    fn a_large_image_is_sent_in_chunks_and_restores_the_cursor() {
        let text = String::from_utf8(place_sequence(
            &vec![7; CHUNK_BYTES * 2],
            Rect::new(4, 2, 10, 5),
            Fit::Rows(5),
            4242,
            0,
        ))
        .unwrap();
        assert!(text.starts_with("\x1b7\x1b[3;5H\x1b_Ga=T,f=100,i=4242,z=0,q=2,C=1,r=5,m=1;"));
        assert!(text.contains("\x1b\\\x1b_Gm=1;"));
        assert!(text.ends_with("\x1b\\\x1b8"));
        assert!(text.contains("\x1b_Gm=0;"));
    }

    #[test]
    fn draws_once_then_stays_quiet_until_cleared() {
        let mut view = ImagePreview::with_converter(|_| Some(png(800, 400)));
        let mut out = Vec::new();
        view.set_target(Some(("a.png".into(), area())));
        sync_until(&mut view, &mut out, drawn);

        out.clear();
        view.sync(&mut out, (10, 20)).unwrap();
        assert!(out.is_empty(), "an unchanged target writes nothing");

        view.set_target(None);
        view.sync(&mut out, (10, 20)).unwrap();
        assert_eq!(out, CLEAR_SEQUENCE);
    }

    #[test]
    fn invalidate_redraws_without_converting_again() {
        let mut view = ImagePreview::with_converter(|_| Some(png(800, 400)));
        let mut out = Vec::new();
        view.set_target(Some(("a.png".into(), area())));
        sync_until(&mut view, &mut out, drawn);

        out.clear();
        view.invalidate();
        view.sync(&mut out, (10, 20)).unwrap();
        assert!(
            drawn(&out),
            "the cached PNG is drawn again in the same call"
        );
    }

    #[test]
    fn a_file_that_cannot_be_converted_removes_the_previous_image() {
        let mut view =
            ImagePreview::with_converter(|p| (p == Path::new("good.png")).then(|| png(800, 400)));
        let mut out = Vec::new();
        view.set_target(Some(("good.png".into(), area())));
        sync_until(&mut view, &mut out, drawn);

        out.clear();
        view.set_target(Some(("broken.png".into(), area())));
        sync_until(&mut view, &mut out, |o| o == CLEAR_SEQUENCE);
        assert!(view.placed.is_none());
    }
}
