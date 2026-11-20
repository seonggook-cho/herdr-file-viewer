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
//!
//! A video with known length (read by `ffprobe`) also gets a seek bar on the pane's last row.
//! Stopped, the arrows step one frame (Shift: one second) and the bar takes clicks and drags; the
//! exact frame is pulled from the original at full preview size. A second worker extracts frames
//! in short batches and keeps them in memory around the position, so stepping answers at once.
//! Playing resumes from the position, picture and sound alike.

use crate::video_seek::{self, Bar, VideoInfo};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind};
use ratatui::layout::Rect;
use std::collections::BTreeMap;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Stdio};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::time::{Duration, Instant, SystemTime};

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
const FFPROBE_PATHS: &[&str] = &[
    "ffprobe",
    "/opt/homebrew/bin/ffprobe",
    "/usr/local/bin/ffprobe",
];
/// Frames extracted per request: ahead of the position, or fewer behind it when stepping back.
const BATCH_AHEAD: u32 = 30;
const BATCH_BEHIND: u32 = 10;
/// Extraction starts again once fewer than this many frames ahead are ready.
const PREFETCH_AHEAD: u64 = 10;
/// Extracted frames kept in memory; the ones farthest from the position go first.
const CACHE_FRAMES: usize = 120;
const CACHE_BYTES: usize = 256 << 20;
/// Stepped frames are drawn no faster than playback, the rate herdr is known to keep up with.
/// Faster, the host falls behind on image data and key presses queue up behind it.
const STEP_DRAW_GAP: Duration = Duration::from_millis(1000 / PLAY_FPS as u64);

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

fn start_playback(path: &Path, from: f64) -> Option<Playback> {
    let seek = format!("{from:.3}");
    let filter = format!(
        "fps={PLAY_FPS},scale=w='min({PLAY_MAX_EDGE_PX},iw)':h='min({PLAY_MAX_EDGE_PX},ih)':force_original_aspect_ratio=decrease"
    );
    let mut child = FFMPEG_PATHS.iter().find_map(|ffmpeg| {
        let mut cmd = Command::new(ffmpeg);
        cmd.args(["-nostdin", "-v", "error", "-protocol_whitelist", "file"])
            .args(["-ss", &seek, "-re", "-i"])
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
    children.extend(start_sound(path, &seek)); // a video with no sound still plays
    Some(Playback { children, frames })
}

fn start_sound(path: &Path, seek: &str) -> Option<Child> {
    FFPLAY_PATHS.iter().find_map(|ffplay| {
        let mut cmd = Command::new(ffplay);
        cmd.args([
            "-nodisp",
            "-vn",
            "-autoexit",
            "-nostats",
            "-loglevel",
            "quiet",
            "-ss",
            seek,
        ])
        .arg(path)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
        crate::proc::in_launch_dir(&mut cmd).spawn().ok()
    })
}

/// Where the frame worker gets a video's facts and frames. Tests swap in a fake.
trait VideoBackend: Send + 'static {
    fn probe(&self, path: &Path) -> Option<VideoInfo>;
    /// Up to `count` consecutive frames starting at `seconds`, no larger than `max` pixels.
    /// Dropping the iterator stops the work.
    fn frames(
        &self,
        path: &Path,
        seconds: f64,
        count: u32,
        max: (u32, u32),
    ) -> Box<dyn Iterator<Item = Vec<u8>>>;
}

/// The real backend: `ffprobe` for the facts, `ffmpeg` for exact frames at preview size.
struct Ffmpeg;

impl VideoBackend for Ffmpeg {
    fn probe(&self, path: &Path) -> Option<VideoInfo> {
        FFPROBE_PATHS.iter().find_map(|ffprobe| {
            let mut cmd = Command::new(ffprobe);
            cmd.args(["-v", "error", "-protocol_whitelist", "file"])
                .args(["-select_streams", "v:0", "-show_entries"])
                .arg("stream=avg_frame_rate,r_frame_rate,nb_frames,duration:format=duration")
                .args(["-of", "json"])
                .arg(path)
                .stdin(Stdio::null())
                .stderr(Stdio::null());
            let out = crate::proc::in_launch_dir(&mut cmd).output().ok()?;
            VideoInfo::from_probe_json(&String::from_utf8_lossy(&out.stdout))
        })
    }

    fn frames(
        &self,
        path: &Path,
        seconds: f64,
        count: u32,
        (max_w, max_h): (u32, u32),
    ) -> Box<dyn Iterator<Item = Vec<u8>>> {
        let scale = format!(
            "scale=w='min({max_w},iw)':h='min({max_h},ih)':force_original_aspect_ratio=decrease"
        );
        let (seek, count_arg) = (format!("{seconds:.4}"), count.to_string());
        let child = FFMPEG_PATHS.iter().find_map(|ffmpeg| {
            let mut cmd = Command::new(ffmpeg);
            cmd.args(["-nostdin", "-v", "error", "-protocol_whitelist", "file"])
                .args(["-ss", &seek, "-i"])
                .arg(path)
                .args(["-an", "-frames:v", &count_arg, "-vf", &scale])
                .args([
                    "-f",
                    "image2pipe",
                    "-c:v",
                    "png",
                    // Smaller frames reach the screen sooner: the event loop waits while the
                    // terminal reads them. Compressing harder costs only the worker's time.
                    "-pred",
                    "mixed",
                    "-compression_level",
                    "6",
                    "-",
                ])
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::null());
            crate::proc::in_launch_dir(&mut cmd).spawn().ok()
        });
        match child {
            Some(mut child) => match child.stdout.take() {
                Some(stdout) => Box::new(FrameStream { child, stdout }),
                None => Box::new(std::iter::empty()),
            },
            None => Box::new(std::iter::empty()),
        }
    }
}

/// PNG frames from a running `ffmpeg`, which is stopped when the stream is dropped.
struct FrameStream {
    child: Child,
    stdout: ChildStdout,
}

impl Iterator for FrameStream {
    type Item = Vec<u8>;
    fn next(&mut self) -> Option<Vec<u8>> {
        read_png(&mut self.stdout)
    }
}

impl Drop for FrameStream {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Used where no video work should happen (still-image tests).
#[cfg(test)]
struct NoVideo;

#[cfg(test)]
impl VideoBackend for NoVideo {
    fn probe(&self, _: &Path) -> Option<VideoInfo> {
        None
    }
    fn frames(&self, _: &Path, _: f64, _: u32, _: (u32, u32)) -> Box<dyn Iterator<Item = Vec<u8>>> {
        Box::new(std::iter::empty())
    }
}

#[derive(Debug, Clone, PartialEq)]
enum VideoJob {
    Probe(Source),
    Frames {
        source: Source,
        info: VideoInfo,
        start: u64,
        count: u32,
        /// The picture's size on screen in pixels: larger frames would only cost transfer time.
        max: (u32, u32),
    },
}

enum VideoResult {
    Info(Source, Option<VideoInfo>),
    Frame(Source, u64, Vec<u8>),
    /// A batch finished without being cut short by a newer job.
    Done(Source, u64),
}

/// One worker for video facts and frames. It always moves to the newest job, and a job that
/// arrives mid-batch stops the batch, so dragging or holding an arrow never queues work.
fn spawn_video_worker(backend: impl VideoBackend) -> (Sender<VideoJob>, Receiver<VideoResult>) {
    let (jobs, inbox) = mpsc::channel::<VideoJob>();
    let (outbox, results) = mpsc::channel();
    std::thread::spawn(move || {
        let mut next = None;
        loop {
            let job = match next.take() {
                Some(job) => job,
                None => match inbox.recv() {
                    Ok(job) => job,
                    Err(_) => return,
                },
            };
            let job = inbox.try_iter().last().unwrap_or(job);
            match job {
                VideoJob::Probe(source) => {
                    let info = backend.probe(&source.0);
                    if outbox.send(VideoResult::Info(source, info)).is_err() {
                        return;
                    }
                }
                VideoJob::Frames {
                    source,
                    info,
                    start,
                    count,
                    max,
                } => {
                    let mut frames = backend.frames(&source.0, info.seek_time(start), count, max);
                    for (n, png) in (start..).zip(frames.by_ref()) {
                        if outbox
                            .send(VideoResult::Frame(source.clone(), n, png))
                            .is_err()
                        {
                            return;
                        }
                        if let Ok(newer) = inbox.try_recv() {
                            next = Some(newer);
                            break;
                        }
                    }
                    drop(frames);
                    if next.is_none() && outbox.send(VideoResult::Done(source, start)).is_err() {
                        return;
                    }
                }
            }
        }
    });
    (jobs, results)
}

/// Drop the frames farthest from `center` until the cache is within its limits. `center` stays.
fn trim_cache(cache: &mut BTreeMap<u64, Vec<u8>>, bytes: &mut usize, center: u64) {
    while cache.len() > CACHE_FRAMES || (*bytes > CACHE_BYTES && cache.len() > 1) {
        let first = *cache.keys().next().expect("non-empty");
        let last = *cache.keys().next_back().expect("non-empty");
        let far = if center.abs_diff(first) >= center.abs_diff(last) {
            first
        } else {
            last
        };
        if far == center {
            break;
        }
        if let Some(png) = cache.remove(&far) {
            *bytes -= png.len();
        }
    }
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

/// The shown video's position, bar and extracted frames.
struct Video {
    source: Source,
    /// `None` until `ffprobe` answers, and for good when it cannot: then there is no bar.
    info: Option<VideoInfo>,
    /// The frame to show while stopped (0-based).
    frame: u64,
    /// The exact frame on screen; `None` forces the next sync to draw `frame` again.
    shown: Option<u64>,
    cache: BTreeMap<u64, Vec<u8>>,
    cache_bytes: usize,
    /// The pixel box the cached frames were extracted for: the picture's size on screen.
    frame_px: (u32, u32),
    /// The batch being extracted: frames `start..end`.
    in_flight: Option<(u64, u64)>,
    /// The bar as last drawn, for mapping clicks to frames, and whether it must be drawn again.
    bar: Option<Bar>,
    bar_stale: bool,
    /// Where the current playback started, and how many frames it has delivered.
    play_from: f64,
    played: u32,
}

impl Video {
    fn new(source: Source) -> Self {
        Self {
            source,
            info: None,
            frame: 0,
            shown: Some(0), // the first frame is what the still conversion shows
            cache: BTreeMap::new(),
            cache_bytes: 0,
            frame_px: (0, 0),
            in_flight: None,
            bar: None,
            bar_stale: true,
            play_from: 0.0,
            played: 0,
        }
    }

    /// Seconds into the video that playback has reached.
    fn play_position(&self) -> f64 {
        self.play_from + f64::from(self.played) / f64::from(PLAY_FPS)
    }

    fn receive(&mut self, result: VideoResult) {
        match result {
            VideoResult::Info(source, info) if source == self.source => {
                self.info = info;
                self.bar_stale = true;
            }
            VideoResult::Frame(source, n, png) if source == self.source => {
                self.cache_bytes += png.len();
                if let Some(old) = self.cache.insert(n, png) {
                    self.cache_bytes -= old.len();
                }
                trim_cache(&mut self.cache, &mut self.cache_bytes, self.frame);
            }
            VideoResult::Done(source, start)
                if source == self.source && self.in_flight.is_some_and(|(s, _)| s == start) =>
            {
                self.in_flight = None;
            }
            _ => {} // left over from a file no longer shown
        }
    }

    /// The batch to extract for `n`, unless one already covers it. Stepping back fetches a short
    /// batch ending at `n`; anything else fetches ahead from `n`.
    fn batch_for(&self, n: u64, info: &VideoInfo) -> Option<(u64, u32)> {
        if self.in_flight.is_some_and(|(s, e)| (s..e).contains(&n)) {
            return None;
        }
        let back = self.shown.is_some_and(|shown| n < shown);
        let start = if back {
            n.saturating_sub(u64::from(BATCH_BEHIND) - 1)
        } else {
            n
        };
        let want = if back { BATCH_BEHIND } else { BATCH_AHEAD };
        let count = (info.frames - start).min(u64::from(want)) as u32;
        Some((start, count))
    }

    /// Ask the worker for the batch holding frame `n`, unless one already covers it.
    fn request(&mut self, n: u64, info: &VideoInfo, jobs: &Sender<VideoJob>) {
        let Some((start, count)) = self.batch_for(n, info) else {
            return;
        };
        self.in_flight = Some((start, start + u64::from(count)));
        let _ = jobs.send(VideoJob::Frames {
            source: self.source.clone(),
            info: *info,
            start,
            count,
            max: self.frame_px,
        });
    }

    /// The first frame just ahead of the position that is not extracted yet, to fetch while idle.
    fn prefetch(&self, info: &VideoInfo) -> Option<u64> {
        if self.in_flight.is_some() {
            return None;
        }
        let end = (self.frame + PREFETCH_AHEAD).min(info.last_frame());
        (self.frame + 1..=end).find(|n| !self.cache.contains_key(n))
    }
}

pub struct ImagePreview {
    jobs: Sender<Source>,
    results: Receiver<Converted>,
    wanted: Option<Target>,
    placed: Option<Target>,
    requested: Option<Source>,
    converted: Option<Converted>,
    player: fn(&Path, f64) -> Option<Playback>,
    playback: Option<Playback>,
    /// How many frames have been drawn since the last clear, which picks each frame's id and layer.
    frames_drawn: u32,
    video_jobs: Sender<VideoJob>,
    video_results: Receiver<VideoResult>,
    video: Option<Video>,
    /// A drag that began on the seek bar, followed even when the pointer leaves the bar's row.
    dragging: bool,
    /// More input is already waiting: a stepped frame (megabytes of image data) waits for it, so
    /// a held arrow key draws only where it stops while the bar follows every step.
    input_waiting: bool,
    /// Where the seek bar is on screen. ratatui does not know about it, so it is blanked before
    /// every repaint and written again afterwards.
    bar_on_screen: Option<Rect>,
    /// When the last stepped frame was drawn, and the least time between two.
    last_step_draw: Option<Instant>,
    step_gap: Duration,
}

/// A video keeps its last row for the seek bar; the picture gets the rest.
fn picture_rect(target: &Target) -> Rect {
    let mut rect = target.rect;
    if has_extension(&target.source.0, ANIMATED_EXTENSIONS) && rect.height >= 2 {
        rect.height -= 1;
    }
    rect
}

fn bar_row(target: &Target) -> Option<Rect> {
    let picture = picture_rect(target);
    (picture.height < target.rect.height).then(|| Rect {
        y: picture.y + picture.height,
        height: 1,
        ..target.rect
    })
}

impl ImagePreview {
    pub fn new() -> Self {
        Self::with_parts(convert, Ffmpeg)
    }

    /// Whether frames are arriving, so the event loop should tick fast enough to show them.
    pub fn playing(&self) -> bool {
        self.playback.is_some()
    }

    /// Whether the loop should tick fast: playing, or a stepped-to frame is still on its way.
    pub fn busy(&self) -> bool {
        self.playing()
            || self
                .video
                .as_ref()
                .is_some_and(|v| v.in_flight.is_some() || v.shown != Some(v.frame))
    }

    /// Whether the pane is showing a video or GIF, so Space should play it.
    pub fn video_shown(&self) -> bool {
        self.placed
            .as_ref()
            .is_some_and(|t| has_extension(&t.source.0, ANIMATED_EXTENSIONS))
    }

    /// Whether the shown video has a known length, so the arrows and the bar move through it.
    pub fn seekable(&self) -> bool {
        self.video_shown() && self.video.as_ref().is_some_and(|v| v.info.is_some())
    }

    /// Play the shown video from its position, or stop it where it is.
    pub fn toggle_play(&mut self) {
        if self.playback.is_some() {
            self.stop_playback();
            return;
        }
        let Some(target) = self.placed.as_ref().filter(|_| self.video_shown()) else {
            return;
        };
        let path = target.source.0.clone();
        let from = match self.video.as_mut() {
            Some(v) => {
                let from = v.info.map_or(0.0, |info| {
                    let start = if v.frame >= info.last_frame() {
                        0
                    } else {
                        v.frame
                    };
                    info.frame_time(start)
                });
                (v.play_from, v.played, v.bar_stale) = (from, 0, true);
                from
            }
            None => 0.0,
        };
        self.playback = (self.player)(&path, from);
    }

    /// Stop playing and settle on the exact frame it reached.
    fn stop_playback(&mut self) {
        if self.playback.take().is_none() {
            return;
        }
        if let Some(v) = self.video.as_mut()
            && let Some(info) = v.info
        {
            v.frame = info.frame_at(v.play_position());
            v.shown = None; // replace the smaller playback frame with the exact one
            v.bar_stale = true;
        }
    }

    /// Step through the stopped video: frames, or whole seconds with Shift.
    pub fn handle_key(&mut self, key: &KeyEvent) -> bool {
        let sign = match key.code {
            KeyCode::Left => -1,
            KeyCode::Right => 1,
            _ => return false,
        };
        let Some(info) = self.video.as_ref().and_then(|v| v.info) else {
            return false;
        };
        let delta = if key.modifiers.contains(KeyModifiers::SHIFT) {
            sign * info.second_frames()
        } else {
            sign
        };
        self.stop_playback();
        self.seek_to(info.step(self.current_frame(), delta));
        true
    }

    /// Click or drag on the seek bar. Returns whether the event was the bar's.
    pub fn handle_mouse(&mut self, ev: &MouseEvent) -> bool {
        let hit =
            self.placed.as_ref().and_then(bar_row).is_some_and(|row| {
                ev.row == row.y && (row.x..row.x + row.width).contains(&ev.column)
            });
        match ev.kind {
            MouseEventKind::Down(MouseButton::Left) if hit && self.seekable() => {
                self.dragging = true;
            }
            MouseEventKind::Drag(MouseButton::Left) if self.dragging => {}
            MouseEventKind::Up(MouseButton::Left) if self.dragging => {
                self.dragging = false;
                return true;
            }
            _ => return false,
        }
        let column = self
            .placed
            .as_ref()
            .map_or(0, |t| ev.column.saturating_sub(t.rect.x));
        let Some((info, bar)) = self.video.as_ref().and_then(|v| v.info.zip(v.bar.clone())) else {
            return true;
        };
        self.stop_playback();
        self.seek_to(bar.frame_at_column(&info, column));
        true
    }

    fn current_frame(&self) -> u64 {
        self.video.as_ref().map_or(0, |v| v.frame)
    }

    fn seek_to(&mut self, frame: u64) {
        if let Some(v) = self.video.as_mut() {
            v.frame = frame;
            v.bar_stale = true;
        }
    }

    /// One worker that skips to the newest job, so holding an arrow key never queues conversions.
    #[cfg(test)]
    fn with_converter(converter: impl Fn(&Path) -> Option<Vec<u8>> + Send + 'static) -> Self {
        Self::with_parts(converter, NoVideo)
    }

    fn with_parts(
        converter: impl Fn(&Path) -> Option<Vec<u8>> + Send + 'static,
        backend: impl VideoBackend,
    ) -> Self {
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
        let (video_jobs, video_results) = spawn_video_worker(backend);
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
            video_jobs,
            video_results,
            video: None,
            dragging: false,
            input_waiting: false,
            bar_on_screen: None,
            last_step_draw: None,
            step_gap: STEP_DRAW_GAP,
        }
    }

    /// Choose what to show. `None` removes the image. Called after every repaint, which may have
    /// drawn over the seek bar, so the bar is written again.
    pub fn set_target(&mut self, target: Option<(PathBuf, Rect)>) {
        self.wanted = target.map(|(path, rect)| Target {
            source: source_of(path),
            rect,
        });
        if let Some(v) = self.video.as_mut() {
            v.bar_stale = true;
        }
    }

    /// Blank the seek bar before ratatui repaints, so a bar never outlives its video. The row is
    /// blank in ratatui's own buffer (a video's placeholder is one line at the top), so the
    /// repaint and the screen agree again. [`sync`](Self::sync) writes the bar back if it stays.
    pub fn erase_bar(&mut self, out: &mut impl Write) -> io::Result<()> {
        let Some(row) = self.bar_on_screen.take() else {
            return Ok(());
        };
        if let Some(v) = self.video.as_mut() {
            v.bar_stale = true;
        }
        let blank = " ".repeat(usize::from(row.width));
        write!(
            out,
            "\x1b7\x1b[{};{}H\x1b[0m{blank}\x1b8",
            row.y + 1,
            row.x + 1
        )?;
        out.flush()
    }

    /// Tell the preview whether input is already waiting to be read.
    pub fn set_input_waiting(&mut self, waiting: bool) {
        self.input_waiting = waiting;
    }

    /// Forget the placement after the screen was wiped, so the next [`sync`](Self::sync) redraws.
    pub fn invalidate(&mut self) {
        self.placed = None;
    }

    /// Advance towards the wanted state, writing to the terminal only when something changes.
    pub fn sync(&mut self, out: &mut impl Write, cell: (u32, u32)) -> io::Result<()> {
        self.converted = self.results.try_iter().last().or(self.converted.take());
        for result in self.video_results.try_iter().collect::<Vec<_>>() {
            if let Some(v) = self.video.as_mut() {
                v.receive(result);
            }
        }
        if self.wanted == self.placed {
            self.play(out, cell)?;
            self.show_frame(out, cell)?;
            return self.draw_bar(out);
        }
        self.stop_playback();
        self.dragging = false;
        let Some(wanted) = self.wanted.clone() else {
            self.placed = None;
            self.video = None;
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
                let picture = picture_rect(&wanted);
                out.write_all(CLEAR_SEQUENCE)?;
                let placement = fit(size, (picture.width, picture.height), cell);
                out.write_all(&place_sequence(png, picture, placement, IMAGE_IDS[0], 0))?;
                self.frames_drawn = 0;
                self.adopt_video(&wanted);
                self.placed = Some(wanted);
                self.show_frame(out, cell)?;
                self.draw_bar(out)?;
            }
            None => {
                self.video = None;
                if self.placed.take().is_some() {
                    out.write_all(CLEAR_SEQUENCE)?;
                }
            }
        }
        out.flush()
    }

    /// Keep the position when the same video is placed again (a resize or a wiped screen), start
    /// fresh for another video, and drop it for a still image.
    fn adopt_video(&mut self, target: &Target) {
        if !has_extension(&target.source.0, ANIMATED_EXTENSIONS) {
            self.video = None;
            return;
        }
        match self.video.as_mut() {
            Some(v) if v.source == target.source => {
                v.shown = Some(0); // the placement just drew the first frame
                v.bar_stale = true;
            }
            _ => {
                self.video = Some(Video::new(target.source.clone()));
                let _ = self.video_jobs.send(VideoJob::Probe(target.source.clone()));
            }
        }
    }

    /// While stopped, put the exact frame at the position on screen, or ask the worker for it.
    fn show_frame(&mut self, out: &mut impl Write, cell: (u32, u32)) -> io::Result<()> {
        if self.playback.is_some() {
            return Ok(());
        }
        let Some(picture) = self.placed.as_ref().map(picture_rect) else {
            return Ok(());
        };
        let Some(v) = self.video.as_mut() else {
            return Ok(());
        };
        let Some(info) = v.info else {
            return Ok(());
        };
        let max = (
            (u32::from(picture.width) * cell.0).clamp(1, MAX_EDGE_PX),
            (u32::from(picture.height) * cell.1).clamp(1, MAX_EDGE_PX),
        );
        if v.frame_px != max {
            // frames extracted for another pane size would be blurry or wastefully large
            (v.frame_px, v.in_flight, v.cache_bytes) = (max, None, 0);
            v.cache.clear();
        }
        if v.shown == Some(v.frame) {
            if let Some(n) = v.prefetch(&info) {
                v.request(n, &info, &self.video_jobs);
            }
            return Ok(());
        }
        let frame = v.frame;
        let Some(png) = v.cache.get(&frame).cloned() else {
            v.request(frame, &info, &self.video_jobs);
            return Ok(()); // the previous frame stays until this one is ready
        };
        let too_soon = self
            .last_step_draw
            .is_some_and(|at| at.elapsed() < self.step_gap);
        if v.shown.is_some() && (self.input_waiting || too_soon) {
            return Ok(()); // the bar shows the step meanwhile; the picture follows shortly
        }
        v.shown = Some(frame);
        self.last_step_draw = Some(Instant::now());
        self.draw_layered(out, &png, cell)
    }

    /// The seek bar under the picture, written straight to the terminal like the picture.
    fn draw_bar(&mut self, out: &mut impl Write) -> io::Result<()> {
        let playing = self.playback.is_some();
        let (Some(v), Some(row)) = (self.video.as_mut(), self.placed.as_ref().and_then(bar_row))
        else {
            return Ok(());
        };
        let Some(info) = v.info.filter(|_| v.bar_stale) else {
            return Ok(());
        };
        let frame = if playing {
            info.frame_at(v.play_position())
        } else {
            v.frame
        };
        let bar = video_seek::bar(&info, frame, playing, row.width);
        write!(
            out,
            "\x1b7\x1b[{};{}H\x1b[0m{}\x1b8",
            row.y + 1,
            row.x + 1,
            bar.text
        )?;
        v.bar = Some(bar);
        v.bar_stale = false;
        self.bar_on_screen = Some(row);
        out.flush()
    }

    /// Draw the newest frame on top of the previous one, then erase the frame before that.
    fn play(&mut self, out: &mut impl Write, cell: (u32, u32)) -> io::Result<()> {
        let Some(playback) = &self.playback else {
            return Ok(());
        };
        let mut newest = None;
        let mut received = 0;
        let ended = loop {
            match playback.frames.try_recv() {
                Ok(frame) => {
                    newest = Some(frame);
                    received += 1;
                }
                Err(TryRecvError::Empty) => break false,
                Err(TryRecvError::Disconnected) => break true,
            }
        };
        if let Some(v) = self.video.as_mut() {
            v.played += received;
            v.bar_stale |= received > 0;
        }
        if ended {
            self.playback = None;
            if let Some(v) = self.video.as_mut()
                && let Some(info) = v.info
            {
                (v.frame, v.shown, v.bar_stale) = (info.last_frame(), None, true);
            }
        }
        let Some(frame) = newest else {
            return Ok(());
        };
        self.draw_layered(out, &frame, cell)
    }

    fn draw_layered(
        &mut self,
        out: &mut impl Write,
        png: &[u8],
        cell: (u32, u32),
    ) -> io::Result<()> {
        let Some(target) = &self.placed else {
            return Ok(());
        };
        let Some(size) = png_size(png) else {
            return Ok(());
        };
        let picture = picture_rect(target);
        self.frames_drawn += 1;
        let id_at = |n: u32| IMAGE_IDS[n as usize % IMAGE_IDS.len()];
        let placement = fit(size, (picture.width, picture.height), cell);
        let (id, oldest) = (id_at(self.frames_drawn), id_at(self.frames_drawn + 1));
        out.write_all(&place_sequence(
            png,
            picture,
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
    fn fake_player(path: &Path, from: f64) -> Option<Playback> {
        PLAYED_FROM.with(|p| p.set(Some(from)));
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
    fn short_player(_: &Path, _: f64) -> Option<Playback> {
        let (sender, frames) = mpsc::sync_channel(1);
        sender.send(png(320, 180)).unwrap();
        Some(Playback {
            children: Vec::new(),
            frames,
        })
    }

    thread_local! {
        /// Where the fake player was last asked to start.
        static PLAYED_FROM: std::cell::Cell<Option<f64>> = const { std::cell::Cell::new(None) };
    }

    /// 2 seconds at 24 fps. Frame `n` is a PNG `100 + n` pixels wide, so a drawn frame is known.
    struct FakeVideo;

    const FAKE: VideoInfo = VideoInfo {
        fps_num: 24,
        fps_den: 1,
        frames: 48,
        duration: 2.0,
    };

    impl VideoBackend for FakeVideo {
        fn probe(&self, _: &Path) -> Option<VideoInfo> {
            Some(FAKE)
        }
        fn frames(
            &self,
            _: &Path,
            seconds: f64,
            count: u32,
            _: (u32, u32),
        ) -> Box<dyn Iterator<Item = Vec<u8>>> {
            let start = FAKE.frame_at(seconds + 0.5 / FAKE.fps());
            Box::new((start..start + u64::from(count)).map(|n| png(100 + n as u32, 50)))
        }
    }

    fn frame_png(n: u64) -> Vec<u8> {
        png(100 + n as u32, 50)
    }

    fn was_drawn(out: &[u8], n: u64) -> bool {
        let encoded = encode_base64(&frame_png(n));
        out.windows(encoded.len()).any(|w| w == encoded.as_slice())
    }

    fn key(code: KeyCode, modifiers: KeyModifiers) -> KeyEvent {
        KeyEvent::new(code, modifiers)
    }

    fn mouse(kind: MouseEventKind, column: u16, row: u16) -> MouseEvent {
        MouseEvent {
            kind,
            column,
            row,
            modifiers: KeyModifiers::NONE,
        }
    }

    /// A video with a known length, shown and probed.
    fn seek_preview() -> (ImagePreview, Vec<u8>) {
        let mut view = ImagePreview::with_parts(|_| Some(png(800, 400)), FakeVideo);
        view.player = fake_player;
        view.step_gap = Duration::ZERO; // the pacing has its own test
        let mut out = Vec::new();
        view.set_target(Some(("v.mp4".into(), area())));
        settle(&mut view, &mut out, |v| v.seekable());
        (view, out)
    }

    /// Sync until the preview reaches a state.
    fn settle(view: &mut ImagePreview, out: &mut Vec<u8>, done: impl Fn(&ImagePreview) -> bool) {
        let start = Instant::now();
        while !done(view) {
            assert!(
                start.elapsed() < Duration::from_secs(5),
                "preview never settled"
            );
            view.sync(out, (10, 20)).unwrap();
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    fn at(view: &ImagePreview) -> (u64, Option<u64>) {
        let v = view.video.as_ref().expect("a video is shown");
        (v.frame, v.shown)
    }

    fn shows(n: u64) -> impl Fn(&ImagePreview) -> bool {
        move |view| at(view) == (n, Some(n))
    }

    #[test]
    fn a_probed_video_gets_a_bar_under_a_picture_one_row_shorter() {
        let (_view, out) = seek_preview();
        let text = String::from_utf8_lossy(&out);
        assert!(
            text.contains("\x1b[42;31H"),
            "the bar sits on the area's last row"
        );
        assert!(
            text.contains("❚❚ ●")
                && text.contains("00:00.00 / 00:02.00")
                && text.contains("F 1/48")
        );
        assert!(
            text.contains("r=39") || text.contains("c=80"),
            "the picture fits 39 of 40 rows"
        );
    }

    #[test]
    fn the_arrows_show_the_exact_next_and_previous_frame() {
        let (mut view, mut out) = seek_preview();
        assert!(view.handle_key(&key(KeyCode::Right, KeyModifiers::NONE)));
        settle(&mut view, &mut out, shows(1));
        assert!(was_drawn(&out, 1));
        assert!(String::from_utf8_lossy(&out).contains("F 2/48"));

        out.clear();
        view.handle_key(&key(KeyCode::Left, KeyModifiers::NONE));
        settle(&mut view, &mut out, shows(0));
        assert!(
            was_drawn(&out, 0),
            "going back draws the exact first frame from the cache"
        );
    }

    #[test]
    fn shift_jumps_a_second_and_both_ends_hold() {
        let (mut view, mut out) = seek_preview();
        view.handle_key(&key(KeyCode::Left, KeyModifiers::NONE));
        assert_eq!(at(&view).0, 0);
        view.handle_key(&key(KeyCode::Right, KeyModifiers::SHIFT));
        assert_eq!(at(&view).0, 24);
        view.handle_key(&key(KeyCode::Right, KeyModifiers::SHIFT));
        assert_eq!(at(&view).0, 47);
        settle(&mut view, &mut out, shows(47));
        view.handle_key(&key(KeyCode::Left, KeyModifiers::SHIFT));
        assert_eq!(at(&view).0, 23);
        assert!(
            !view.handle_key(&key(KeyCode::Up, KeyModifiers::NONE)),
            "other keys pass through"
        );
    }

    #[test]
    fn the_bar_is_blanked_before_a_repaint_and_gone_after_leaving_the_video() {
        let (mut view, mut out) = seek_preview();
        out.clear();
        view.erase_bar(&mut out).unwrap();
        let text = String::from_utf8_lossy(&out);
        assert!(text.contains(&format!("\x1b[42;31H\x1b[0m{}", " ".repeat(80))));

        out.clear();
        view.set_target(Some(("v.mp4".into(), area())));
        view.sync(&mut out, (10, 20)).unwrap();
        assert!(
            String::from_utf8_lossy(&out).contains("F 1/48"),
            "a video gets it back"
        );

        view.erase_bar(&mut out).unwrap();
        view.set_target(Some(("a.png".into(), area())));
        out.clear();
        sync_until(&mut view, &mut out, drawn);
        view.erase_bar(&mut out).unwrap();
        assert!(!String::from_utf8_lossy(&out).contains("F 1/48"));
        out.clear();
        view.erase_bar(&mut out).unwrap();
        assert!(out.is_empty(), "nothing to blank once the bar is gone");
    }

    #[test]
    fn stepped_frames_are_drawn_no_faster_than_playback() {
        let (mut view, mut out) = seek_preview();
        view.step_gap = Duration::from_millis(200);
        view.handle_key(&key(KeyCode::Right, KeyModifiers::NONE));
        settle(&mut view, &mut out, shows(1));
        view.handle_key(&key(KeyCode::Right, KeyModifiers::NONE));
        out.clear();
        view.sync(&mut out, (10, 20)).unwrap();
        assert_eq!(
            at(&view),
            (2, Some(1)),
            "too soon after the last drawn step"
        );
        assert!(
            String::from_utf8_lossy(&out).contains("F 3/48"),
            "the bar moves at once"
        );
        assert!(
            view.busy(),
            "the loop keeps ticking until the frame is drawn"
        );
        settle(&mut view, &mut out, shows(2));
    }

    #[test]
    fn frames_are_extracted_at_the_pictures_size_on_screen() {
        let (mut view, mut out) = seek_preview();
        view.handle_key(&key(KeyCode::Right, KeyModifiers::NONE));
        settle(&mut view, &mut out, shows(1));
        // 80 x 39 cells (one row is the bar) at 10 x 20 pixels each, capped at MAX_EDGE_PX
        assert_eq!(view.video.as_ref().unwrap().frame_px, (800, 780));

        view.set_target(Some(("v.mp4".into(), Rect::new(30, 2, 40, 21))));
        sync_until(&mut view, &mut Vec::new(), drawn);
        settle(&mut view, &mut out, shows(1));
        let v = view.video.as_ref().unwrap();
        assert_eq!(v.frame_px, (400, 400));
        assert!(
            v.cache.len() <= BATCH_AHEAD as usize,
            "frames for the old size were dropped"
        );
    }

    #[test]
    fn waiting_input_moves_the_bar_but_draws_only_the_last_step() {
        let (mut view, mut out) = seek_preview();
        view.handle_key(&key(KeyCode::Right, KeyModifiers::NONE));
        settle(&mut view, &mut out, shows(1)); // frames 1.. are in memory now
        view.set_input_waiting(true);
        for _ in 0..3 {
            view.handle_key(&key(KeyCode::Right, KeyModifiers::NONE));
            out.clear();
            view.sync(&mut out, (10, 20)).unwrap();
            assert!(!drawn(&out), "no image while keys are still coming");
        }
        assert!(
            String::from_utf8_lossy(&out).contains("F 5/48"),
            "the bar follows each step"
        );
        view.set_input_waiting(false);
        view.sync(&mut out, (10, 20)).unwrap();
        assert_eq!(at(&view), (4, Some(4)));
        assert!(was_drawn(&out, 4) && !was_drawn(&out, 3));
    }

    #[test]
    fn stepped_frames_come_from_one_batch_then_from_memory() {
        let (mut view, mut out) = seek_preview();
        view.handle_key(&key(KeyCode::Right, KeyModifiers::NONE));
        settle(&mut view, &mut out, |v| {
            shows(1)(v) && v.video.as_ref().unwrap().in_flight.is_none()
        });
        let cached = view.video.as_ref().unwrap().cache.len();
        assert_eq!(
            cached, BATCH_AHEAD as usize,
            "one batch ahead of the position"
        );
        for _ in 0..5 {
            view.handle_key(&key(KeyCode::Right, KeyModifiers::NONE));
            view.sync(&mut out, (10, 20)).unwrap();
        }
        assert_eq!(
            at(&view),
            (6, Some(6)),
            "cached frames show in the same sync"
        );
    }

    #[test]
    fn a_still_image_and_an_unprobed_video_leave_the_arrows_alone() {
        let mut view = ImagePreview::with_parts(|_| Some(png(800, 400)), FakeVideo);
        let mut out = Vec::new();
        view.set_target(Some(("a.png".into(), area())));
        sync_until(&mut view, &mut out, drawn);
        assert!(!view.seekable());
        assert!(!view.handle_key(&key(KeyCode::Right, KeyModifiers::NONE)));

        let mut view = video_preview(); // its backend never learns the length
        out.clear();
        view.set_target(Some(("v.mp4".into(), area())));
        sync_until(&mut view, &mut out, drawn);
        assert!(view.video_shown() && !view.seekable());
        assert!(
            !String::from_utf8_lossy(&out).contains("❚❚"),
            "no bar without a length"
        );
    }

    #[test]
    fn stopping_settles_on_the_exact_frame_playback_reached() {
        let (mut view, mut out) = seek_preview();
        view.toggle_play();
        view.sync(&mut out, (10, 20)).unwrap(); // the fake player delivers two frames
        out.clear();
        view.toggle_play();
        // two frames at 15 fps = 0.133 s = frame 3 at 24 fps
        assert_eq!(at(&view), (3, None));
        settle(&mut view, &mut out, shows(3));
        assert!(was_drawn(&out, 3));
    }

    #[test]
    fn playing_starts_from_the_position_and_again_from_the_top_at_the_end() {
        let (mut view, _out) = seek_preview();
        view.handle_key(&key(KeyCode::Right, KeyModifiers::SHIFT));
        view.toggle_play();
        assert_eq!(PLAYED_FROM.with(|p| p.get()), Some(1.0));
        view.toggle_play();

        view.seek_to(47);
        view.toggle_play();
        assert_eq!(PLAYED_FROM.with(|p| p.get()), Some(0.0));
    }

    #[test]
    fn an_arrow_while_playing_stops_then_steps() {
        let (mut view, mut out) = seek_preview();
        view.toggle_play();
        view.sync(&mut out, (10, 20)).unwrap();
        view.handle_key(&key(KeyCode::Right, KeyModifiers::NONE));
        assert!(!view.playing());
        assert_eq!(at(&view).0, 4);
    }

    #[test]
    fn the_end_of_playback_rests_on_the_last_frame() {
        let (mut view, mut out) = seek_preview();
        view.player = short_player;
        view.toggle_play();
        view.sync(&mut out, (10, 20)).unwrap();
        assert!(!view.playing());
        settle(&mut view, &mut out, shows(47));
    }

    #[test]
    fn clicking_and_dragging_the_bar_seeks() {
        let (mut view, mut out) = seek_preview();
        let bar = view.video.as_ref().unwrap().bar.clone().unwrap();
        let (x, row) = (area().x, area().y + area().height - 1);
        let right_end = x + bar.track_start + bar.track_width - 1;

        assert!(!view.handle_mouse(&mouse(
            MouseEventKind::Down(MouseButton::Left),
            right_end,
            row - 1
        )));
        assert!(view.handle_mouse(&mouse(
            MouseEventKind::Down(MouseButton::Left),
            right_end,
            row
        )));
        assert_eq!(at(&view).0, 47);
        // the drag is followed above the bar's row too
        assert!(view.handle_mouse(&mouse(MouseEventKind::Drag(MouseButton::Left), x, row - 5)));
        assert_eq!(at(&view).0, 0);
        assert!(view.handle_mouse(&mouse(MouseEventKind::Up(MouseButton::Left), x, row)));
        assert!(!view.handle_mouse(&mouse(
            MouseEventKind::Drag(MouseButton::Left),
            right_end,
            row
        )));
        settle(&mut view, &mut out, shows(0));
    }

    #[test]
    fn a_resize_keeps_the_position_and_another_file_starts_over() {
        let (mut view, mut out) = seek_preview();
        view.seek_to(10);
        settle(&mut view, &mut out, shows(10));
        view.set_target(Some(("v.mp4".into(), Rect::new(30, 2, 60, 30))));
        out.clear();
        sync_until(&mut view, &mut out, drawn);
        settle(&mut view, &mut out, shows(10));
        assert!(
            was_drawn(&out, 10),
            "the stepped frame is drawn again over the first one"
        );

        view.set_target(Some(("w.mp4".into(), area())));
        settle(&mut view, &mut out, |v| {
            v.seekable() && v.placed.as_ref().unwrap().source.0 == Path::new("w.mp4")
        });
        assert_eq!(at(&view), (0, Some(0)));
    }

    #[test]
    fn the_cache_drops_the_frames_farthest_from_the_position() {
        let mut cache = BTreeMap::new();
        let mut bytes = 0;
        for n in 0..(CACHE_FRAMES as u64 + 10) {
            bytes += 1;
            cache.insert(n, vec![0]);
        }
        trim_cache(&mut cache, &mut bytes, 20);
        assert_eq!(cache.len(), CACHE_FRAMES);
        assert_eq!(bytes, CACHE_FRAMES);
        assert!(cache.contains_key(&20) && cache.contains_key(&0));
        assert!(
            !cache.contains_key(&(CACHE_FRAMES as u64 + 9)),
            "the far end goes first"
        );
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
    fn only_a_video_claims_the_play_key() {
        let mut view = video_preview();
        let mut out = Vec::new();
        assert!(!view.video_shown(), "nothing is shown yet");
        view.set_target(Some(("v.mp4".into(), area())));
        sync_until(&mut view, &mut out, drawn);
        assert!(view.video_shown());
    }

    #[test]
    fn the_key_does_nothing_on_a_still_image() {
        let mut view = video_preview();
        let mut out = Vec::new();
        view.set_target(Some(("a.png".into(), area())));
        sync_until(&mut view, &mut out, drawn);
        assert!(!view.video_shown(), "Space stays page-down on an image");
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
