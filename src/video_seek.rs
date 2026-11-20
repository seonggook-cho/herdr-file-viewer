//! Positions inside a video: what `ffprobe` reports, frame and time arithmetic, and the seek bar
//! line under the preview. Pure: no terminal and no processes, so every rule is unit-tested here.

/// What the preview needs to know about a video's picture stream.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VideoInfo {
    /// Frames per second as a fraction, e.g. 30000/1001.
    pub fps_num: u32,
    pub fps_den: u32,
    /// Number of frames; always at least 1.
    pub frames: u64,
    /// Length in seconds.
    pub duration: f64,
}

/// Parse `"30000/1001"`. A zero part (ffprobe prints `0/0` when it does not know) is `None`.
fn parse_rate(text: &str) -> Option<(u32, u32)> {
    let (num, den) = text.split_once('/')?;
    let (num, den) = (num.trim().parse().ok()?, den.trim().parse().ok()?);
    (num > 0 && den > 0).then_some((num, den))
}

fn number(value: &serde_json::Value) -> Option<f64> {
    match value {
        serde_json::Value::String(s) => s.parse().ok(),
        other => other.as_f64(),
    }
    .filter(|n: &f64| n.is_finite() && *n > 0.0)
}

impl VideoInfo {
    /// Read the JSON printed by
    /// `ffprobe -select_streams v:0 -show_entries stream=avg_frame_rate,r_frame_rate,nb_frames,duration:format=duration -of json`.
    pub fn from_probe_json(json: &str) -> Option<Self> {
        let root: serde_json::Value = serde_json::from_str(json).ok()?;
        let stream = root.get("streams")?.get(0)?;
        let rate = |key: &str| stream.get(key)?.as_str().and_then(parse_rate);
        let (fps_num, fps_den) = rate("avg_frame_rate").or_else(|| rate("r_frame_rate"))?;
        let duration = stream
            .get("duration")
            .and_then(number)
            .or_else(|| root.get("format")?.get("duration").and_then(number))?;
        let fps = f64::from(fps_num) / f64::from(fps_den);
        let frames = stream
            .get("nb_frames")
            .and_then(number)
            .map(|n| n as u64)
            .unwrap_or_else(|| (duration * fps).round() as u64)
            .max(1);
        Some(Self {
            fps_num,
            fps_den,
            frames,
            duration,
        })
    }

    pub fn fps(&self) -> f64 {
        f64::from(self.fps_num) / f64::from(self.fps_den)
    }

    pub fn last_frame(&self) -> u64 {
        self.frames - 1
    }

    /// When frame `n` (0-based) starts.
    pub fn frame_time(&self, n: u64) -> f64 {
        n as f64 / self.fps()
    }

    /// Where to seek so ffmpeg's first output is exactly frame `n`: half a frame early, so
    /// rounding in either direction still lands on `n` and never on `n - 1`.
    pub fn seek_time(&self, n: u64) -> f64 {
        if n == 0 {
            0.0
        } else {
            (n as f64 - 0.5) / self.fps()
        }
    }

    /// The frame showing at `seconds`, kept inside the video.
    pub fn frame_at(&self, seconds: f64) -> u64 {
        let n = (seconds.max(0.0) * self.fps()).floor() as u64;
        n.min(self.last_frame())
    }

    /// Move `delta` frames from `from`, stopping at the first and last frame.
    pub fn step(&self, from: u64, delta: i64) -> u64 {
        let to = from as i64 + delta;
        (to.max(0) as u64).min(self.last_frame())
    }

    /// Frames in one second, for the Shift+arrow jump.
    pub fn second_frames(&self) -> i64 {
        (self.fps().round() as i64).max(1)
    }
}

/// `65.123` → `"01:05.12"`. Minutes keep counting past an hour; a preview never needs hours.
pub fn format_time(seconds: f64) -> String {
    let centis = (seconds.max(0.0) * 100.0).floor() as u64;
    format!(
        "{:02}:{:02}.{:02}",
        centis / 6000,
        centis / 100 % 60,
        centis % 100
    )
}

/// The seek bar line and where its track sits, so a click can be mapped back to a frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bar {
    /// Exactly `width` characters, every one a single terminal cell wide.
    pub text: String,
    /// Columns of the track, relative to the start of the line.
    pub track_start: u16,
    pub track_width: u16,
}

const ICON_COLS: u16 = 3;
const MIN_TRACK: u16 = 10;

/// Draw `▶ ━━━━●──── 00:05.12 / 00:15.00  F 123/362` into `width` columns. Narrow lines drop the
/// frame counter first, then the times.
pub fn bar(info: &VideoInfo, frame: u64, playing: bool, width: u16) -> Bar {
    let icon = if playing { "▶  " } else { "❚❚ " };
    let times = format!(
        " {} / {}",
        format_time(info.frame_time(frame)),
        format_time(info.duration)
    );
    let counter = format!("  F {}/{}", frame + 1, info.frames);
    let fits = |suffix: &str| width >= ICON_COLS + MIN_TRACK + suffix.chars().count() as u16;
    let suffix = [format!("{times}{counter}"), times, String::new()]
        .into_iter()
        .find(|s| fits(s))
        .unwrap_or_default();
    let track_width = width.saturating_sub(ICON_COLS + suffix.chars().count() as u16);
    let mut text: String = icon.chars().take(width as usize).collect();
    if track_width > 0 {
        let dot = track_column(info, frame, track_width);
        text.extend((0..track_width).map(|c| match c.cmp(&dot) {
            std::cmp::Ordering::Less => '━',
            std::cmp::Ordering::Equal => '●',
            std::cmp::Ordering::Greater => '─',
        }));
        text.push_str(&suffix);
    }
    let pad = (width as usize).saturating_sub(text.chars().count());
    text.extend(std::iter::repeat_n(' ', pad));
    Bar {
        text,
        track_start: ICON_COLS.min(width),
        track_width,
    }
}

/// The track column of `frame`: the first frame at the left end, the last at the right end.
fn track_column(info: &VideoInfo, frame: u64, track_width: u16) -> u16 {
    if info.last_frame() == 0 || track_width <= 1 {
        return 0;
    }
    let span = f64::from(track_width - 1);
    (frame as f64 / info.last_frame() as f64 * span).round() as u16
}

impl Bar {
    /// The frame under column `col` of the line. Columns left or right of the track clamp to the
    /// first or last frame, so dragging past the ends still reaches them.
    pub fn frame_at_column(&self, info: &VideoInfo, col: u16) -> u64 {
        if self.track_width <= 1 {
            return 0;
        }
        let into = col
            .saturating_sub(self.track_start)
            .min(self.track_width - 1);
        let fraction = f64::from(into) / f64::from(self.track_width - 1);
        (fraction * info.last_frame() as f64).round() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(frames: u64) -> VideoInfo {
        VideoInfo {
            fps_num: 24,
            fps_den: 1,
            frames,
            duration: frames as f64 / 24.0,
        }
    }

    #[test]
    fn reads_a_typical_probe() {
        let json = r#"{"streams":[{"r_frame_rate":"24/1","avg_frame_rate":"24/1","duration":"5.041667","nb_frames":"121"}],"format":{"duration":"5.050000"}}"#;
        let v = VideoInfo::from_probe_json(json).unwrap();
        assert_eq!((v.fps_num, v.fps_den, v.frames), (24, 1, 121));
        assert!((v.duration - 5.041667).abs() < 1e-9);
    }

    #[test]
    fn falls_back_when_fields_are_missing() {
        // webm and GIFs often lack nb_frames and the stream duration; avg_frame_rate may be 0/0.
        let json = r#"{"streams":[{"r_frame_rate":"30000/1001","avg_frame_rate":"0/0"}],"format":{"duration":"10.0"}}"#;
        let v = VideoInfo::from_probe_json(json).unwrap();
        assert_eq!((v.fps_num, v.fps_den), (30000, 1001));
        assert_eq!(v.frames, 300); // round(10 * 29.97)
    }

    #[test]
    fn rejects_a_probe_without_a_usable_rate_or_length() {
        assert_eq!(VideoInfo::from_probe_json("not json"), None);
        assert_eq!(VideoInfo::from_probe_json(r#"{"streams":[]}"#), None);
        let no_rate = r#"{"streams":[{"r_frame_rate":"0/0"}],"format":{"duration":"3"}}"#;
        assert_eq!(VideoInfo::from_probe_json(no_rate), None);
        let no_length = r#"{"streams":[{"r_frame_rate":"25/1"}],"format":{}}"#;
        assert_eq!(VideoInfo::from_probe_json(no_length), None);
    }

    #[test]
    fn frames_and_times_round_trip() {
        let v = info(121);
        for n in [0, 1, 23, 24, 60, 120] {
            assert_eq!(v.frame_at(v.frame_time(n)), n);
            assert_eq!(v.frame_at(v.seek_time(n) + 0.5 / v.fps()), n);
        }
        assert_eq!(v.seek_time(0), 0.0);
        assert!(v.seek_time(1) > v.frame_time(0) && v.seek_time(1) < v.frame_time(1));
    }

    #[test]
    fn a_ntsc_rate_lands_on_whole_frames() {
        let v = VideoInfo {
            fps_num: 30000,
            fps_den: 1001,
            frames: 300,
            duration: 10.01,
        };
        assert_eq!(v.frame_at(v.frame_time(150)), 150);
        assert_eq!(v.second_frames(), 30);
    }

    #[test]
    fn stepping_stops_at_both_ends() {
        let v = info(121);
        assert_eq!(v.step(0, -1), 0);
        assert_eq!(v.step(5, 1), 6);
        assert_eq!(v.step(120, 1), 120);
        assert_eq!(v.step(10, -24), 0);
        assert_eq!(v.step(110, 24), 120);
        assert_eq!(v.frame_at(999.0), 120);
        assert_eq!(v.frame_at(-1.0), 0);
    }

    #[test]
    fn formats_minutes_seconds_and_hundredths() {
        assert_eq!(format_time(0.0), "00:00.00");
        assert_eq!(format_time(5.129), "00:05.12");
        assert_eq!(format_time(65.5), "01:05.50");
        assert_eq!(format_time(-3.0), "00:00.00");
    }

    #[test]
    fn a_wide_bar_shows_everything_in_exactly_its_width() {
        let v = info(121);
        let b = bar(&v, 60, false, 70);
        assert_eq!(b.text.chars().count(), 70);
        assert!(b.text.starts_with("❚❚ ━"));
        assert!(b.text.contains("● ") || b.text.contains("●─"));
        assert!(b.text.contains("00:02.50 / 00:05.04"));
        assert!(b.text.trim_end().ends_with("F 61/121"));
        assert!(bar(&v, 0, true, 70).text.starts_with("▶  ●"));
    }

    #[test]
    fn a_narrow_bar_drops_the_counter_then_the_times() {
        let v = info(121);
        let mid = bar(&v, 60, false, 36);
        assert!(!mid.text.contains("F 61"));
        assert!(mid.text.contains("00:02.50"));
        let small = bar(&v, 60, false, 16);
        assert!(!small.text.contains(':'));
        assert_eq!(small.text.chars().count(), 16);
        assert_eq!(bar(&v, 0, false, 2).text.chars().count(), 2);
    }

    #[test]
    fn the_dot_moves_from_the_left_end_to_the_right_end() {
        let v = info(121);
        let first = bar(&v, 0, false, 40);
        let last = bar(&v, 120, false, 40);
        let dot = |b: &Bar| b.text.chars().position(|c| c == '●').unwrap() as u16;
        assert_eq!(dot(&first), first.track_start);
        assert_eq!(dot(&last), last.track_start + last.track_width - 1);
    }

    #[test]
    fn a_column_maps_back_to_its_frame_and_clamps_outside_the_track() {
        let v = info(121);
        let b = bar(&v, 0, false, 70);
        assert_eq!(b.frame_at_column(&v, b.track_start), 0);
        assert_eq!(
            b.frame_at_column(&v, b.track_start + b.track_width - 1),
            120
        );
        assert_eq!(b.frame_at_column(&v, 0), 0);
        assert_eq!(b.frame_at_column(&v, 500), 120);
        // the dot drawn for a frame is read back as that frame (within one column's worth)
        let at = bar(&v, 60, false, 70);
        let dot = at.text.chars().position(|c| c == '●').unwrap() as u16;
        assert!(at.frame_at_column(&v, dot).abs_diff(60) <= 2);
    }

    #[test]
    fn a_single_frame_video_never_divides_by_zero() {
        let v = info(1);
        let b = bar(&v, 0, false, 40);
        assert_eq!(b.frame_at_column(&v, 20), 0);
        assert_eq!(v.step(0, 5), 0);
    }
}
