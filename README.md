# Cut-Helper

Desktop toolkit for streamer video editing. It detects Russian and English profanity locally with `faster-whisper`, can censor selected intervals, and includes a local studio for transcripts, highlight signals, subtitles, chat activity, projects, and export profiles.

The **Студия** tab is independent from **Цензура**: it has its own video field, model, transcript, chat, moments, project and export settings. **Полная расшифровка** creates searchable word-level speech context and detects the language. **Импорт чата** accepts timestamped text logs or JSON messages. **Посмотреть видео** samples the actual frames, runs Russian/English OCR, detects dense right-side chat blocks like a stream overlay, and combines the visible text with nearby transcript speech into a reviewable description. This is a local visual review aid, not a fabricated claim about events the model cannot see. Projects use the `.cutproj` format and include the studio source, transcript, chat, moments, and censor decisions.

Export profiles are `Original 16:9`, `YouTube 16:9`, `Shorts 9:16`, `TikTok 9:16`, `Square 1:1`, and `Audio MP3`. For vertical exports, layouts include a normal frame, `Стример сверху + чат снизу`, and `Игра сверху + чат снизу`; the latter two reserve a separate lower chat panel. SRT subtitles can be saved separately or burned into exported video, with detected profanity replaced by blocks. Shorts subtitles use the installed `Pixeloid Sans` pixel font in a small bottom position; if that font is unavailable, FFmpeg falls back to Arial. The vertical profiles use face-centered cropping when OpenCV is installed; otherwise they use a centered crop.


## Requirements

- Python 3.10+
- `ffmpeg` and `ffprobe` on `PATH`
- Tk support for your Python installation
- Install `requirements.txt` for the GUI, drag-and-drop support, and speech recognition. The PyAV and tkinterdnd2 version limits keep compatibility with faster-whisper and the Tcl 8.6 libraries provided on current Arch systems. The first run downloads the selected Whisper model. CPU uses int8; CUDA uses float16 when available, with automatic CPU fallback if CUDA libraries are missing.
- OpenCV is optional. Install `opencv-python` if you want face-centered framing for Shorts/TikTok exports.
- The recommended accuracy model is `large-v3-turbo`; it is downloaded by faster-whisper from the model registry on first use. Use `small` on a low-memory CPU. `large-v3-turbo` needs more VRAM/RAM but handles Russian, English, accents, and noisy stream audio better than the old `base` model.

On Debian/Ubuntu install the system packages with `sudo apt install ffmpeg python3-tk`. On Arch Linux use `sudo pacman -S ffmpeg tk`. On Windows, install FFmpeg and add its `bin` directory to `PATH`.

## Run

### Fish on Linux

On Arch Linux, install the system dependencies, including pip:

```fish
sudo pacman -S ffmpeg tk python python-pip
```

Create the environment and install the Python dependencies:

```fish
cd /home/future/video_censor
python3 -m venv .venv
source .venv/bin/activate.fish
python -m pip install -r requirements.txt
```

Start the application with `fish run.fish`. During analysis, recognized words are added to the results list as they arrive with exact start/end timecodes. Paste a YouTube link and press **Скачать превью** to save the thumbnail and show the video title; its destination is controlled by **Папка превью**. Use **Скачать видео** for a public YouTube/Twitch recording or live stream; its destination is controlled independently by **Папка стримов**. Active streams continue downloading until they end or you press **Остановить**; partial data is retained for resuming. Completed renders are stored in `~/.local/share/cut-helper/history.json` and shown in the in-app history panel.

The app remembers the last normal window size and position in `~/.local/share/cut-helper/settings.json`.

For NVIDIA speech recognition, install the CUDA 12 libraries expected by CTranslate2, then launch through `run.fish` so their library paths are set:

```fish
python -m pip install -r requirements-nvidia.txt
fish run.fish
```

The CUDA libraries are installed inside `.venv`; the system CUDA installation is left alone. If CUDA initialization still fails, the app reports the reason and retries on CPU.

### Niri application menu

Install a launcher for the current user. The script finds the project directory automatically and writes the desktop entry to the XDG applications folder:

```fish
fish install.fish
```

The launcher will be available to desktop launchers that read XDG application entries, such as Fuzzel or Wofi under Niri. Tk uses the system's available X11/Wayland compatibility layer.

### Other platforms

```sh
python -m venv .venv
source .venv/bin/activate  # POSIX sh/bash/zsh; Fish: source .venv/bin/activate.fish
python -m pip install -r requirements.txt
python app.py
```

Choose a video with the file picker or drag it into the app window. Select an output folder, censoring mode, and padding, then press **Analyze video**. Review the detected words and uncheck false positives before pressing **Render**. A custom sound file is required for Custom Sound mode. In `blacklist.txt` or an imported list, plain words match whole tokens, `word*` matches tokens beginning with that form, and `*word*` explicitly searches inside a token. You can load another list or add entries in the app.

Drag-and-drop is registered on the whole interface, including tabs and input fields. The app accepts normal file paths and `file://` paths from Linux file managers. If an existing process is still open, close it and start `fish run.fish` again after updating the application.

## Notes

- Detection uses word-level timestamps and exact token matching by default. Prefix and substring matching only happen when a list entry includes an explicit `*` mask.
- Bleep, Mute, and Custom Sound preserve the original video stream where possible. Fast-Forward speeds the flagged interval up 3x; Cut removes it. These editing modes re-encode video and audio for clean joins.
- Censor renders preserve the original audio level outside the selected intervals: no automatic loudness normalization, compression, or gain adjustment is applied. Bleep/custom-sound mixing does not attenuate the source according to the number of censored intervals. Studio exports also preserve source levels. Audio is re-encoded, so small codec differences are possible; replacement sounds have their own levels.
- Output is MP4. Audio-only censor presets copy compatible video without re-encoding. If the source codec cannot be stored in MP4 (for example, DNxHD/DNxHR), the renderer automatically retries with H.264 (`libx264`, CRF 18). Cut/Fast-Forward try NVENC when available and fall back to `libx264`.
- The application does not upload media. Model files are downloaded from the model registry on first use.

## Stream downloads and censor presets

The downloader uses a separate process so Stop also works during connection,
extraction, or FFmpeg processing. Missing output folders are created automatically.
Warnings are displayed, and the final path comes from yt-dlp after merging rather
than guessing from filenames. Partial downloads remain on disk; resuming depends
on the source protocol and the availability of the original stream fragments.
Live streams start at the current point by default. Enable the YouTube checkbox
only if you want to try downloading from the beginning (the broadcaster must allow it).
For recordings requiring your account, select an exported Netscape cookies.txt file.
Cookie files contain account credentials: keep them local and out of Git.

Install or update with `python -m pip install --upgrade -r requirements.txt`.
Full YouTube support needs the bundled yt-dlp EJS dependency and an installed
JavaScript runtime: Deno (recommended), Node.js, or Bun. The app selects an installed
runtime automatically. See https://github.com/yt-dlp/yt-dlp/wiki/EJS.

Additional censor presets: Low Bleep (400 Hz), Double Bleep (pulsed 1 kHz),
Noise (pink noise), Robot (modulated tone), Blur + Mute, Pixelate + Mute,
and Blackout + Mute. Visual presets cover the entire frame during the marked
intervals, mute speech, and re-encode video. All presets use the selected padding
and merge overlapping intervals. The original five presets remain available.

Run integration checks with `python -m unittest discover -s tests -v` in the
installed virtual environment. They render all twelve presets with FFmpeg,
download a local test video, and stop a stalled connection.

## Automatic reaction Shorts

Open **Авто Shorts**, choose the source and output folder, and click
**Создать Shorts автоматически**. The app transcribes the source locally,
ranks speech windows with audio peaks, and exports up to the requested number
of non-overlapping clips (10–60 seconds, default 35 seconds / 5 clips).
It prefers speech boundaries; the selected length is a target, with a maximum
of 60 seconds. A shorter source may produce a shorter clip, and there may be
fewer usable clips than requested. Automatic ranking is a review aid, not a
semantic guarantee that the selected moments are funny or interesting.

The reference-style layout is 1080×1920: streamer camera in the top 672 pixels,
main content in the lower 1248, and a persistent uppercase yellow title
with a black outline near the bottom of the camera panel. There are no speech
subtitles in this layout. Leave **Название** blank to build a title from the entire clip: complete
sentences and connected setup/payoff phrases are ranked by recurring topics,
actions, reactions, and reveals. Greetings are filtered out; sentences are not
cut to their first eight words. Titles use the spoken text rather than invented
events. The default **Стиль → P52** profile adapts the event-focused, uppercase
structure studied from [Drake Rofls Shorts](https://www.youtube.com/@DrakeRofls/shorts):
`P52 + action/event`, `P52 ПРО + topic`, or `P52: + question/context`.
First-person statements are changed to third person only for supported verb forms;
other people's names remain intact. Emotion/action claims require evidence in the
clip text. No Drake branding or hashtags are added. Choose **По фрагменту** to
keep the previous contextual quote style. See [P52 title style](docs/title-style-p52.md).
You can also enter your own title (up to 96 characters). The title
is visible throughout the clip and included in its filename. Longer titles
use a slightly smaller font and wrap across lines without dropping the ending. In automatic mode,
the frame preview shows a sample title; the actual title is chosen after analysis.
Audio retains the original level with no loudness
normalization or compression. Every run creates a new `shorts-*` subfolder.

**Камера** accepts `Авто` (local YuNet neural face detection) or `x,y,width,height` as
percentages of the source frame, for example `0,0,30,35` for a camera at top left.
**Контент** uses the same coordinates; `0,0,100,100` starts with the full frame
and center-crops it into the lower panel. For a reaction recorded as side-by-side
video, specify the content region separately to exclude the webcam.
**Проверить кадр** saves and opens a preview at the entered source time.
The bundled 227 KiB YuNet model works offline. Detection checks 12 frames,
including enlarged corners to locate small webcams, and prefers stable faces
near frame edges. Auto detection can confuse a face in the watched content with the streamer;
use the preview and manual regions to correct it. An undetected camera produces
an actionable error instead of silently duplicating the full frame.

Install updated dependencies with `python -m pip install -r requirements.txt`.
`opencv-python-headless` adds local face detection without a separate desktop UI.
The existing Studio and its export layouts remain available independently.
