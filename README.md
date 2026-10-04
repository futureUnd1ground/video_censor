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
- Rendered audio is loudness-normalized to `-16 LUFS` with a `-1.5 dBTP` peak target. Bleep/custom-sound mixing does not attenuate the source according to the number of censored intervals. This makes quiet source recordings easier to hear while keeping peaks controlled.
- Output is MP4. The renderer tries NVENC when available and falls back to `libx264`.
- The application does not upload media. Model files are downloaded from the model registry on first use.
