"""Isolated yt-dlp worker; cancellable even while extraction or FFmpeg is running."""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time


def runtime_options():
    # EJS comes from yt-dlp[default]. Enable an installed runtime explicitly.
    for runtime in ("deno", "node", "bun"):
        path = shutil.which(runtime)
        if path:
            return {"js_runtimes": {runtime: {"path": path}}}
    return {}


def download_stream(url, output_dir, stop, emit, *, from_start=False, cookies=""):
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not shutil.which("ffmpeg"):
        raise RuntimeError("Для скачивания и объединения потоков установите FFmpeg.")
    if stop.is_set():
        emit(("stream_stopped", ""))
        return
    request = dict(url=url, output_dir=str(output_dir), from_start=from_start,
                   cookies=str(Path(cookies).expanduser()) if cookies else "")
    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker"],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True,
                               start_new_session=(os.name != "nt"))
    process.stdin.write(json.dumps(request) + "\n")
    process.stdin.close()
    lines = queue.Queue()
    def read_lines():
        for line in process.stdout:
            lines.put(line)
        lines.put(None)
    reader = threading.Thread(target=read_lines, daemon=True)
    reader.start()
    stopped_at = None
    done = False
    error = ""
    eof = False
    try:
        while not eof or process.poll() is None:
            if stop.is_set():
                if stopped_at is None:
                    stopped_at = time.monotonic()
                    if os.name == "nt":
                        process.terminate()
                    else:
                        try:
                            os.killpg(process.pid, signal.SIGINT)
                        except ProcessLookupError:
                            pass
                elif process.poll() is None and time.monotonic() - stopped_at > 5:
                    if os.name == "nt":
                        process.kill()
                    else:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
            try:
                line = lines.get(timeout=0.1)
            except queue.Empty:
                continue
            if line is None:
                eof = True
                continue
            if not line.startswith("CUT_EVENT "):
                continue
            event, payload = json.loads(line[len("CUT_EVENT "):])
            if event == "stream_error":
                error = payload
            elif event == "stream_done":
                done = True
                if not stop.is_set():
                    emit((event, payload))
            elif not stop.is_set():
                emit((event, payload))
        code = process.wait()
        if stop.is_set():
            emit(("stream_stopped", ""))
        elif code or not done:
            raise RuntimeError(error or f"Загрузчик завершился с кодом {code}. Проверьте ссылку и доступ к сайту.")
    finally:
        if process.poll() is None:
            if os.name == "nt":
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        reader.join(timeout=2)
        process.stdout.close()


def _worker(request):
    from yt_dlp import YoutubeDL
    def emit(event, payload):
        print("CUT_EVENT " + json.dumps((event, payload), ensure_ascii=False), flush=True)
    class Logger:
        def debug(self, text):
            if not text.startswith("[debug]"):
                emit("stream_status", text)
        def warning(self, text):
            emit("stream_status", "Предупреждение: " + text)
        def error(self, text):
            emit("stream_status", text)
    def progress(data):
        if data.get("status") == "downloading":
            total = data.get("total_bytes") or data.get("total_bytes_estimate")
            downloaded = data.get("downloaded_bytes") or 0
            emit("stream_progress", (downloaded / total if total else None,
                                     data.get("speed"), data.get("eta"), downloaded))
        elif data.get("status") == "finished":
            emit("stream_status", "Поток получен; обработка файла…")
    finished = []
    def postprocess(data):
        if data.get("status") == "finished" and data.get("postprocessor") == "MoveFiles":
            finished.append(data["info_dict"]["filepath"])
    options = {
        **runtime_options(), "format": "bv*+ba/b", "noplaylist": True,
        "outtmpl": str(Path(request["output_dir"]) / "%(title).150B [%(id)s].%(ext)s"),
        "merge_output_format": "mp4/mkv", "continuedl": True,
        "live_from_start": bool(request["from_start"]),
        "socket_timeout": 20, "retries": 5, "fragment_retries": 5,
        "skip_unavailable_fragments": False,
        "progress_hooks": [progress], "postprocessor_hooks": [postprocess],
        "logger": Logger(), "quiet": True,
    }
    if request.get("cookies"):
        options["cookiefile"] = request["cookies"]
    with YoutubeDL(options) as downloader:
        info = downloader.extract_info(request["url"], download=True)
    if not finished or not Path(finished[-1]).is_file():
        raise RuntimeError("Итоговый файл не найден после обработки потоков.")
    emit("stream_done", (info.get("title", "Стрим"), finished[-1]))


if __name__ == "__main__" and "--worker" in sys.argv:
    try:
        _worker(json.loads(sys.stdin.readline()))
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        print("CUT_EVENT " + json.dumps(("stream_error", str(exc)), ensure_ascii=False), flush=True)
        sys.exit(1)
