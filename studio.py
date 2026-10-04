"""Project, transcript, highlight and export helpers for Cut-Helper."""

from __future__ import annotations

import array
import json
import math
import re
import statistics
import subprocess
import tempfile
import wave
import shutil
from collections import Counter
from pathlib import Path
from typing import Callable

from engine import CensorError, require_tools


def _subtitle_font() -> str:
    """Use the installed compact pixel font, with an FFmpeg-safe fallback."""
    try:
        result = subprocess.run(["fc-match", "Pixeloid Sans", "-f", "%{family}"],
                                capture_output=True, text=True, timeout=2)
        family = result.stdout.strip()
        if family:
            return family.replace("'", "")
    except (OSError, subprocess.SubprocessError):
        pass
    return "Arial"


def transcribe_video(source: Path, model_name: str = "base",
                     progress: Callable[[str], None] | None = None) -> tuple[list[dict], str]:
    """Return segment-level transcript and detected language using faster-whisper."""
    ffmpeg, _ = require_tools()
    with tempfile.TemporaryDirectory(prefix="cut-helper-transcript-") as temp_dir:
        wav_path = Path(temp_dir) / "audio.wav"
        result = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
                                 "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav_path)],
                                capture_output=True, text=True)
        if result.returncode:
            raise CensorError("Не удалось извлечь аудио: " + result.stderr.strip())
        try:
            from faster_whisper import WhisperModel
            import ctranslate2
        except ImportError as exc:
            raise CensorError("Не установлены зависимости faster-whisper/ctranslate2.") from exc
        try:
            device = "cuda" if ctranslate2.get_cuda_device_count() else "cpu"
        except Exception:
            device = "cpu"
        if progress:
            progress("Транскрибация полной расшифровки на " + ("NVIDIA GPU" if device == "cuda" else "CPU"))
        try:
            model = WhisperModel(model_name, device=device,
                                 compute_type="float16" if device == "cuda" else "int8")
            chunks, info = model.transcribe(
                str(wav_path), language=None, word_timestamps=True, vad_filter=True,
                beam_size=5, best_of=5, temperature=0.0, condition_on_previous_text=True,
                vad_parameters={"min_silence_duration_ms": 450, "speech_pad_ms": 180},
                no_speech_threshold=0.5, log_prob_threshold=-1.0,
                compression_ratio_threshold=2.4, chunk_length=30)
            segments = []
            for chunk in chunks:
                segments.append({"start": float(chunk.start), "end": float(chunk.end),
                                 "text": chunk.text.strip(),
                                 "words": [{"word": word.word.strip(), "start": float(word.start),
                                            "end": float(word.end)} for word in (chunk.words or [])]})
            return segments, str(info.language or "")
        except Exception as exc:
            if device != "cuda":
                raise CensorError(f"Ошибка расшифровки: {exc}") from exc
            if progress:
                progress("GPU не отвечает, продолжаю полную расшифровку на CPU")
            try:
                model = WhisperModel(model_name, device="cpu", compute_type="int8")
                chunks, info = model.transcribe(
                    str(wav_path), language=None, word_timestamps=True, vad_filter=True,
                    beam_size=5, best_of=5, temperature=0.0, condition_on_previous_text=True,
                    vad_parameters={"min_silence_duration_ms": 450, "speech_pad_ms": 180},
                    no_speech_threshold=0.5, log_prob_threshold=-1.0,
                    compression_ratio_threshold=2.4, chunk_length=30)
                segments = [{"start": float(chunk.start), "end": float(chunk.end),
                             "text": chunk.text.strip(),
                             "words": [{"word": word.word.strip(), "start": float(word.start),
                                        "end": float(word.end)} for word in (chunk.words or [])]}
                            for chunk in chunks]
                return segments, str(info.language or "")
            except Exception as cpu_exc:
                raise CensorError(f"Ошибка расшифровки на GPU и CPU: {cpu_exc}") from cpu_exc


def format_srt_time(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{whole:02},{milliseconds:03}"


def transcript_to_srt(segments: list[dict], censored_words: set[str] | None = None) -> str:
    censored = {word.casefold() for word in (censored_words or set())}
    blocks = []
    for index, segment in enumerate(segments, 1):
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        if censored:
            text = re.sub(r"\b[\w-]+\b", lambda match: "█" * len(match.group(0))
                          if match.group(0).casefold() in censored else match.group(0), text)
        start = float(segment["start"])
        end = max(start + 0.05, float(segment["end"]))
        blocks.append(f"{index}\n{format_srt_time(start)} --> {format_srt_time(end)}\n{text}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def parse_chat_log(path: Path) -> list[dict]:
    """Read common Twitch text exports or timestamped JSON chat messages."""
    if path.suffix.casefold() == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CensorError(f"Не удалось прочитать чат: {exc}") from exc
        rows = payload if isinstance(payload, list) else payload.get("messages", [])
        messages = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            time_value = row.get("timestamp", row.get("time", row.get("offset")))
            try:
                seconds = float(time_value)
            except (TypeError, ValueError):
                continue
            body = row.get("message", row.get("text", ""))
            if isinstance(body, dict):
                body = body.get("text", "")
            if isinstance(body, list):
                body = "".join(str(part.get("text", "")) if isinstance(part, dict) else str(part)
                               for part in body)
            if str(body).strip():
                messages.append({"time": seconds, "user": str(row.get("author", row.get("user", ""))),
                                 "text": str(body).strip()})
        return messages

    pattern = re.compile(r"^\[?(?P<time>\d{1,2}:\d{2}:\d{2}(?:[.,]\d+)?)\]?\s+(?P<user>[^:]{1,64}):\s*(?P<text>.+)$")
    messages = []
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        hours, minutes, seconds = match.group("time").replace(",", ".").split(":")
        total = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        messages.append({"time": total, "user": match.group("user").strip(),
                         "text": match.group("text").strip()})
    if not messages:
        raise CensorError("В файле не найдены сообщения с таймкодами HH:MM:SS user: message.")
    return messages


def chat_activity(messages: list[dict], window: float = 10.0) -> list[dict]:
    counts = Counter(int(float(message["time"]) // window) for message in messages)
    if not counts:
        return []
    values = list(counts.values())
    baseline = statistics.median(values)
    deviation = statistics.median(abs(value - baseline) for value in values) or 1.0
    moments = []
    for bucket, count in counts.items():
        score = (count - baseline) / deviation
        if count >= max(4, baseline * 2) and score >= 2.0:
            samples = [message["text"] for message in messages
                       if int(float(message["time"]) // window) == bucket]
            moments.append({"start": bucket * window, "end": (bucket + 1) * window,
                            "score": min(1.0, 0.45 + score * 0.08), "source": "chat",
                            "label": f"Всплеск чата · {count} сообщений",
                            "chat_sample": samples[:4]})
    return moments


def audio_peaks(source: Path, step: float = 5.0,
                progress: Callable[[str], None] | None = None) -> list[dict]:
    ffmpeg, _ = require_tools()
    with tempfile.TemporaryDirectory(prefix="cut-helper-audio-") as temp_dir:
        wav_path = Path(temp_dir) / "levels.wav"
        result = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
                                 "-vn", "-ac", "1", "-ar", "8000", "-c:a", "pcm_s16le", str(wav_path)],
                                capture_output=True, text=True)
        if result.returncode:
            raise CensorError("Не удалось измерить громкость: " + result.stderr.strip())
        levels = []
        with wave.open(str(wav_path), "rb") as audio:
            rate = audio.getframerate()
            samples_per_bin = int(rate * step)
            cursor = 0
            while data := audio.readframes(samples_per_bin):
                raw = array.array("h")
                raw.frombytes(data)
                if not raw:
                    break
                square_mean = sum(sample * sample for sample in raw) / len(raw)
                rms = math.sqrt(square_mean) / 32768.0
                levels.append({"start": cursor / rate, "end": (cursor + len(raw)) / rate, "rms": rms})
                cursor += len(raw)
                if progress and len(levels) % 100 == 0:
                    progress(f"Анализ звука: {cursor / rate:.0f} с")
    if len(levels) < 3:
        return []
    values = [entry["rms"] for entry in levels]
    baseline = statistics.median(values)
    deviation = statistics.median(abs(value - baseline) for value in values) or 0.01
    moments = []
    for index, entry in enumerate(levels):
        local = values[max(0, index - 3):min(len(values), index + 4)]
        local_floor = statistics.median(local)
        if entry["rms"] < max(0.035, baseline + 2.5 * deviation) or entry["rms"] < local_floor * 1.8:
            continue
        moments.append({"start": max(0, entry["start"] - 12), "end": entry["end"] + 18,
                        "score": min(1.0, 0.45 + (entry["rms"] - baseline) / (deviation * 12)),
                        "source": "audio", "label": f"Всплеск громкости · {entry['rms']:.2f}"})
    moments.sort(key=lambda item: item["start"])
    merged = []
    for moment in moments:
        if merged and moment["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], moment["end"])
            merged[-1]["score"] = max(merged[-1]["score"], moment["score"])
        else:
            merged.append(moment)
    return merged


def visual_highlights(source: Path, transcript: list[dict] | None = None,
                      progress: Callable[[str], None] | None = None) -> list[dict]:
    """Scan sampled stream frames for on-screen chat and scene text.

    OCR is deliberately conservative: a right-side block with several text lines
    is treated as chat, while nearby transcript text gives the reviewer context.
    """
    ffmpeg, ffprobe = require_tools()
    tesseract = shutil.which("tesseract")
    if not tesseract:
        raise CensorError("Для поиска чата на кадре нужен tesseract.")
    meta = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format=duration:stream=duration",
                           "-of", "default=nw=1:nk=1", str(source)], capture_output=True, text=True)
    try:
        duration = next(float(line) for line in meta.stdout.splitlines() if line.strip() not in ("N/A", ""))
    except (StopIteration, ValueError) as exc:
        raise CensorError("Не удалось определить длительность видео.") from exc
    with tempfile.TemporaryDirectory(prefix="cut-helper-vision-") as temp_dir:
        pattern = str(Path(temp_dir) / "frame-%05d.png")
        result = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
                                 "-vf", "fps=1/8,scale=960:-2", "-frames:v", "300", pattern],
                                capture_output=True, text=True)
        if result.returncode:
            raise CensorError("Не удалось получить кадры для визуального анализа: " + result.stderr.strip())
        moments = []
        chat_boxes = []
        video_meta = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0",
                                     "-show_entries", "stream=width,height", "-of", "json", str(source)],
                                    capture_output=True, text=True)
        try:
            stream = json.loads(video_meta.stdout)["streams"][0]
            source_width, source_height = int(stream["width"]), int(stream["height"])
        except (ValueError, KeyError, IndexError, TypeError):
            source_width, source_height = 0, 0
        for index, frame in enumerate(sorted(Path(temp_dir).glob("frame-*.png"))):
            frame_time = index * 8.0
            if progress:
                progress(f"Просматриваю кадры: {frame_time:.0f} / {duration:.0f} с")
            ocr = subprocess.run([tesseract, str(frame), "stdout", "-l", "rus+eng", "--psm", "6", "tsv"],
                                 capture_output=True, text=True)
            rows = []
            for line in ocr.stdout.splitlines()[1:]:
                fields = line.split("\t")
                if len(fields) < 12 or not fields[-1].strip():
                    continue
                try:
                    left, top, width, height = (int(value) for value in fields[6:10])
                    confidence = int(float(fields[10]))
                except (ValueError, TypeError):
                    continue
                rows.append((left, top, width, height, fields[-1].strip(), confidence))
            if not rows:
                continue
            image_width = 960
            right_words = [row for row in rows if row[0] >= image_width * 0.52 and row[5] >= 25]
            lines = {row[1] // 12 for row in right_words}
            if len(right_words) < 6 or len(lines) < 3:
                continue
            text = " ".join(row[4] for row in right_words)
            chat_boxes.extend((row[0], row[1], row[2], row[3]) for row in right_words)
            context = ""
            for segment in transcript or []:
                if float(segment.get("end", 0)) >= frame_time - 8 and float(segment.get("start", 0)) <= frame_time + 8:
                    context = segment.get("text", "").strip()
                    if context:
                        break
            moments.append({"start": max(0, frame_time - 8), "end": min(duration, frame_time + 12),
                            "score": min(1.0, 0.55 + min(len(lines), 8) * 0.04),
                            "source": "visual", "label": "На кадре найден чат",
                            "description": (f"Текст справа: {text[:180]}" +
                                            (f" · Речь: {context[:180]}" if context else ""))})
    merged = []
    for moment in moments:
        if merged and moment["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], moment["end"])
            merged[-1]["description"] = moment.get("description", merged[-1].get("description", ""))
        else:
            merged.append(moment)
    if merged and chat_boxes and source_width and source_height:
        scaled_height = source_height * 960 / source_width
        left = max(0, min(box[0] for box in chat_boxes) - 12)
        top = max(0, min(box[1] for box in chat_boxes) - 12)
        right = min(960, max(box[0] + box[2] for box in chat_boxes) + 16)
        bottom = min(round(scaled_height), max(box[1] + box[3] for box in chat_boxes) + 16)
        chat_box = (left / 960, top / scaled_height, (right - left) / 960, (bottom - top) / scaled_height)
        for moment in merged:
            moment["chat_box"] = chat_box
    return merged


def write_project(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_project(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CensorError(f"Не удалось открыть проект: {exc}") from exc
    if not isinstance(data, dict) or data.get("format") != "cut-helper-project-v1":
        raise CensorError("Это не файл проекта Cut-Helper.")
    return data


def _escape_filter_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def _face_crop(source: Path, output_height: int = 1920) -> str | None:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    ffmpeg, ffprobe = require_tools()
    meta = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0",
                           "-show_entries", "stream=width,height,duration", "-of", "json", str(source)],
                          capture_output=True, text=True)
    try:
        stream = json.loads(meta.stdout)["streams"][0]
        width, height = int(stream["width"]), int(stream["height"])
        duration = float(stream.get("duration", 0))
    except (ValueError, KeyError, IndexError, TypeError):
        return None
    cap = cv2.VideoCapture(str(source))
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    centers = []
    for second in range(0, max(1, min(int(duration), 180)), 3):
        cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=5, minSize=(32, 32))
        if len(faces):
            x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
            centers.append((x + w / 2) / width)
    cap.release()
    if not centers:
        return None
    crop_width = min(width, int(height * 9 / 16))
    center = float(np.median(centers)) * width
    x = round(max(0, min(width - crop_width, center - crop_width / 2)))
    return f"crop={crop_width}:{height}:{x}:0,scale=1080:{output_height}"


def _chat_srt(messages: list[dict], start: float, end: float) -> str:
    blocks = []
    for message in messages:
        time = float(message.get("time", -1))
        if not start <= time < end:
            continue
        text = f"{message.get('user', '')}: {message.get('text', '')}".strip(": ")
        block_start = time - start
        blocks.append(f"{len(blocks) + 1}\n{format_srt_time(block_start)} --> "
                      f"{format_srt_time(min(end - start, block_start + 5))}\n{text}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def render_moment(source: Path, moment: dict, destination: Path, profile: str,
                  transcript: list[dict] | None = None, chat: list[dict] | None = None,
                  subtitles: bool = False, chat_overlay: bool = False, watermark: str = "",
                  censored_words: set[str] | None = None, layout: str = "Обычный Shorts",
                  progress: Callable[[str], None] | None = None) -> Path:
    ffmpeg, _ = require_tools()
    start, end = float(moment["start"]), float(moment["end"])
    if start < 0 or end <= start:
        raise CensorError("У хайлайта некорректные границы времени.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if progress:
        progress(f"Экспорт: {destination.name}")
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.3f}",
               "-i", str(source)]
    temp_context = tempfile.TemporaryDirectory(prefix="cut-helper-export-")
    temp_dir = Path(temp_context.name)
    filters = []
    if profile == "YouTube 16:9":
        filters.append("scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2")
    elif profile in ("Shorts 9:16", "TikTok 9:16"):
        crop = _face_crop(source)
        if layout == "Стример сверху + чат снизу":
            filters.append(crop or "scale=1080:1500:force_original_aspect_ratio=increase,crop=1080:1500")
            filters.append("pad=1080:1920:0:0:black")
        elif layout == "Игра сверху + чат снизу":
            filters.append("scale=1080:1500:force_original_aspect_ratio=increase,crop=1080:1500")
            filters.append("pad=1080:1920:0:0:black")
        else:
            filters.append(crop or "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920")
    elif profile == "Square 1:1":
        filters.append("scale=1080:1080:force_original_aspect_ratio=increase,crop=1080:1080")
    elif profile == "Audio MP3":
        command.extend(["-t", f"{end - start:.3f}", "-map", "0:a:0?", "-vn", "-c:a", "libmp3lame", "-q:a", "2", str(destination)])
        result = subprocess.run(command, capture_output=True, text=True)
        temp_context.cleanup()
        if result.returncode:
            raise CensorError("Ошибка экспорта аудио: " + result.stderr.strip())
        return destination
    elif profile != "Original 16:9":
        temp_context.cleanup()
        raise CensorError(f"Неизвестный профиль экспорта: {profile}")

    end_time = end - start
    font = _subtitle_font()
    if subtitles and transcript:
        subtitle_file = temp_dir / "captions.srt"
        relevant = [{**segment, "start": max(0, float(segment["start"]) - start),
                     "end": min(end_time, float(segment["end"]) - start)}
                    for segment in transcript if float(segment["end"]) > start and float(segment["start"]) < end]
        subtitle_file.write_text(transcript_to_srt(relevant, censored_words), encoding="utf-8")
        if relevant:
            subtitle_size = 16 if profile in ("Shorts 9:16", "TikTok 9:16") else 22
            subtitle_margin = 86 if layout != "Обычный Shorts" else 64
            filters.append(f"subtitles=filename='{_escape_filter_path(subtitle_file)}':force_style='FontName={font},FontSize={subtitle_size},Outline=2,Shadow=0,Alignment=2,MarginV={subtitle_margin}'")
    dedicated_chat_panel = layout in ("Стример сверху + чат снизу", "Игра сверху + чат снизу")
    if (chat_overlay or dedicated_chat_panel) and chat:
        chat_file = temp_dir / "chat.srt"
        chat_file.write_text(_chat_srt(chat, start, end), encoding="utf-8")
        if chat_file.stat().st_size:
            if dedicated_chat_panel and profile in ("Shorts 9:16", "TikTok 9:16"):
                chat_style = f"FontName={font},FontSize=16,Outline=1,Alignment=2,MarginV=18"
            else:
                chat_style = f"FontName={font},FontSize=14,Outline=2,Alignment=7,MarginL=34,MarginV=70"
            filters.append(f"subtitles=filename='{_escape_filter_path(chat_file)}':force_style='{chat_style}'")
    if watermark.strip():
        safe_text = watermark.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'").replace("%", "\\%")
        filters.append(f"drawtext=text='{safe_text}':x=w-tw-30:y=h-th-30:fontsize=28:fontcolor=white:borderw=2:bordercolor=black")
    command.extend(["-t", f"{end_time:.3f}", "-map", "0:v:0", "-map", "0:a:0?"])
    if filters:
        command.extend(["-vf", ",".join(filters)])
    command.extend(["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(destination)])
    result = subprocess.run(command, capture_output=True, text=True)
    temp_context.cleanup()
    if result.returncode:
        destination.unlink(missing_ok=True)
        raise CensorError("Ошибка экспорта: " + "\n".join(result.stderr.strip().splitlines()[-10:]))
    return destination
