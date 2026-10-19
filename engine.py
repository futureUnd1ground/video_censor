"""Speech detection and FFmpeg rendering for Video Censor."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import threading
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


CENSOR_MODES = ("Bleep", "Low Bleep", "Double Bleep", "Noise", "Robot", "Mute",
                "Blur + Mute", "Pixelate + Mute", "Blackout + Mute", "Fast-Forward", "Cut", "Custom Sound")
AUDIO_MODES = ("Bleep", "Low Bleep", "Double Bleep", "Noise", "Robot", "Mute", "Custom Sound")
VISUAL_MODES = ("Blur + Mute", "Pixelate + Mute", "Blackout + Mute")


class CensorError(RuntimeError):
    pass


@dataclass
class Hit:
    word: str
    start: float
    end: float
    enabled: bool = True


def require_tools() -> tuple[str, str]:
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise CensorError("Не найдены ffmpeg и/или ffprobe. Установите FFmpeg и добавьте его в PATH.")
    return ffmpeg, ffprobe


def probe_media(path: Path) -> tuple[float, bool]:
    ffmpeg, ffprobe = require_tools()
    del ffmpeg
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-show_entries",
         "stream=codec_type", "-of", "json", str(path)], capture_output=True, text=True,
    )
    if result.returncode:
        raise CensorError(f"Не удалось прочитать видео: {result.stderr.strip()}")
    try:
        data = json.loads(result.stdout)
        duration = float(data["format"]["duration"])
        has_audio = any(stream.get("codec_type") == "audio" for stream in data.get("streams", []))
    except (ValueError, KeyError, TypeError) as exc:
        raise CensorError("Файл не содержит корректного видео/аудио потока.") from exc
    if duration <= 0:
        raise CensorError("Длительность видео не определена.")
    return duration, has_audio


def load_terms(paths: Iterable[Path]) -> list[str]:
    terms: set[str] = set()
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            term = line.split("#", 1)[0].strip().casefold()
            if term:
                leading_wildcard = term.startswith("*")
                trailing_wildcard = term.endswith("*")
                body = _normalize_word(term.strip("*"))
                if body:
                    terms.add(("*" if leading_wildcard else "") + body + ("*" if trailing_wildcard else ""))
    return sorted(term for term in terms if term)


def _normalize_word(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().replace("ё", "е")
    return re.sub(r"[^\w]+", "", value)


def _matches_profanity(word: str, terms: list[str]) -> bool:
    for pattern in terms:
        prefix = pattern.endswith("*")
        suffix = pattern.startswith("*")
        term = pattern.strip("*")
        if prefix and suffix and term in word:
            return True
        if prefix and not suffix and word.startswith(term):
            return True
        if suffix and not prefix and word.endswith(term):
            return True
        if not prefix and not suffix and word == term:
            return True
    return False


def detect(path: Path, terms: list[str], model_name: str = "base",
           progress: Callable[[str], None] | None = None,
           on_hit: Callable[[Hit], None] | None = None,
           on_transcription_progress: Callable[[float, float], None] | None = None) -> list[Hit]:
    ffmpeg, _ = require_tools()
    if not terms:
        return []
    if progress:
        progress("Извлечение аудио")
    with tempfile.TemporaryDirectory(prefix="video-censor-") as temp_dir:
        wav = Path(temp_dir) / "audio.wav"
        extraction = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(path),
             "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)],
            capture_output=True, text=True,
        )
        if extraction.returncode:
            raise CensorError(f"Не удалось извлечь аудио: {extraction.stderr.strip()}")
        if progress:
            progress("Транскрибация (загрузка модели при первом запуске может занять время)")
        try:
            from faster_whisper import WhisperModel
            import ctranslate2
        except ImportError as exc:
            raise CensorError("Не установлены зависимости. Выполните: python -m pip install -r requirements.txt") from exc
        try:
            device = "cuda" if ctranslate2.get_cuda_device_count() else "cpu"
        except Exception:
            device = "cpu"
        if progress:
            progress("Транскрибация на NVIDIA GPU (CUDA/float16)" if device == "cuda"
                     else "Транскрибация на CPU (int8)")

        def transcribe_on(target_device: str) -> list[Hit]:
            compute_type = "float16" if target_device == "cuda" else "int8"
            model = WhisperModel(model_name, device=target_device, compute_type=compute_type)
            segments, info = model.transcribe(
                str(wav), language=None, word_timestamps=True, vad_filter=True,
                beam_size=5, best_of=5, temperature=0.0, condition_on_previous_text=True,
                vad_parameters={"min_silence_duration_ms": 450, "speech_pad_ms": 180},
                no_speech_threshold=0.5, log_prob_threshold=-1.0,
                compression_ratio_threshold=2.4, chunk_length=30
            )
            found: list[Hit] = []
            for segment in segments:
                for item in segment.words or []:
                    original = item.word.strip()
                    normalized = _normalize_word(original)
                    if normalized and _matches_profanity(normalized, terms):
                        hit = Hit(original, max(0.0, float(item.start)), float(item.end))
                        found.append(hit)
                        if on_hit:
                            on_hit(hit)
                if on_transcription_progress and info.duration:
                    on_transcription_progress(min(float(segment.end), float(info.duration)),
                                              float(info.duration))
            return found

        try:
            hits = transcribe_on(device)
        except Exception as exc:
            if device != "cuda":
                raise
            if progress:
                progress(f"GPU недоступна ({exc}). Повторяю распознавание на CPU…")
            try:
                hits = transcribe_on("cpu")
            except Exception as cpu_exc:
                raise CensorError(f"Не удалось распознать видео ни на GPU, ни на CPU: {cpu_exc}") from cpu_exc
        if progress:
            progress(f"Поиск таймкодов: найдено {len(hits)}")
        return hits


def _merge_ranges(hits: list[Hit], padding: float, duration: float) -> list[tuple[float, float]]:
    ranges = sorted((max(0.0, h.start - padding), min(duration, h.end + padding)) for h in hits if h.enabled)
    merged: list[list[float]] = []
    for start, end in ranges:
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def render(source: Path, destination: Path, hits: list[Hit], mode: str, padding_ms: int,
           custom_sound: Path | None = None, progress: Callable[[str], None] | None = None) -> None:
    ffmpeg, _ = require_tools()
    duration, has_audio = probe_media(source)
    ranges = _merge_ranges(hits, padding_ms / 1000.0, duration)
    if not ranges:
        raise CensorError("Нет отмеченных слов для цензурирования.")
    if mode == "Custom Sound" and (custom_sound is None or not custom_sound.is_file()):
        raise CensorError("Выберите файл звука для режима Custom Sound.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if progress:
        progress("Рендеринг видео")

    command = [ffmpeg, "-hide_banner", "-y", "-i", str(source)]
    graph: list[str] = []
    if mode in AUDIO_MODES or mode in VISUAL_MODES:
        if not has_audio:
            raise CensorError("В исходном видео отсутствует аудиодорожка.")
        conditions = "+".join(f"between(t,{start:.6f},{end:.6f})" for start, end in ranges)
        graph.append(f"[0:a]volume=0:enable='{conditions}'[muted]")
        if mode == "Mute" or mode in VISUAL_MODES:
            graph.append("[muted]anull[aout]")
        else:
            sounds: list[str] = []
            for index, (start, end) in enumerate(ranges):
                duration_s = max(0.03, end - start)
                if mode != "Custom Sound":
                    if mode == "Noise":
                        sound = f"anoisesrc=color=pink:amplitude=0.18:sample_rate=48000:duration={duration_s:.6f}"
                    elif mode == "Robot":
                        sound = f"aevalsrc=0.15*sin(2*PI*220*t)*sin(2*PI*30*t):s=48000:d={duration_s:.6f}"
                    else:
                        frequency = 400 if mode == "Low Bleep" else 1000
                        sound = f"sine=frequency={frequency}:sample_rate=48000:duration={duration_s:.6f},volume=0.35"
                        if mode == "Double Bleep":
                            sound += ",volume=0:enable='gte(mod(t,0.16),0.09)'"
                    graph.append(f"{sound},afade=t=in:d=0.005,afade=t=out:st={max(0, duration_s - 0.005):.6f}:d=0.005,adelay={round(start * 1000)}|{round(start * 1000)}[s{index}]")
                else:
                    command.extend(["-stream_loop", "-1", "-i", str(custom_sound)])
                    input_index = index + 1
                    graph.append(f"[{input_index}:a]atrim=duration={duration_s:.6f},asetpts=PTS-STARTPTS,volume=0.8,adelay={round(start * 1000)}|{round(start * 1000)}[s{index}]")
                sounds.append(f"[s{index}]")
            graph.append("[muted]" + "".join(sounds) +
                         f"amix=inputs={len(sounds) + 1}:duration=first:dropout_transition=0:normalize=0[aout]")
        video_map = "0:v:0"
        video_codec = ["-c:v", "copy"]
        if mode in VISUAL_MODES:
            if mode == "Blur + Mute":
                graph.append(f"[0:v]boxblur=12:2:enable='{conditions}'[vout]")
            elif mode == "Blackout + Mute":
                graph.append(f"[0:v]drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill:enable='{conditions}'[vout]")
            else:
                graph.append("[0:v]split[original][small]")
                graph.append("[small]scale=w='max(2,trunc(iw/24))':h='max(2,trunc(ih/24))'[pixels]")
                graph.append("[pixels][original]scale2ref=w=rw:h=rh:flags=neighbor[pixelated][base]")
                graph.append(f"[base][pixelated]overlay=enable='{conditions}'[vout]")
            video_map = "[vout]"
            video_codec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]
        command.extend(["-filter_complex", ";".join(graph), "-map", video_map, "-map", "[aout]", *video_codec, "-c:a", "aac", "-b:a", "192k"])
    elif mode in ("Cut", "Fast-Forward"):
        boundaries: list[tuple[float, float, bool]] = []
        cursor = 0.0
        for start, end in ranges:
            if start > cursor:
                boundaries.append((cursor, start, False))
            if mode == "Fast-Forward":
                boundaries.append((start, end, True))
            cursor = end
        if cursor < duration:
            boundaries.append((cursor, duration, False))
        if not boundaries:
            raise CensorError("После вырезки не осталось видео.")
        vlabels, alabels = [], []
        for index, (start, end, sped) in enumerate(boundaries):
            if end - start < 0.01:
                continue
            vlabel, alabel = f"v{index}", f"a{index}"
            rate = 3.0 if sped else 1.0
            graph.append(f"[0:v]trim=start={start:.6f}:end={end:.6f},setpts=(PTS-STARTPTS)/{rate:.1f}[{vlabel}]")
            if has_audio:
                audio_speed = "aresample=48000,asetrate=144000,aresample=48000" if sped else "anull"
                graph.append(f"[0:a]atrim=start={start:.6f}:end={end:.6f},asetpts=PTS-STARTPTS,{audio_speed}[{alabel}]")
            vlabels.append(f"[{vlabel}]")
            if has_audio:
                alabels.append(f"[{alabel}]")
        graph.append("".join(vlabels) + f"concat=n={len(vlabels)}:v=1:a=0[vout]")
        if has_audio:
            graph.append("".join(alabels) +
                         f"concat=n={len(alabels)}:v=0:a=1[aout]")
        command.extend(["-filter_complex", ";".join(graph), "-map", "[vout]"])
        if has_audio:
            command.extend(["-map", "[aout]"])
        encoders = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True)
        nvenc_available = "h264_nvenc" in encoders.stdout
        video_codec = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "21"] if nvenc_available else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
        command.extend(video_codec)
        if has_audio:
            command.extend(["-c:a", "aac", "-b:a", "192k"])
    else:
        raise CensorError(f"Неизвестный режим: {mode}")
    command.extend(["-movflags", "+faststart", str(destination)])
    expected_duration = duration
    if mode == "Cut":
        expected_duration -= sum(end - start for start, end in ranges)
    elif mode == "Fast-Forward":
        expected_duration -= sum((end - start) * (1 - 1 / 3) for start, end in ranges)
    expected_duration = max(0.1, expected_duration)

    def run_with_progress(args: list[str]):
        progress_args = args[:-1] + ["-progress", "pipe:1", "-nostats", args[-1]]
        process = subprocess.Popen(progress_args, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, bufsize=1)
        errors: list[str] = []
        def drain_errors():
            if process.stderr:
                for line in process.stderr:
                    errors.append(line)
                    if len(errors) > 100:
                        del errors[:50]
        reader = threading.Thread(target=drain_errors, daemon=True)
        reader.start()
        current_seconds = 0.0
        assert process.stdout is not None
        for line in process.stdout:
            key, separator, value = line.strip().partition("=")
            if separator and key in ("out_time_us", "out_time_ms"):
                try:
                    current_seconds = int(value) / 1_000_000
                except ValueError:
                    continue
                if progress:
                    ratio = max(0.0, min(1.0, current_seconds / expected_duration))
                    progress(f"Рендеринг видео: {round(ratio * 100)}%")
        code = process.wait()
        reader.join()
        process.stdout.close()
        if process.stderr:
            process.stderr.close()
        return code, "".join(errors)

    result_code, stderr = run_with_progress(command)
    if result_code and mode in ("Cut", "Fast-Forward") and nvenc_available:
        fallback = command.copy()
        encoder_index = fallback.index("h264_nvenc")
        fallback[encoder_index:encoder_index + 5] = ["libx264", "-preset", "veryfast", "-crf", "20"]
        if progress:
            progress("NVENC не сработал, продолжаю рендер на CPU…")
        result_code, stderr = run_with_progress(fallback)
    if result_code and "-c:v" in command and command[command.index("-c:v") + 1] == "copy" and (
        "could not find tag for codec" in stderr.casefold()
        or "codec not currently supported in container" in stderr.casefold()
    ):
        fallback = command.copy()
        codec_index = fallback.index("-c:v") + 1
        fallback[codec_index:codec_index + 1] = [
            "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"]
        if progress:
            progress("Исходный кодек несовместим с MP4; сохраняю видео в H.264…")
        result_code, stderr = run_with_progress(fallback)
    if result_code:
        details = stderr.strip().splitlines()[-12:]
        raise CensorError("Ошибка FFmpeg:\n" + "\n".join(details))
