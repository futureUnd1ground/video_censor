"""Local automatic reaction Shorts: moment selection, split framing and timed captions."""
from __future__ import annotations

import math
from pathlib import Path
import re
import statistics
import subprocess
import tempfile
import textwrap

from engine import CensorError, probe_media, require_tools
from studio import _escape_filter_path, audio_peaks, transcribe_video

def parse_region(value: str) -> tuple[float, float, float, float] | None:
    """A normalized crop, supplied as x,y,width,height in percent; Auto uses faces."""
    if value.strip().casefold() in ("авто", "auto", ""):
        return None
    try:
        x, y, width, height = [float(part.strip()) / 100 for part in value.split(',')]
    except (TypeError, ValueError) as exc:
        raise CensorError("Область: укажите x,y,ширину,высоту в процентах, например 0,0,30,35.") from exc
    if not all(math.isfinite(n) for n in (x, y, width, height)) or (
        min(x, y) < 0 or min(width, height) <= 0 or x + width > 1.000001 or y + height > 1.000001
    ):
        raise CensorError("Область должна целиком находиться в кадре (0–100%).")
    return x, y, width, height


def detect_camera(source: Path, start: float, end: float) -> tuple[float, float, float, float]:
    """Estimate a stable face region; manual crop remains available for overlays."""
    try:
        import cv2
    except ImportError as exc:
        raise CensorError("Для автопоиска камеры установите зависимости или задайте область камеры вручную.") from exc
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    capture = cv2.VideoCapture(str(source))
    groups = []
    try:
        for index in range(9):
            second = start + (end - start) * (index + .5) / 9
            capture.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
            ok, frame = capture.read()
            if not ok:
                continue
            height, width = frame.shape[:2]
            gray = cv2.cvtColor(cv2.resize(frame, (800, max(1, round(height * 800 / width)))), cv2.COLOR_BGR2GRAY)
            gh, gw = gray.shape
            seen = set()
            for x, y, w, h in detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)):
                face = ((x + w / 2) / gw, (y + h / 2) / gh, w / gw, h / gh)
                if face[1] > .9:  # Ignore avatar badges along the bottom edge.
                    continue
                group = next((g for g in groups if abs(g[0][0] - face[0]) < .09 and abs(g[0][1] - face[1]) < .09), None)
                if group is None:
                    group = []
                    groups.append(group)
                if id(group) not in seen:
                    group.append(face)
                    seen.add(id(group))
        if not groups:
            raise CensorError("Камера не найдена. Задайте её область вручную и нажмите «Проверить кадр».")
        # A webcam face is usually smaller than faces in the content being watched.
        group = max(groups, key=lambda g: (len(g), -statistics.median(f[2] * f[3] for f in g)))
        cx, cy, fw, fh = [statistics.median(face[i] for face in group) for i in range(4)]
        crop_w = min(1., max(fw * 3.6, fh * height / width * 3.0))
        crop_h = min(1., crop_w * width / height * 672 / 1080)
        crop_w = min(crop_w, crop_h * height / width * 1080 / 672)
        return max(0., min(1 - crop_w, cx - crop_w / 2)), max(0., min(1 - crop_h, cy - crop_h * .38)), crop_w, crop_h
    finally:
        capture.release()


def select_moments(transcript, duration, length=35, count=5, signals=()):
    if not 10 <= length <= 60 or not 1 <= count <= 20:
        raise CensorError("Длина Shorts: 10–60 секунд; количество: 1–20.")
    segments = sorted((s for s in transcript if str(s.get('text', '')).strip()
                       and float(s['end']) > float(s['start']) and float(s['start']) < duration),
                      key=lambda s: float(s['start']))
    if not segments:
        raise CensorError("Речь не найдена: не удалось выбрать моменты для Shorts.")
    candidates = []
    for segment in segments:
        start = max(0., float(segment['start']) - .15)
        limit = min(duration, start + 60, start + length * 1.25)
        endings = [float(s['end']) for s in segments
                   if start + length * .7 <= float(s['end']) <= limit]
        if not endings:
            # Only the final short source may be shorter than the requested length.
            if duration <= length and segment is segments[0]:
                end = duration
            else:
                continue
        else:
            end = min(endings, key=lambda t: abs(t - start - length))
        relevant = [s for s in segments if float(s['start']) < end and float(s['end']) > start]
        text = ' '.join(str(s['text']) for s in relevant)
        spoken = sum(max(0., min(end, float(s['end'])) - max(start, float(s['start']))) for s in relevant)
        score = min(1., spoken / max(1., end - start))
        score += min(.6, len(re.findall(r'[!?]|\b(?:смех|ахаха|неожиданно|смотри|wow|laugh)\b', text, re.I)) * .1)
        score += sum(float(s.get('score', .5)) for s in signals
                     if float(s['start']) < end and float(s['end']) > start)
        candidates.append(dict(start=start, end=min(duration, end), score=score,
                               source='auto-shorts', label=text[:85], enabled=True))
    selected = []
    for candidate in sorted(candidates, key=lambda s: (-s['score'], s['start'])):
        if all(candidate['end'] <= other['start'] or candidate['start'] >= other['end'] for other in selected):
            selected.append(candidate)
            if len(selected) >= count:
                break
    if not selected:
        raise CensorError("Нет законченных речевых фрагментов нужной длины. Уменьшите длину Shorts.")
    return sorted(selected, key=lambda s: s['start'])


def _ass_time(seconds):
    ticks = max(0, round(seconds * 100))
    hours, ticks = divmod(ticks, 360000)
    minutes, ticks = divmod(ticks, 6000)
    seconds, ticks = divmod(ticks, 100)
    return f'{hours}:{minutes:02}:{seconds:02}.{ticks:02}'


def short_title(transcript, start, end):
    """Choose a short quote from the clip without inventing a description."""
    phrases = []
    for segment in transcript:
        if float(segment['end']) <= start or float(segment['start']) >= end:
            continue
        words = segment.get('words')
        text = (' '.join(str(w.get('word', '')) for w in words
                         if float(w['end']) > start and float(w['start']) < end)
                if words else str(segment.get('text', '')))
        phrases.extend(phrase.strip() for phrase in re.split(r'[.!?…]+', text) if phrase.strip())
    if not phrases:
        return 'Момент из стрима'
    phrase = max(phrases, key=lambda text: (2 <= len(text.split()) <= 8, min(len(text.split()), 8), -len(text)))
    title = []
    for word in phrase.split():
        if len(' '.join(title + [word])) > 64:
            break
        title.append(word)
        if len(title) == 8:
            break
    return ' '.join(title) or phrase[:64]


def reaction_title(title, duration):
    """One persistent yellow title; no word-timed speech captions."""
    text = ' '.join(str(title).split()).replace('\\', '/').replace('{', '(').replace('}', ')').upper()
    if not text or len(text) > 72:
        raise CensorError('Название должно содержать от 1 до 72 символов.')
    text = r'\N'.join(textwrap.wrap(text, width=26, break_on_hyphens=False))
    header = '''[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Reaction,DejaVu Sans,58,&H0000FFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,1,2,50,50,1292,1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
'''
    return header + f"Dialogue: 0,0:00:00.00,{_ass_time(duration)},Reaction,,0,0,0,,{text}\n"


def _crop(region):
    x, y, width, height = region
    return (f"crop=w='max(2,trunc(iw*{width:.8f}/2)*2)':h='max(2,trunc(ih*{height:.8f}/2)*2)'"
            f":x='trunc(iw*{x:.8f}/2)*2':y='trunc(ih*{y:.8f}/2)*2'")


def render_reaction(source, moment, destination, transcript=(), camera='Авто', content='0,0,100,100',
                    progress=None, preview=False, title=None):
    source, destination = Path(source), Path(destination)
    duration, _ = probe_media(source)
    start, end = max(0., float(moment['start'])), min(duration, float(moment['end']))
    if not math.isfinite(start + end) or end <= start:
        raise CensorError('Некорректные границы Shorts.')
    camera_region = parse_region(camera)
    if camera_region is None:
        if progress:
            progress('Ищу камеру стримера…')
        camera_region = detect_camera(source, start, end)
    content_region = parse_region(content) or (0., 0., 1., 1.)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg, _ = require_tools()
    with tempfile.TemporaryDirectory(prefix='cut-helper-shorts-') as directory:
        graph = ["[0:v]split=2[cam][main]",
                 f"[cam]{_crop(camera_region)},scale=1080:672:force_original_aspect_ratio=increase,crop=1080:672,setsar=1[top]",
                 f"[main]{_crop(content_region)},scale=1080:1248:force_original_aspect_ratio=increase,crop=1080:1248,setsar=1[bottom]",
                 '[top][bottom]vstack=inputs=2[stack]']
        title = title or moment.get('title') or short_title(transcript, start, end)
        title_file = Path(directory) / 'title.ass'
        title_file.write_text(reaction_title(title, end - start), encoding='utf-8')
        graph.append(f"[stack]ass=filename='{_escape_filter_path(title_file)}'[titled]")
        output_label = '[titled]'
        command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-ss', str(start), '-i', str(source),
                   '-filter_complex', ';'.join(graph), '-map', output_label]
        if preview:
            command += ['-frames:v', '1', '-update', '1', str(destination)]
        else:
            command += ['-t', str(end - start), '-map', '0:a:0?', '-c:v', 'libx264', '-preset', 'veryfast',
                        '-crf', '20', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k',
                        '-movflags', '+faststart', str(destination)]
        if progress:
            progress(('Предпросмотр: ' if preview else 'Shorts: ') + destination.name)
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            destination.unlink(missing_ok=True)
            raise CensorError('Ошибка Shorts:\n' + '\n'.join(result.stderr.splitlines()[-10:]))
    return destination


def create_shorts(source, output, *, model='small', length=35, count=5, camera='Авто',
                  content='0,0,100,100', progress=None, title=''):
    source, output = Path(source), Path(output)
    parse_region(camera)
    parse_region(content)
    if not 10 <= length <= 60 or not 1 <= count <= 20:
        raise CensorError('Длина Shorts: 10–60 секунд; количество: 1–20.')
    if title.strip():
        reaction_title(title, 1)
    duration, has_audio = probe_media(source)
    if not has_audio:
        raise CensorError('Для автоматических Shorts нужна речь в аудиодорожке.')
    transcript, _ = transcribe_video(source, model, progress)
    if progress:
        progress('Выбираю моменты по речи и всплескам звука…')
    signals = audio_peaks(source, progress=progress)
    moments = select_moments(transcript, duration, length, count, signals)
    paths = []
    # Separate batch directory avoids overwriting previous exports.
    output.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="shorts-", dir=output))
    for index, moment in enumerate(moments, 1):
        moment['title'] = title.strip() or short_title(transcript, moment['start'], moment['end'])
        filename_title = re.sub(r'[^\w -]', '', moment['title']).strip()[:60] or 'Shorts'
        destination = output / f'{index:02}_{filename_title}.mp4'
        paths.append(render_reaction(source, moment, destination, transcript, camera, content, progress))
    return paths
