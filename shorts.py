"""Local automatic reaction Shorts: moment selection, split framing and persistent titles."""
from __future__ import annotations

import math
from pathlib import Path
import re
import statistics
import subprocess
import tempfile
import textwrap

from engine import CensorError, probe_media, require_tools
from titles import TITLE_LIMIT, short_title
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


def _camera_faces(frame, detector):
    """Detect at frame scale and in enlarged corners; return normalized boxes."""
    import cv2
    height, width = frame.shape[:2]
    corner_w, corner_h = max(1, round(width * .45)), max(1, round(height * .5))
    views = [(0, 0, width, height, 1280)]
    views += [(x, y, corner_w, corner_h, 640)
              for x in (0, width - corner_w) for y in (0, height - corner_h)]
    faces = []
    for left, top, view_w, view_h, limit in views:
        scale = min(2., limit / max(view_w, view_h))
        image = cv2.resize(frame[top:top + view_h, left:left + view_w],
                           (max(1, round(view_w * scale)), max(1, round(view_h * scale))))
        image_h, image_w = image.shape[:2]
        detector.setInputSize((image_w, image_h))
        _, found = detector.detect(image)
        if found is None:
            continue
        for row in found:
            x, y, w, h = [float(value) for value in row[:4]]
            w, h = w * view_w / image_w, h * view_h / image_h
            cx = left + x * view_w / image_w + w / 2
            cy = top + y * view_h / image_h + h / 2
            if min(w, h) < 12 or not (0 <= cx < width and 0 <= cy < height) or cy / height > .9:
                continue  # Tiny badges and bottom-row avatars are not webcams.
            faces.append((cx / width, cy / height, w / width, h / height, float(row[-1])))
    # One face can appear in both the full frame and a corner. Count it once.
    unique = []
    for face in sorted(faces, key=lambda face: -face[4]):
        if not any(abs(face[0] - other[0]) < .4 * min(face[2], other[2])
                   and abs(face[1] - other[1]) < .4 * min(face[3], other[3]) for other in unique):
            unique.append(face)
    return unique


def detect_camera(source: Path, start: float, end: float) -> tuple[float, float, float, float]:
    """Use local YuNet detections and temporal stability to locate the webcam."""
    try:
        import cv2
    except ImportError as exc:
        raise CensorError("Для автопоиска камеры установите зависимости или задайте область камеры вручную.") from exc
    model = Path(__file__).resolve().parent / 'models' / 'yunet' / 'face_detection_yunet_2023mar.onnx'
    if not model.is_file():
        raise CensorError("Не найдена модель YuNet. Обновите приложение вместе с папкой models или задайте камеру вручную.")
    try:
        detector = cv2.FaceDetectorYN.create(str(model), '', (320, 320), .7, .3, 5000)
    except cv2.error as exc:
        raise CensorError("Не удалось загрузить модель камеры YuNet. Переустановите зависимости приложения.") from exc
    capture = cv2.VideoCapture(str(source))
    groups = []
    valid_frames = 0
    try:
        for index in range(12):
            second = start + (end - start) * (index + .5) / 12
            capture.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
            ok, frame = capture.read()
            if not ok:
                continue
            valid_frames += 1
            height, width = frame.shape[:2]
            seen = set()
            for face in _camera_faces(frame, detector):
                group = next((g for g in groups
                              if abs(statistics.median(f[0] for f in g) - face[0]) < max(.025, min(.08, face[2]))
                              and abs(statistics.median(f[1] for f in g) - face[1]) < max(.025, min(.08, face[3]))), None)
                if group is None:
                    group = []
                    groups.append(group)
                if id(group) not in seen:
                    group.append(face)
                    seen.add(id(group))
        if not valid_frames:
            raise CensorError("Не удалось прочитать кадры исходника для поиска камеры. Проверьте видео.")
        groups = [g for g in groups if len(g) >= min(2, valid_frames)]
        if not groups:
            raise CensorError("Камера не найдена. Задайте её область вручную и нажмите «Проверить кадр».")
        def rank(group):
            cx, cy = (statistics.median(f[i] for f in group) for i in (0, 1))
            edge = max(abs(cx - .5), abs(cy - .5)) * 2
            return (len(group) * (1 + .4 * edge), statistics.mean(f[4] for f in group),
                    -statistics.median(f[2] * f[3] for f in group))
        group = max(groups, key=rank)
        cx, cy, fw, fh = [statistics.median(face[i] for face in group) for i in range(4)]
        crop_w = min(1., max(fw * 4.2, fh * height / width * 3.2))
        crop_h = min(1., crop_w * width / height * 672 / 1080)
        crop_w = min(crop_w, crop_h * height / width * 1080 / 672)
        return max(0., min(1 - crop_w, cx - crop_w / 2)), max(0., min(1 - crop_h, cy - crop_h * .48)), crop_w, crop_h
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


def reaction_title(title, duration):
    """One persistent yellow title; no word-timed speech captions."""
    text = ' '.join(str(title).split()).replace('\\', '/').replace('{', '(').replace('}', ')').upper()
    if not text or len(text) > TITLE_LIMIT:
        raise CensorError('Название должно содержать от 1 до 96 символов.')
    font_size, wrap_width = (52, 30) if len(text) > 72 else (58, 26)
    text = r'\N'.join(textwrap.wrap(text, width=wrap_width, break_on_hyphens=False))
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
    header = header.replace('DejaVu Sans,58,', f'DejaVu Sans,{font_size},')
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
