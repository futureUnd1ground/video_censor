from array import array
import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest

from engine import CENSOR_MODES, Hit, render


def audio_rms(path, start, duration):
    samples = array('f')
    samples.frombytes(subprocess.check_output([
        'ffmpeg', '-v', 'error', '-i', str(path), '-ss', str(start),
        '-t', str(duration), '-vn', '-ac', '1', '-ar', '48000',
        '-f', 'f32le', 'pipe:1']))
    return math.sqrt(sum(value * value for value in samples) / len(samples))


class RenderTests(unittest.TestCase):
    def test_dnxhd_source_exports_to_mp4(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'editing-source.mov'
            subprocess.run([
                'ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                'testsrc2=size=320x180:rate=25:duration=1.2',
                '-f', 'lavfi', '-i', 'sine=frequency=600:duration=1.2',
                '-c:v', 'dnxhd', '-profile:v', 'dnxhr_lb', '-pix_fmt', 'yuv422p',
                '-c:a', 'pcm_s16le', '-shortest', str(source)], check=True)
            original_rms = audio_rms(source, .1, .2)
            for mode in ('Mute', 'Bleep'):
                with self.subTest(mode=mode):
                    output = root / f'{mode}.mp4'
                    render(source, output, [Hit('word', .5, .8)], mode, 0)
                    data = json.loads(subprocess.check_output([
                        'ffprobe', '-v', 'error', '-show_entries',
                        'format=duration:stream=codec_type,codec_name', '-of', 'json', str(output)]))
                    codecs = {stream['codec_type']: stream['codec_name'] for stream in data['streams']}
                    self.assertEqual(codecs, {'video': 'h264', 'audio': 'aac'})
                    self.assertAlmostEqual(float(data['format']['duration']), 1.2, delta=.1)
                    difference_db = 20 * math.log10(audio_rms(output, .1, .2) / original_rms)
                    self.assertLess(abs(difference_db), .5)

    def test_every_mode_renders_expected_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                            'testsrc2=size=160x120:rate=20:duration=2', '-f', 'lavfi', '-i',
                            'sine=frequency=600:duration=2', '-c:v', 'libx264', '-c:a', 'aac',
                            '-af', "volume=0.25:enable='lt(t,1.1)'",
                            '-shortest', str(source)], check=True)
            sound = root / 'sound.wav'
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                            'sine=frequency=300:duration=0.1', str(sound)], check=True)
            reference = [(start, audio_rms(source, start, .2)) for start in (.15, 1.4)]
            self.assertGreater(20 * math.log10(reference[1][1] / reference[0][1]), 10)
            for mode in CENSOR_MODES:
                with self.subTest(mode=mode):
                    output = root / f'{mode}.mp4'
                    render(source, output, [Hit('a', .5, .8), Hit('b', .7, 1),
                                            Hit('ignored', 1.4, 1.9, False)], mode, 0, sound)
                    data = json.loads(subprocess.check_output(['ffprobe', '-v', 'error',
                        '-show_entries', 'format=duration:stream=codec_type', '-of', 'json', str(output)]))
                    expected = 1.5 if mode == 'Cut' else 2 - .5 * 2/3 if mode == 'Fast-Forward' else 2
                    self.assertAlmostEqual(float(data['format']['duration']), expected, delta=.16)
                    self.assertEqual({s['codec_type'] for s in data['streams']}, {'video', 'audio'})
                    if mode == 'Mute':
                        def video_hash(path):
                            return subprocess.check_output([
                                'ffmpeg', '-v', 'error', '-i', str(path), '-map', '0:v:0',
                                '-c:v', 'copy', '-f', 'hash', 'pipe:1'])
                        self.assertEqual(video_hash(source), video_hash(output))
                    # Both quiet and loud speech must keep their original level,
                    # including after timeline changes in Cut / Fast-Forward.
                    for start, original_rms in reference:
                        offset = .5 if mode == 'Cut' else 1/3 if mode == 'Fast-Forward' else 0
                        output_start = start - offset if start > 1 else start
                        processed_rms = audio_rms(output, output_start, .2)
                        difference_db = 20 * math.log10(processed_rms / original_rms)
                        self.assertLess(abs(difference_db), .5,
                                        f'{mode}: level changed by {difference_db:.2f} dB')


if __name__ == '__main__':
    unittest.main()
