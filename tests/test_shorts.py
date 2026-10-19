import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import cv2

from engine import CensorError
from shorts import create_shorts, detect_camera, parse_region, reaction_captions, render_reaction, select_moments
from test_render import audio_rms


class ShortsTests(unittest.TestCase):
    def test_regions_reject_invalid_bounds(self):
        self.assertIsNone(parse_region('Авто'))
        self.assertEqual(parse_region('10,20,30,40'), (.1, .2, .3, .4))
        for region in ('0,0,0,10', '90,0,30,10', 'nan,0,10,10', '0,0,101,10', '1,2,3'):
            with self.subTest(region=region), self.assertRaises(CensorError):
                parse_region(region)

    def test_moment_selection_is_bounded_and_does_not_overlap(self):
        transcript = [dict(start=i, end=i + 3.8, text='Обычный разговор.') for i in range(0, 120, 4)]
        moments = select_moments(transcript, 120, length=30, count=3)
        self.assertEqual(len(moments), 3)
        for moment in moments:
            self.assertGreaterEqual(moment['start'], 0)
            self.assertLessEqual(moment['end'], 120)
            self.assertGreaterEqual(moment['end'] - moment['start'], 21)
            self.assertLessEqual(moment['end'] - moment['start'], 37.5)
        self.assertTrue(all(a['end'] <= b['start'] for a, b in zip(moments, moments[1:])))
        selected = select_moments(transcript, 120, length=30, count=1,
                                  signals=[dict(start=90, end=92, score=2)])
        self.assertLess(selected[0]['start'], 92)
        self.assertGreater(selected[0]['end'], 90)
        with self.assertRaises(CensorError):
            select_moments([], 120)

    def test_word_captions_are_relative_clamped_and_escaped(self):
        transcript = [dict(start=8, end=15, text='фраза', words=[
            dict(start=8, end=9, word='до'), dict(start=9.5, end=10.5, word='мальчики'),
            dict(start=10.5, end=10.8, word='и'), dict(start=10.8, end=11.5, word='девочки'),
            dict(start=11.7, end=14, word=r'{\pos(0,0)}после')])]
        captions = reaction_captions(transcript, 10, 12)
        self.assertIn('МАЛЬЧИКИ И ДЕВОЧКИ', captions)
        self.assertIn('0:00:00.00,0:00:01.50', captions)
        self.assertIn('0:00:01.70,0:00:02.00', captions)
        self.assertNotIn(r'{\pos', captions)
        self.assertIn('&H0000FFFF', captions)

    def test_split_export_preview_and_automatic_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                            'color=red:size=640x360:rate=25:duration=2', '-f', 'lavfi', '-i',
                            'sine=frequency=600:duration=2', '-vf',
                            'drawbox=x=0:y=0:w=256:h=144:color=blue:t=fill',
                            '-c:v', 'libx264', '-c:a', 'aac', '-shortest', str(source)], check=True)
            transcript = [dict(start=.1, end=1.8, text='Мальчики и девочки', words=[
                dict(start=.1, end=.8, word='Мальчики'), dict(start=.8, end=1, word='и'),
                dict(start=1, end=1.8, word='девочки')])]
            # A real detector on a faceless source must report a useful error.
            with self.assertRaisesRegex(CensorError, 'Камера не найдена'):
                detect_camera(source, 0, 2)
            # A stable webcam is preferred to a larger content face, ignoring badges.
            with patch('cv2.CascadeClassifier') as classifier:
                classifier.return_value.detectMultiScale.return_value = [
                    (30, 30, 50, 60), (300, 180, 120, 120), (30, 415, 25, 25)]
                camera_region = detect_camera(source, 0, 2)
                self.assertLess(camera_region[0], .15)
                self.assertLess(camera_region[1], .15)
                self.assertGreater(camera_region[2], .1)
            camera, content = '0,0,40,40', '40,0,60,100'
            output = root / 'clip.mp4'
            render_reaction(source, dict(start=.5, end=1.5), output, transcript, camera, content)
            data = json.loads(subprocess.check_output([
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration:stream=width,height,codec_type',
                '-of', 'json', str(output)]))
            video = next(stream for stream in data['streams'] if stream['codec_type'] == 'video')
            self.assertEqual((video['width'], video['height']), (1080, 1920))
            self.assertAlmostEqual(float(data['format']['duration']), 1, delta=.1)
            ratio = audio_rms(output, .1, .2) / audio_rms(source, .6, .2)
            self.assertAlmostEqual(ratio, 1, delta=.05)
            preview = root / 'preview.png'
            render_reaction(source, dict(start=.5, end=1.5), preview, transcript, camera, content, preview=True)
            frame = cv2.imread(str(preview))
            self.assertEqual(frame.shape[:2], (1920, 1080))
            self.assertGreater(int(frame[200, 500, 0]), 200)  # Blue camera above.
            self.assertGreater(int(frame[1000, 500, 2]), 200)  # Red content below.
            yellow = (frame[500:650, :, 0] < 100) & (frame[500:650, :, 1] > 170) & (frame[500:650, :, 2] > 170)
            self.assertGreater(int(yellow.sum()), 100)
            with patch('shorts.transcribe_video', return_value=(transcript, 'ru')):
                first = create_shorts(source, root / 'batch', count=2, length=10, camera=camera, content=content)
                second = create_shorts(source, root / 'batch', count=2, length=10, camera=camera, content=content)
            self.assertEqual(len(first), 1)
            self.assertNotEqual(first[0].parent, second[0].parent)
            self.assertTrue(first[0].is_file())
            self.assertTrue(second[0].is_file())


if __name__ == '__main__':
    unittest.main()
