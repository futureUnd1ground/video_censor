import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from engine import CensorError
from shorts import _camera_faces, create_shorts, detect_camera, parse_region, reaction_title, render_reaction, short_title, select_moments
from test_render import audio_rms


class ShortsTests(unittest.TestCase):
    def test_corner_detections_map_to_source_and_are_not_counted_twice(self):
        detector = Mock()
        # Same source face at x=650,y=50,w=40,h=50, seen at two scales.
        full = np.array([[1040, 80, 64, 80] + [0] * 10 + [.8]], dtype=np.float32)
        corner = np.array([[210 * 640/360, 50 * 356/200, 40 * 640/360,
                            50 * 356/200] + [0] * 10 + [.95]], dtype=np.float32)
        detector.detect.side_effect = [(None, full), (None, None), (None, None),
                                       (None, corner), (None, None)]
        faces = _camera_faces(np.zeros((400, 800, 3), dtype=np.uint8), detector)
        self.assertEqual(detector.setInputSize.call_count, 5)
        self.assertEqual(len(faces), 1)
        for actual, expected in zip(faces[0], (.8375, .1875, .05, .125, .95)):
            self.assertAlmostEqual(actual, expected, places=5)

    def test_one_frame_false_positive_is_rejected(self):
        capture = Mock()
        capture.read.return_value = (True, np.zeros((400, 800, 3), dtype=np.uint8))
        face = (.85, .2, .05, .1, .95)
        with patch('cv2.VideoCapture', return_value=capture), patch(
                'shorts._camera_faces', side_effect=[[face]] + [[]] * 11):
            with self.assertRaisesRegex(CensorError, 'Камера не найдена'):
                detect_camera(Path('synthetic.mp4'), 0, 30)
        capture.release.assert_called_once()

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

    def test_title_is_persistent_and_escaped(self):
        transcript = [dict(start=8, end=15, text='фраза', words=[
            dict(start=8, end=9, word='до'), dict(start=10, end=10.5, word='мальчики'),
            dict(start=10.5, end=10.8, word='и'), dict(start=10.8, end=11.5, word='девочки'),
            dict(start=13, end=14, word='после')])]
        title = short_title(transcript, 10, 12)
        self.assertEqual(title, 'мальчики и девочки')
        overlay = reaction_title(title, 35)
        self.assertIn('МАЛЬЧИКИ И ДЕВОЧКИ', overlay)
        self.assertEqual(overlay.count('Dialogue:'), 1)
        self.assertIn('0:00:00.00,0:00:35.00', overlay)
        self.assertIn('&H0000FFFF', overlay)
        self.assertNotIn(r'{\pos', reaction_title(r'{\pos(0,0)}текст', 10))
        with self.assertRaises(CensorError):
            reaction_title('а' * 97, 10)

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
                dict(start=1, end=1.1, word='девочки')])]
            # A real detector on a faceless source must report a useful error.
            with self.assertRaisesRegex(CensorError, 'Камера не найдена'):
                detect_camera(source, 0, 2)
            # A stable webcam is preferred to a larger content face, ignoring badges.
            with patch('shorts._camera_faces', return_value=[
                    (.07, .13, .06, .13, .95), (.45, .53, .15, .26, .95)]):
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
            # The title is still visible after all spoken words have ended.
            capture = cv2.VideoCapture(str(output))
            capture.set(cv2.CAP_PROP_POS_MSEC, 900)
            ok, late_frame = capture.read()
            capture.release()
            self.assertTrue(ok)
            late_yellow = ((late_frame[500:650, :, 0] < 100)
                           & (late_frame[500:650, :, 1] > 170) & (late_frame[500:650, :, 2] > 170))
            self.assertGreater(int(late_yellow.sum()), 100)
            with patch('shorts.transcribe_video', return_value=(transcript, 'ru')):
                first = create_shorts(source, root / 'batch', count=2, length=10, camera=camera, content=content)
                second = create_shorts(source, root / 'batch', count=2, length=10, camera=camera, content=content, title='Моё название: никто не ожидал три легендарных меча в самом последнем сундуке!')
            self.assertEqual(len(first), 1)
            self.assertNotEqual(first[0].parent, second[0].parent)
            self.assertTrue(first[0].is_file())
            self.assertTrue(second[0].is_file())
            self.assertIn('P52', first[0].name)
            self.assertIn('мальчики и девочки', first[0].name.lower())
            self.assertIn('Моё название', second[0].name)


if __name__ == '__main__':
    unittest.main()
