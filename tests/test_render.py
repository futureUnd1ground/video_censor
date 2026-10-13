import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from engine import CENSOR_MODES, Hit, render


class RenderTests(unittest.TestCase):
    def test_every_mode_renders_expected_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                            'testsrc2=size=160x120:rate=20:duration=2', '-f', 'lavfi', '-i',
                            'sine=frequency=600:duration=2', '-c:v', 'libx264', '-c:a', 'aac',
                            '-shortest', str(source)], check=True)
            sound = root / 'sound.wav'
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                            'sine=frequency=300:duration=0.1', str(sound)], check=True)
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


if __name__ == '__main__':
    unittest.main()
