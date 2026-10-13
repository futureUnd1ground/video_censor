from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unittest

from downloads import download_stream


class QuietFiles(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


class Stalled(BaseHTTPRequestHandler):
    def do_GET(self):
        time.sleep(3)
    def log_message(self, *args):
        pass


class DownloadTests(unittest.TestCase):
    def test_download_returns_final_path_and_creates_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                            'testsrc2=size=160x120:rate=10:duration=0.5',
                            '-c:v', 'libx264', str(root / 'clip.mp4')], check=True)
            server = ThreadingHTTPServer(('127.0.0.1', 0), partial(QuietFiles, directory=directory))
            threading.Thread(target=server.serve_forever, daemon=True).start()
            try:
                events = []
                download_stream(f'http://127.0.0.1:{server.server_port}/clip.mp4',
                                root / 'new-folder', threading.Event(), events.append)
                done = [payload for event, payload in events if event == 'stream_done']
                self.assertEqual(len(done), 1)
                self.assertTrue(Path(done[0][1]).is_file())
                self.assertEqual(Path(done[0][1]).parent, root / 'new-folder')
            finally:
                server.shutdown()
                server.server_close()

    def test_stop_during_extraction(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), Stalled)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                stop = threading.Event()
                timer = threading.Timer(.5, stop.set)
                timer.start()
                events = []
                start = time.monotonic()
                download_stream(f'http://127.0.0.1:{server.server_port}/live',
                                directory, stop, events.append)
                self.assertLess(time.monotonic() - start, 7)
                self.assertTrue(any(event == 'stream_stopped' for event, _ in events))
                self.assertFalse(any(event == 'stream_done' for event, _ in events))
                timer.join()
        finally:
            server.shutdown()
            server.server_close()
