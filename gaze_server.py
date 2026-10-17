# gaze_server.py
"""WebSocket-Bridge zwischen WebGazer (Browser) und dem CARLA-LCT-Client.

* Startet einen WebSocket-Server (Default ws://localhost:8765), über den die
  Browser-Seite ``gaze_tracker.html`` jedes Gaze-Sample schickt.
* Startet einen kleinen HTTP-Server (Default http://localhost:8000), der den
  Nutzerstudien-Ordner ausliefert. WebGazer braucht einen "secure context"
  (localhost oder https) für den Kamerazugriff -> file:// ist unzuverlässig.
* Rückkanal: Python kann Befehle an den Browser schicken (Kalibrierpunkt
  start/stop, Daten löschen, aktuelle AOI für die Versuchsleiter-Anzeige).

Alle Samples werden thread-sicher in einer Queue gepuffert und vom Hauptloop
(pygame) pro Frame mit ``drain()`` abgeholt -> es geht kein Sample verloren,
auch wenn der Simulator langsamer als die Kamera läuft.
"""

import asyncio
import collections
import functools
import http.server
import json
import os
import threading
import time

try:
    import websockets
except ImportError:  # pragma: no cover
    websockets = None


GazeSample = collections.namedtuple(
    'GazeSample',
    [
        't',          # Unix-Zeit (s) der Aufnahme im Browser (Date.now()/1000)
        't_recv',     # Unix-Zeit (s) des Empfangs in Python
        'x', 'y',     # WebGazer-Vorhersage in pygame-Fensterkoordinaten (px), None ohne Gesicht
        'face',       # True, wenn ein Gesicht/Augen erkannt wurden
        'yaw',        # relative Kopfdrehung (Nase vs. Augenmitte / Augenabstand)
        'pitch',      # relative Kopfneigung
        'iod',        # Augenabstand im Kamerabild (px) -> Distanz-Proxy
    ],
)


class GazeServer(object):

    def __init__(self, ws_host='localhost', ws_port=8765, http_port=8000,
                 web_root=None, max_queue=20000):
        self.ws_host = ws_host
        self.ws_port = ws_port
        self.http_port = http_port
        self.web_root = web_root or os.path.dirname(os.path.abspath(__file__))

        self._lock = threading.Lock()
        self._queue = collections.deque(maxlen=max_queue)
        self._clients = set()
        self._loop = None
        self._started = False

        self.latest = None               # letztes GazeSample
        self.browser_info = {}           # Infos aus der 'hello'-Nachricht
        self.calib_samples = 0           # Anzahl Trainingssamples im WebGazer-Modell
        self._rate_window = collections.deque(maxlen=120)
        self.backend_name = 'WebGazer'
        self.is_direct_camera = False

    # ------------------------------------------------------------------ start
    def start(self):
        if self._started:
            return
        if websockets is None:
            raise RuntimeError("Paket 'websockets' fehlt: pip install websockets")
        self._started = True
        threading.Thread(target=self._run_ws, daemon=True).start()
        threading.Thread(target=self._run_http, daemon=True).start()
        print('[GAZE] WebSocket-Server: ws://%s:%d' % (self.ws_host, self.ws_port))
        print('[GAZE] Tracker-Seite im Browser öffnen: http://localhost:%d/gaze_tracker.html'
              % self.http_port)

    def _run_http(self):
        handler = functools.partial(_QuietHandler, directory=self.web_root)
        try:
            httpd = http.server.ThreadingHTTPServer(('localhost', self.http_port), handler)
        except OSError as e:
            print('[GAZE] HTTP-Server konnte nicht starten (%s) - läuft evtl. schon.' % e)
            return
        httpd.serve_forever()

    def _run_ws(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        async def main():
            async with websockets.serve(self._handler, self.ws_host, self.ws_port,
                                        max_size=2 ** 20):
                await asyncio.Future()

        try:
            self._loop.run_until_complete(main())
        except OSError as e:
            print('[GAZE] WebSocket-Server konnte nicht starten: %s' % e)

    # websockets < 10.1 übergibt zusätzlich 'path'
    async def _handler(self, websocket, path=None):
        self._clients.add(websocket)
        print('[GAZE] Browser verbunden.')
        try:
            async for message in websocket:
                self._on_message(message)
        except Exception:
            pass
        finally:
            self._clients.discard(websocket)
            print('[GAZE] Browser getrennt.')

    def _on_message(self, message):
        try:
            data = json.loads(message)
        except ValueError:
            return
        mtype = data.get('type', 'gaze')
        now = time.time()

        if mtype == 'gaze':
            face = bool(data.get('face', False))
            x = data.get('x')
            y = data.get('y')
            sample = GazeSample(
                t=float(data.get('t', now * 1000.0)) / 1000.0,
                t_recv=now,
                x=float(x) if (face and x is not None) else None,
                y=float(y) if (face and y is not None) else None,
                face=face,
                yaw=_f(data.get('yaw')),
                pitch=_f(data.get('pitch')),
                iod=_f(data.get('iod')),
            )
            with self._lock:
                self._queue.append(sample)
                self.latest = sample
                self._rate_window.append(now)
            if 'calib_samples' in data:
                self.calib_samples = int(data['calib_samples'])
        elif mtype == 'hello':
            self.browser_info = data
            self.calib_samples = int(data.get('calib_samples', 0))
            print('[GAZE] Browser-Info: %s' % data)
        elif mtype == 'calib_ack':
            self.calib_samples = int(data.get('calib_samples', self.calib_samples))

    # ----------------------------------------------------------------- public
    @property
    def connected(self):
        return len(self._clients) > 0

    def drain(self):
        """Gibt alle seit dem letzten Aufruf empfangenen Samples zurück."""
        with self._lock:
            items = list(self._queue)
            self._queue.clear()
        return items

    def sample_rate(self, window_s=2.0):
        now = time.time()
        with self._lock:
            n = sum(1 for t in self._rate_window if now - t <= window_s)
        return n / window_s

    def send(self, payload):
        """Schickt ein JSON-Kommando an alle verbundenen Browser (thread-sicher)."""
        if self._loop is None or not self._clients:
            return
        msg = json.dumps(payload)

        async def _broadcast():
            for ws in list(self._clients):
                try:
                    await ws.send(msg)
                except Exception:
                    pass

        asyncio.run_coroutine_threadsafe(_broadcast(), self._loop)

    # Komfort-Kommandos ------------------------------------------------------
    def calib_start(self, x, y, label=''):
        self.send({'cmd': 'calib_start', 'x': float(x), 'y': float(y), 'label': label})

    def calib_stop(self):
        self.send({'cmd': 'calib_stop'})

    def clear_model(self):
        self.calib_samples = 0
        self.send({'cmd': 'clear'})

    def push_status(self, **kwargs):
        payload = {'cmd': 'status'}
        payload.update(kwargs)
        self.send(payload)


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

