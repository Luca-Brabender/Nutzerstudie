# eyetrax_tracker.py
"""EyeTrax Eyetracking-Backend für den CARLA-LCT-Fahrsimulator.

Nutzt die EyeTrax-Python-Bibliothek bzw. OpenCV und MediaPipe direkt in Python,
um Blickkoordinaten und Kopfpose ohne separaten Browser-Tab zu erfassen.

Kompatibel zum GazeServer-Interface:
- drain() -> Liste von GazeSample
- sample_rate() -> float Hz
- connected -> bool
- calib_start(x, y, label=''), calib_stop(), clear_model()
- push_status(...)
"""

import collections
import math
import os
import sys
import threading
import time

from gaze_server import GazeSample


def _solve_linear_system(A, B):
    """Löst A * W = B für quadratisches A (n x n) und B (n x m) via Gauss-Jordan."""
    n = len(A)
    m = len(B[0])
    M = [A[i][:] + B[i][:] for i in range(n)]
    for i in range(n):
        pivot = i
        for r in range(i + 1, n):
            if abs(M[r][i]) > abs(M[pivot][i]):
                pivot = r
        M[i], M[pivot] = M[pivot], M[i]
        diag = M[i][i]
        if abs(diag) < 1e-12:
            diag = 1e-12
        for c in range(i, n + m):
            M[i][c] /= diag
        for r in range(n):
            if r != i:
                factor = M[r][i]
                for c in range(i, n + m):
                    M[r][c] -= factor * M[i][c]
    return [[M[i][n + j] for j in range(m)] for i in range(n)]


class EyeTraxTracker(object):

    def __init__(self, camera_idx=0, width=640, height=480, max_queue=20000):
        self.backend_name = 'EyeTrax'
        self.is_direct_camera = True
        self.camera_idx = camera_idx
        self.width = width
        self.height = height
        self.http_port = 8000  # Fallback

        self._lock = threading.Lock()
        self._queue = collections.deque(maxlen=max_queue)
        self._rate_window = collections.deque(maxlen=120)

        self._running = False
        self._thread = None
        self._cap = None

        self.latest = None
        self.calib_samples = 0
        self._calib_target = None  # (x, y) während Kalibrierpunkt
        self._calib_X = []        # Liste von Feature-Vektoren
        self._calib_Y = []        # Liste von [x, y]-Zielen
        self._weights = None      # 7 x 2 Regressionsmatrix

        self._estimator = None
        self._face_mesh = None

    # ------------------------------------------------------------------ start / stop
    def start(self):
        """Initialisiert Kamera, MediaPipe/EyeTrax und startet den Hintergrund-Thread."""
        if self._running:
            return

        # 1. Prüfen, ob OpenCV installiert ist
        try:
            import cv2
        except ImportError:
            raise RuntimeError(
                "[EYETRAX] OpenCV ('cv2') ist nicht installiert.\n"
                "Bitte installieren mit:\n"
                "    pip install opencv-python eyetrax mediapipe"
            )

        # 2. Prüfen, ob EyeTrax oder MediaPipe installiert ist
        # Wichtig: Lokalen Ordner './mediapipe' temporär aus sys.path nehmen,
        # damit er das echte Python-Paket aus site-packages nicht überschattet
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        saved_path = list(sys.path)
        sys.path = [p for p in sys.path if os.path.abspath(p) != cur_dir and p != '']

        has_eyetrax = False
        has_mediapipe = False
        mp_module = None

        try:
            import eyetrax
            has_eyetrax = True
            print("[EYETRAX] EyeTrax Python-Bibliothek erfolgreich geladen.")
        except ImportError:
            pass

        try:
            import mediapipe as mp
            has_mediapipe = True
            mp_module = mp
        except ImportError:
            pass
        finally:
            sys.path = saved_path

        if not has_eyetrax and not has_mediapipe:
            raise RuntimeError(
                "[EYETRAX] Weder 'eyetrax' noch 'mediapipe' gefunden.\n"
                "Bitte installieren mit:\n"
                "    pip install eyetrax opencv-python mediapipe"
            )

        # 3. EyeTrax Estimator oder MediaPipe FaceMesh initialisieren
        if has_eyetrax and hasattr(eyetrax, 'GazeEstimator'):
            try:
                self._estimator = eyetrax.GazeEstimator()
                print("[EYETRAX] GazeEstimator instanziiert.")
            except Exception as e:
                print(f"[EYETRAX] Hinweis: GazeEstimator konnte nicht direkt instanziiert werden ({e}). Nutze MediaPipe FaceMesh.")

        if mp_module is not None and hasattr(mp_module, 'solutions'):
            try:
                self._face_mesh = mp_module.solutions.face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                print("[EYETRAX] MediaPipe FaceMesh Pipeline bereit.")
            except Exception as e:
                print(f"[EYETRAX] Warnung beim Initialisieren von FaceMesh: {e}")

        # 4. Kamera öffnen
        self._cap = cv2.VideoCapture(self.camera_idx)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"[EYETRAX] Kamera {self.camera_idx} konnte nicht geöffnet werden!\n"
                "Ist eine Webcam angeschlossen oder greift ein anderes Programm darauf zu?"
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self._running = True
        self._thread = threading.Thread(target=self._worker, name="EyeTraxCapture", daemon=True)
        self._thread.start()
        print(f"[EYETRAX] Eyetracking auf Kamera {self.camera_idx} erfolgreich gestartet.")

    def stop(self):
        """Beendet den Kamera-Thread und gibt Ressourcen frei."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

        if self._face_mesh is not None:
            try:
                self._face_mesh.close()
            except Exception:
                pass
            self._face_mesh = None

    # ------------------------------------------------------------------ worker thread
    def _worker(self):
        import cv2

        while self._running:
            if self._cap is None:
                break
            ret, frame = self._cap.read()
            now = time.time()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            face = False
            yaw, pitch, iod = None, None, None
            pred_x, pred_y = None, None

            # 1. Gesichts- und Landmarken-Extraktion
            if self._face_mesh is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self._face_mesh.process(rgb)

                if res.multi_face_landmarks:
                    face = True
                    lm = res.multi_face_landmarks[0].landmark
                    n_lm = len(lm)

                    # Äußere Augenwinkel (33, 263), Nasenspitze (1)
                    if n_lm > 263:
                        L = (lm[33].x, lm[33].y)
                        R = (lm[263].x, lm[263].y)
                        N = (lm[1].x, lm[1].y)

                        mx = (L[0] + R[0]) / 2.0
                        my = (L[1] + R[1]) / 2.0
                        iod = math.hypot(R[0] - L[0], R[1] - L[1])

                        if iod > 1e-4:
                            yaw = (N[0] - mx) / iod
                            pitch = (N[1] - my) / iod

                        # Iris- und Augenzentren für Gaze-Features
                        # Wenn Iris-Landmarks (468+) vorhanden:
                        if n_lm >= 473:
                            iris_l = (lm[468].x, lm[468].y)
                            iris_r = (lm[473].x, lm[473].y)
                        else:
                            # Fallback: Mittelpunkte der Augenkonturen
                            iris_l = ((lm[33].x + lm[133].x) / 2.0, (lm[33].y + lm[133].y) / 2.0)
                            iris_r = ((lm[362].x + lm[263].x) / 2.0, (lm[362].y + lm[263].y) / 2.0)

                        center_l = ((lm[33].x + lm[133].x) / 2.0, (lm[33].y + lm[133].y) / 2.0)
                        center_r = ((lm[362].x + lm[263].x) / 2.0, (lm[362].y + lm[263].y) / 2.0)

                        dx_l = iris_l[0] - center_l[0]
                        dy_l = iris_l[1] - center_l[1]
                        dx_r = iris_r[0] - center_r[0]
                        dy_r = iris_r[1] - center_r[1]

                        # Feature-Vektor für Blickpunkt-Regression (7 Dimensionen)
                        feat = [dx_l, dy_l, dx_r, dy_r, yaw if yaw is not None else 0.0,
                                pitch if pitch is not None else 0.0, 1.0]

                        # Kalibrierung: Sample aufzeichnen, falls Zielpunkt aktiv
                        with self._lock:
                            if self._calib_target is not None:
                                self._calib_X.append(feat)
                                self._calib_Y.append(list(self._calib_target))
                                self.calib_samples += 1

                            # Blickpunkt vorhersagen, falls Modell trainiert
                            if self._weights is not None:
                                pred_x = sum(feat[k] * self._weights[k][0] for k in range(7))
                                pred_y = sum(feat[k] * self._weights[k][1] for k in range(7))

            # 2. Fallback auf EyeTrax Estimator, falls vorhanden und kein internes Modell aktiv
            if pred_x is None and self._estimator is not None:
                try:
                    pt = self._estimator.predict(frame)
                    if pt is not None and len(pt) >= 2:
                        pred_x, pred_y = float(pt[0]), float(pt[1])
                        face = True
                except Exception:
                    pass

            sample = GazeSample(
                t=now,
                t_recv=now,
                x=pred_x,
                y=pred_y,
                face=face,
                yaw=yaw,
                pitch=pitch,
                iod=iod,
            )

            with self._lock:
                self._queue.append(sample)
                self.latest = sample
                self._rate_window.append(now)

            # Kleine Pause zur CPU-Schonung (max. ~60 Hz)
            time.sleep(0.005)

    # ------------------------------------------------------------------ public API
    @property
    def connected(self):
        return self._cap is not None and self._cap.isOpened() and self._running

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

    def calib_start(self, x, y, label=''):
        with self._lock:
            self._calib_target = (float(x), float(y))

    def calib_stop(self):
        with self._lock:
            self._calib_target = None
            self._fit_model()

    def clear_model(self):
        with self._lock:
            self._calib_target = None
            self._calib_X = []
            self._calib_Y = []
            self._weights = None
            self.calib_samples = 0

    def push_status(self, **kwargs):
        pass

    def _fit_model(self):
        """Trainiert eine Ridge-Regression (f -> [x, y]) auf den Kalibrierpunkten."""
        if len(self._calib_X) < 5:
            return

        X = self._calib_X
        Y = self._calib_Y
        n = len(X)
        d = len(X[0])  # 7

        # A = X^T * X + lambda * I
        lam = 1e-3
        A = [[0.0] * d for _ in range(d)]
        for i in range(d):
            for j in range(d):
                s = sum(X[k][i] * X[k][j] for k in range(n))
                if i == j and i < d - 1:
                    s += lam
                A[i][j] = s

        # B = X^T * Y (d x 2)
        B = [[0.0, 0.0] for _ in range(d)]
        for i in range(d):
            B[i][0] = sum(X[k][i] * Y[k][0] for k in range(n))
            B[i][1] = sum(X[k][i] * Y[k][1] for k in range(n))

        try:
            self._weights = _solve_linear_system(A, B)
            print(f"[EYETRAX] Modell erfolgreich mit {n} Kalibriersamples trainiert.")
        except Exception as e:
            print(f"[EYETRAX] Fehler beim Trainieren des Kalibriermodells: {e}")

