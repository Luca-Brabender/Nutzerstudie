# gaze_aoi.py
"""AOI-Klassifikation (Simulator-Monitor vs. AAOS-Headunit) und Blickzuwendungs-
Erkennung für WebGazer-Daten.

Grundidee
---------
WebGazer liefert nur eine (verrauschte) 2D-Vorhersage in Bildschirm-Pixeln.
Die Headunit liegt aber *außerhalb* des Monitors. Deshalb:

1. **Erweiterte Kalibrierung**: Die Headunit bekommt "virtuelle" Pixel-
   koordinaten in derselben Ebene wie der Monitor (z.B. y > Bildschirmhöhe,
   wenn sie unterhalb liegt). Beim Kalibrieren schaut der Proband auf
   Markierungen an der Headunit, und WebGazer wird mit diesen virtuellen
   Koordinaten trainiert -> die Ridge-Regression extrapoliert über den
   Bildschirmrand hinaus.
2. **Kopfpose**: Für den Blick zur Headunit wird meist der Kopf gedreht/geneigt.
   Aus dem FaceMesh werden Yaw/Pitch geschätzt.
3. **Binärer Klassifikator** (logistische Regression) auf
   [gaze_x/W, gaze_y/H, yaw, pitch], trainiert mit den Validierungs-Samples
   (Monitor-Punkte = SIMULATOR, Headunit-Marker = HEADUNIT).
4. **Hysterese** (GlanceDetector): Ein Zustandswechsel wird erst akzeptiert,
   wenn der neue Zustand ``min_dwell_s`` anhält; der Zeitstempel des Wechsels
   wird aber auf das *erste* Sample des neuen Zustands zurückdatiert. So bleibt
   der Wechselzeitpunkt exakt, ohne dass Ausreißer Fehl-Glances erzeugen.

Begriffe nach ISO 15007 / DIN EN ISO 15007-1: Blickzuwendung (glance) =
Zeitraum, in dem der Blick auf ein Zielgebiet (AOI) gerichtet ist, inkl. der
Übergangszeit zu diesem Gebiet.

Pures Python (kein numpy nötig).
"""

import bisect
import collections
import csv
import json
import math
import os

AOI_SIM = 'SIMULATOR'
AOI_HU = 'HEADUNIT'
AOI_OFF = 'OFFSCREEN'
AOI_NOFACE = 'NO_FACE'
AOI_UNKNOWN = 'UNKNOWN'


# ==============================================================================
# -- Konfiguration -------------------------------------------------------------
# ==============================================================================

DEFAULT_CONFIG = {
    # Physikalische Maße des Simulator-Monitors (sichtbare Bildfläche)
    'monitor_width_cm': 59.8,
    'monitor_height_cm': 33.6,
    # Lage des Headunit-DISPLAYS relativ zur linken oberen Ecke der Monitor-
    # Bildfläche, projiziert in die Monitorebene (rechts = +x, unten = +y).
    # Beispiel: Headunit rechts unterhalb des Monitors.
    'headunit_rect_cm': {'left': 38.0, 'top': 42.0, 'width': 17.5, 'height': 11.0},
    # Toleranzrand um den Monitor, innerhalb dessen noch SIMULATOR gilt
    'screen_margin_frac': 0.10,
    # Toleranz für Kopfpose (yaw/pitch) um den Kalibrierbereich
    'head_pose_tolerance': 0.12,
    # Entscheidungsschwelle P(HEADUNIT)
    'hu_threshold': 0.5,
    # Hysterese
    'min_dwell_s': 0.10,          # neuer Zustand muss >= 100 ms stabil sein
    'no_face_bridge_s': 0.50,     # kürzere Gesichtsverluste werden überbrückt
    # Glances kürzer als dies werden im Glance-Log markiert (nicht gelöscht)
    'min_glance_s': 0.12,
}


def load_config(path):
    cfg = dict(DEFAULT_CONFIG)
    if path and os.path.exists(path):
        with open(path) as fh:
            user = json.load(fh)
        cfg.update({k: v for k, v in user.items() if not k.startswith('_')})
    elif path:
        with open(path, 'w') as fh:
            out = {'_hinweis': 'Maße in cm. headunit_rect_cm relativ zur linken oberen '
                               'Ecke der Monitor-Bildfläche (rechts=+x, unten=+y).'}
            out.update(DEFAULT_CONFIG)
            json.dump(out, fh, indent=2, ensure_ascii=False)
        print('[GAZE] Standard-Konfiguration angelegt: %s (bitte Maße anpassen!)' % path)
    return cfg


class Geometry(object):
    """Rechnet die cm-Angaben in (virtuelle) Fenster-Pixel um."""

    def __init__(self, cfg, width_px, height_px):
        self.W = float(width_px)
        self.H = float(height_px)
        sx = self.W / float(cfg['monitor_width_cm'])
        sy = self.H / float(cfg['monitor_height_cm'])
        r = cfg['headunit_rect_cm']
        self.hu_rect = (r['left'] * sx, r['top'] * sy,
                        r['width'] * sx, r['height'] * sy)  # x, y, w, h
        self.margin = float(cfg.get('screen_margin_frac', 0.1))

    def on_screen(self, x, y):
        mx, my = self.margin * self.W, self.margin * self.H
        return -mx <= x <= self.W + mx and -my <= y <= self.H + my

    def in_headunit(self, x, y, expand=0.25):
        hx, hy, hw, hh = self.hu_rect
        ex, ey = expand * hw, expand * hh
        return hx - ex <= x <= hx + hw + ex and hy - ey <= y <= hy + hh + ey

    def screen_targets(self, n=9, inset=0.08):
        """3x3 (oder 5) Kalibrierpunkte auf dem Monitor."""
        xs = [inset, 0.5, 1.0 - inset]
        ys = [inset, 0.5, 1.0 - inset]
        pts = [(x * self.W, y * self.H) for y in ys for x in xs]
        if n == 5:
            pts = [pts[0], pts[2], pts[4], pts[6], pts[8]]
        return pts

    def headunit_targets(self, inset=0.12):
        """5 Marker auf der Headunit: 4 Ecken (eingerückt) + Mitte.

        Reihenfolge = Beschriftung der Aufkleber H1..H5:
        H1 oben links, H2 oben rechts, H3 Mitte, H4 unten links, H5 unten rechts.
        """
        hx, hy, hw, hh = self.hu_rect
        l, r = hx + inset * hw, hx + (1 - inset) * hw
        t, b = hy + inset * hh, hy + (1 - inset) * hh
        return [('H1 (oben links)', l, t), ('H2 (oben rechts)', r, t),
                ('H3 (Mitte)', hx + hw / 2, hy + hh / 2),
                ('H4 (unten links)', l, b), ('H5 (unten rechts)', r, b)]


# ==============================================================================
# -- Klassifikator -------------------------------------------------------------
# ==============================================================================


def _features(sample, geo):
    # Kopfdrehung (yaw) & Kopfneigung (pitch) sind im Fahrsimulator die stabilsten Prädiktoren
    yaw = sample.yaw if sample.yaw is not None else 0.0
    pitch = sample.pitch if sample.pitch is not None else 0.0
    # Gaze-Koordinaten (normalisiert 0..1), falls vorhanden, sonst neutral (0.5)
    gx = (sample.x / geo.W) if (sample.x is not None) else 0.5
    gy = (sample.y / geo.H) if (sample.y is not None) else 0.5
    return [yaw, pitch, gx, gy]


def _sigmoid(z):
    if z < -40:
        return 0.0
    if z > 40:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


class AOIClassifier(object):

    def __init__(self, cfg, geometry):
        self.cfg = cfg
        self.geo = geometry
        self.mean = None
        self.std = None
        self.w = None
        self.b = 0.0
        self.bounds = None
        self.train_report = {}

    @property
    def trained(self):
        return self.w is not None

    # -------------------------------------------------------------- training
    def fit(self, sim_samples, hu_samples, l2=1e-2, lr=0.5, iters=800):
        # Alle Samples mit erkanntem Gesicht nutzen (auch wenn x, y None sind)
        X = [_features(s, self.geo) for s in sim_samples + hu_samples if s.face]
        Y = [0] * sum(1 for s in sim_samples if s.face) + \
            [1] * sum(1 for s in hu_samples if s.face)
        n0, n1 = Y.count(0), Y.count(1)
        if n0 < 5 or n1 < 5:
            print('[GAZE] Zu wenige Samples für Klassifikator (SIM=%d, HU=%d)' % (n0, n1))
            self.w = None
            return None

        d = len(X[0])
        self.mean = [sum(x[j] for x in X) / len(X) for j in range(d)]
        self.std = []
        for j in range(d):
            var = sum((x[j] - self.mean[j]) ** 2 for x in X) / len(X)
            self.std.append(math.sqrt(var) if var > 1e-12 else 1.0)
        Z = [[(x[j] - self.mean[j]) / self.std[j] for j in range(d)] for x in X]
        # Klassengewichte -> balanciert
        cw = {0: len(Y) / (2.0 * n0), 1: len(Y) / (2.0 * n1)}

        w = [0.0] * d
        b = 0.0
        n = float(len(Z))
        for _ in range(iters):
            gw = [0.0] * d
            gb = 0.0
            for z, y in zip(Z, Y):
                p = _sigmoid(sum(wj * zj for wj, zj in zip(w, z)) + b)
                e = (p - y) * cw[y]
                for j in range(d):
                    gw[j] += e * z[j]
                gb += e
            for j in range(d):
                w[j] -= lr * (gw[j] / n + l2 * w[j])
            b -= lr * gb / n
        self.w, self.b = w, b

        # Cluster-Zentren fuer Kopfpose (yaw/pitch) erfassen
        sim_yaws = [s.yaw for s in sim_samples if s.face and s.yaw is not None]
        sim_pitches = [s.pitch for s in sim_samples if s.face and s.pitch is not None]
        hu_yaws = [s.yaw for s in hu_samples if s.face and s.yaw is not None]
        hu_pitches = [s.pitch for s in hu_samples if s.face and s.pitch is not None]

        center_yaw = sum(sim_yaws) / len(sim_yaws) if sim_yaws else 0.0
        center_pitch = sum(sim_pitches) / len(sim_pitches) if sim_pitches else 0.25
        hu_center_yaw = sum(hu_yaws) / len(hu_yaws) if hu_yaws else -0.45
        hu_center_pitch = sum(hu_pitches) / len(hu_pitches) if hu_pitches else 0.35

        pose_tol = float(self.cfg.get('head_pose_tolerance', 0.14))
        sim_yaw_min = min(sim_yaws) - pose_tol if sim_yaws else -0.25
        sim_yaw_max = max(sim_yaws) + pose_tol if sim_yaws else 0.20
        sim_pitch_min = min(sim_pitches) - pose_tol if sim_pitches else -0.05
        sim_pitch_max = max(sim_pitches) + pose_tol if sim_pitches else 0.40

        hu_yaw_min = min(hu_yaws) - pose_tol if hu_yaws else -0.85
        hu_yaw_max = max(hu_yaws) + pose_tol if hu_yaws else -0.20
        hu_pitch_min = min(hu_pitches) - pose_tol if hu_pitches else 0.18
        hu_pitch_max = max(hu_pitches) + pose_tol if hu_pitches else 0.55

        self.bounds = {
            'center_yaw': round(center_yaw, 4),
            'center_pitch': round(center_pitch, 4),
            'hu_center_yaw': round(hu_center_yaw, 4),
            'hu_center_pitch': round(hu_center_pitch, 4),
            'sim_yaw': [round(sim_yaw_min, 4), round(sim_yaw_max, 4)],
            'sim_pitch': [round(sim_pitch_min, 4), round(sim_pitch_max, 4)],
            'hu_yaw': [round(hu_yaw_min, 4), round(hu_yaw_max, 4)],
            'hu_pitch': [round(hu_pitch_min, 4), round(hu_pitch_max, 4)],
        }

        # Trainingsgüte
        correct = {0: 0, 1: 0}
        for z, y in zip(Z, Y):
            p = _sigmoid(sum(wj * zj for wj, zj in zip(w, z)) + b)
            if (p >= self.cfg['hu_threshold']) == bool(y):
                correct[y] += 1
        self.train_report = {
            'n_sim': n0, 'n_hu': n1,
            'acc_sim': correct[0] / float(n0), 'acc_hu': correct[1] / float(n1),
            'weights': dict(zip(['yaw', 'pitch', 'gx', 'gy'], [round(v, 3) for v in w])),
            'bounds': self.bounds,
        }
        print('[GAZE] Klassifikator trainiert: %s' % self.train_report)
        return self.train_report

    # ---------------------------------------------------------- prediction
    def p_headunit(self, sample):
        if not sample.face:
            return None
        if self.w is None:
            # Einfacher Heuristik-Fallback: Kopfdrehung nach rechts/unten
            yaw = sample.yaw if sample.yaw is not None else 0.0
            pitch = sample.pitch if sample.pitch is not None else 0.0
            return 1.0 if (yaw < -0.22 and pitch > 0.25) else 0.0
        f = _features(sample, self.geo)
        z = [(f[j] - self.mean[j]) / self.std[j] for j in range(len(f))]
        return _sigmoid(sum(wj * zj for wj, zj in zip(self.w, z)) + self.b)

    def classify(self, sample):
        """Klassifiziert das Blickziel in:
        - AOI_NOFACE: Kein Gesicht vor der Kamera erkannt
        - AOI_HU: Blick auf die Headunit (Android Automotive)
        - AOI_SIM: Blick auf den Simulator-Monitor (Straße / Desktop)
        - AOI_OFF: Weder Bildschirm noch Headunit (Blick abgewandt / zur Seite / oben / unten)
        - AOI_UNKNOWN: Unbestimmt / zu wenig Daten
        -> Rückgabe: (aoi_raw, p_headunit)
        """
        if not sample.face:
            return AOI_NOFACE, None

        yaw = sample.yaw if sample.yaw is not None else 0.0
        pitch = sample.pitch if sample.pitch is not None else 0.0

        # Kalibrierte Referenzwerte für Monitor- und Headunit-Kopfpose
        bounds = self.bounds or {}
        if 'center_yaw' in bounds:
            center_yaw = float(bounds['center_yaw'])
            center_pitch = float(bounds['center_pitch'])
        elif 'sim_yaw' in bounds and len(bounds['sim_yaw']) == 2:
            center_yaw = (float(bounds['sim_yaw'][0]) + float(bounds['sim_yaw'][1])) / 2.0
            center_pitch = (float(bounds['sim_pitch'][0]) + float(bounds['sim_pitch'][1])) / 2.0
        else:
            center_yaw = 0.0
            center_pitch = 0.25

        hu_thresh = float(self.cfg.get('hu_threshold', 0.5))

        # Wenn noch kein Modell trainiert ist -> robuste Heuristik
        if self.w is None:
            # 1. Blick deutlich abgewandt (OFFSCREEN)
            if (yaw > max(center_yaw + 0.18, 0.20) or
                pitch < min(center_pitch - 0.25, -0.05) or
                (pitch > max(center_pitch + 0.20, 0.44) and yaw > -0.18) or
                yaw < -0.80):
                return AOI_OFF, 0.0
            # 2. Blick zur Headunit
            if (-0.80 <= yaw <= -0.18 and pitch >= 0.18):
                return AOI_HU, 0.9
            # 3. Blick auf den Simulator (Geradeaus)
            return AOI_SIM, 0.1

        p = self.p_headunit(sample)
        if p is None:
            return AOI_UNKNOWN, None

        # --- A) OFFSCREEN (Weder Monitor noch Headunit): ---
        # 1. Blick nach links (Fenster / linker Außenspiegel): Kopf deutlich nach links gedreht
        if yaw > max(center_yaw + 0.18, 0.20):
            return AOI_OFF, p
        # 2. Blick nach oben (Decke / Rückspiegel): Kopf nach oben geneigt
        if pitch < min(center_pitch - 0.25, -0.05):
            return AOI_OFF, p
        # 3. Blick nach unten in den Schoß / aufs Lenkrad: Kopf nach unten geneigt, aber NICHT zur Headunit gedreht
        if pitch > max(center_pitch + 0.20, 0.44) and yaw > -0.18:
            return AOI_OFF, p
        # 4. Blick extrem weit rechts an der Headunit vorbei
        if yaw < -0.80:
            return AOI_OFF, p

        # --- B) HEADUNIT (Android Automotive): ---
        # p >= Schwelle UND Kopfpose passt zur Headunit (nach rechts & unten geneigt)
        is_hu_pose = (-0.80 <= yaw <= -0.18 and pitch >= 0.18)
        if (p >= hu_thresh and is_hu_pose) or (-0.75 <= yaw <= -0.30 and pitch >= 0.22 and p >= 0.35):
            return AOI_HU, p

        # --- C) SIMULATOR (Monitor / Straße / Desktop): ---
        # Proband blickt nach vorn (weder abgewandt noch auf die Headunit)
        return AOI_SIM, p

    # ----------------------------------------------------------- persistence
    def save(self, path):
        with open(path, 'w') as fh:
            json.dump({'mean': self.mean, 'std': self.std, 'w': self.w, 'b': self.b,
                       'W': self.geo.W, 'H': self.geo.H,
                       'bounds': self.bounds,
                       'train_report': self.train_report}, fh, indent=2)

    def load(self, path):
        if not os.path.exists(path):
            return False
        with open(path) as fh:
            d = json.load(fh)
        if abs(d.get('W', 0) - self.geo.W) > 0.5 or abs(d.get('H', 0) - self.geo.H) > 0.5:
            print('[GAZE] Gespeicherter Klassifikator passt nicht zur Fensterauflösung.')
            return False
        self.mean, self.std, self.w, self.b = d['mean'], d['std'], d['w'], d['b']
        self.bounds = d.get('bounds', None)
        self.train_report = d.get('train_report', {})
        return self.w is not None


# ==============================================================================
# -- Hysterese / Glance-Erkennung ----------------------------------------------
# ==============================================================================


class GlanceDetector(object):
    """Zustandsautomat über die rohe AOI-Folge.

    ``update(t, aoi_raw)`` gibt (aoi_state, finished_segments) zurück.
    ``finished_segments`` ist eine Liste von dicts {aoi, start, end, duration}.
    """

    def __init__(self, min_dwell_s=0.1, no_face_bridge_s=0.5):
        self.min_dwell = min_dwell_s
        self.bridge = no_face_bridge_s
        self.state = AOI_UNKNOWN
        self.state_start = None
        self._cand = None
        self._cand_t = None

    def update(self, t, aoi):
        done = []
        if self.state_start is None:
            self.state, self.state_start = aoi, t
            return self.state, done

        if aoi == self.state:
            self._cand = None
            return self.state, done

        if aoi != self._cand:
            self._cand, self._cand_t = aoi, t

        need = self.bridge if aoi == AOI_NOFACE else self.min_dwell
        if t - self._cand_t >= need:
            done.append(self._segment(self.state, self.state_start, self._cand_t))
            self.state, self.state_start = self._cand, self._cand_t
            self._cand = None
        return self.state, done

    def flush(self, t_end):
        if self.state_start is None:
            return []
        seg = self._segment(self.state, self.state_start, t_end)
        self.state_start = None
        self.state = AOI_UNKNOWN
        return [seg]

    @staticmethod
    def _segment(aoi, start, end):
        return {'aoi': aoi, 'start': start, 'end': end, 'duration': end - start}


# ==============================================================================
# -- Logger --------------------------------------------------------------------
# ==============================================================================


class GazeRecorder(object):
    """Klassifiziert alle Samples eines Runs und schreibt:

    * ``<prefix>_gaze.csv``    – jedes WebGazer-Sample (≈15–30 Hz)
    * ``<prefix>_glances.csv`` – jede AOI-Episode mit exaktem Start/Ende
    """

    def __init__(self, classifier, cfg, prefix):
        self.clf = classifier
        self.cfg = cfg
        self.detector = GlanceDetector(cfg['min_dwell_s'], cfg['no_face_bridge_s'])
        self.state = AOI_UNKNOWN
        self.last_raw = AOI_UNKNOWN
        self.last_p = None
        self.glance_id = 0              # laufende Nr. der Headunit-Glances
        self.in_hu_glance = False
        self._ctx_t = collections.deque(maxlen=4000)
        self._ctx = collections.deque(maxlen=4000)
        self.t0 = None

        self._gaze_fh = open(prefix + '_gaze.csv', 'w', newline='')
        self._gaze = csv.writer(self._gaze_fh)
        self._gaze.writerow(['unix_time', 't_rel_s', 'latency_ms', 'pos_x', 'sign_idx',
                             'gaze_x', 'gaze_y', 'face', 'head_yaw', 'head_pitch', 'iod_px',
                             'p_headunit', 'aoi_raw', 'aoi_state', 'hu_glance_id'])
        self._gl_fh = open(prefix + '_glances.csv', 'w', newline='')
        self._gl = csv.writer(self._gl_fh)
        self._gl.writerow(['aoi', 'hu_glance_id', 'start_unix', 'end_unix', 'start_rel_s',
                           'end_rel_s', 'duration_s', 'start_pos_x', 'end_pos_x',
                           'start_sign_idx', 'below_min_duration'])
        self.n_samples = 0

    def set_context(self, t, pos_x, sign_idx):
        """Vom Hauptloop pro Frame: aktuelle Fahrzeugposition (für Zuordnung)."""
        if self.t0 is None:
            self.t0 = t
        self._ctx_t.append(t)
        self._ctx.append((pos_x, sign_idx))

    def _ctx_at(self, t):
        if not self._ctx_t:
            return ('', '')
        i = bisect.bisect_left(self._ctx_t, t)
        i = min(max(i, 0), len(self._ctx_t) - 1)
        px, si = self._ctx[i]
        return (round(px, 2), si)

    def process(self, samples):
        for s in samples:
            if self.t0 is None:
                self.t0 = s.t
            aoi_raw, p = self.clf.classify(s)
            state, segments = self.detector.update(s.t, aoi_raw)
            for seg in segments:
                self._write_segment(seg)
            if state == AOI_HU and not self.in_hu_glance:
                self.in_hu_glance = True
                self.glance_id += 1
            elif state != AOI_HU:
                self.in_hu_glance = False
            self.state, self.last_raw, self.last_p = state, aoi_raw, p

            pos_x, sign_idx = self._ctx_at(s.t)
            self._gaze.writerow([
                '%.4f' % s.t, '%.4f' % (s.t - self.t0),
                round((s.t_recv - s.t) * 1000.0, 1), pos_x, sign_idx,
                _r(s.x, 1), _r(s.y, 1), int(s.face), _r(s.yaw, 4), _r(s.pitch, 4),
                _r(s.iod, 1), _r(p, 3), aoi_raw, state,
                self.glance_id if self.in_hu_glance else 0,
            ])
            self.n_samples += 1

    def _write_segment(self, seg):
        if seg['aoi'] == AOI_UNKNOWN:
            return
        sp, si = self._ctx_at(seg['start'])
        ep, _ = self._ctx_at(seg['end'])
        gid = self.glance_id if seg['aoi'] == AOI_HU else ''
        self._gl.writerow([
            seg['aoi'], gid, '%.4f' % seg['start'], '%.4f' % seg['end'],
            '%.4f' % (seg['start'] - self.t0), '%.4f' % (seg['end'] - self.t0),
            '%.4f' % seg['duration'], sp, ep, si,
            int(seg['duration'] < self.cfg['min_glance_s']),
        ])
        if seg['aoi'] == AOI_HU:
            print('[GAZE] Headunit-Glance #%s: %.2f s (x=%s m)' % (gid, seg['duration'], sp))
        elif seg['aoi'] == AOI_OFF:
            print('[GAZE] Blick abgewandt (Offscreen): %.2f s (x=%s m)' % (seg['duration'], sp))

    def close(self, t_end=None):
        if self._gaze_fh is None:
            return
        if t_end is not None:
            for seg in self.detector.flush(t_end):
                self._write_segment(seg)
        self._gaze_fh.close()
        self._gl_fh.close()
        self._gaze_fh = None
        print('[GAZE] Gaze-Logs geschlossen (%d Samples).' % self.n_samples)


def _r(v, nd):
    return '' if v is None else round(v, nd)

