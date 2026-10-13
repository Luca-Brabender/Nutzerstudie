# gaze_calibration.py
"""Augen- und Kopfkalibrierung für den Fahrsimulator (Simulator vs. Headunit).

Ablauf
------
1. Warten auf Browser-Verbindung (http://localhost:8000/gaze_tracker.html).
2. Schritt 1 (Augen-Kalibrierung Simulator):
   Der Proband hält den Kopf geradeaus und folgt einem roten Punkt NUR MIT DEN AUGEN
   (Mitte -> Linke Spur -> Rechte Spur -> Oben/Schilder -> Mitte).
   Dadurch trainiert WebGazer die Pupillen- und Blickbewegungen über die gesamte
   Breite des Simulator-Bildschirms.
3. Schritt 2 (Headunit):
   Der Proband schaut ganz natürlich auf die Headunit (Android Automotive).
   Hierbei werden die charakteristische Kopfneigung/Kopfdrehung und der Blick
   nach unten-rechts erfasst.
4. Live-Test:
   Der Bildschirm zeigt in Echtzeit ein großes Feld an:
   [ SIMULATOR (Straße) ] oder [ HEADUNIT ].
   Versuchsleiter/Proband testen es kurz und bestätigen mit [ENTER].
"""

import array
import math
import time

import pygame

from gaze_aoi import AOI_HU, AOI_SIM, AOI_OFF, AOI_NOFACE, AOI_UNKNOWN

BG = (35, 39, 47)
FG = (255, 255, 255)
GREEN = (40, 180, 90)
RED = (220, 60, 60)
YELLOW = (230, 175, 40)
GRAY = (90, 95, 105)


class _Abort(Exception):
    pass


class GazeCalibration(object):

    def __init__(self, display, gaze_server, geometry, classifier, cfg):
        self.display = display
        self.gaze = gaze_server
        self.geo = geometry
        self.clf = classifier
        self.cfg = cfg
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(pygame.font.get_default_font(), 26)
        self.big_font = pygame.font.Font(pygame.font.get_default_font(), 38)
        self.small = pygame.font.Font(pygame.font.get_default_font(), 18)
        self._beep_hi = _make_beep(880, 0.12)
        self._beep_lo = _make_beep(520, 0.18)
        self.report = {}
        self._pending = []

    # ================================================================== API
    def run(self, allow_reuse=True):
        """Gibt True zurück, wenn Eyetracking einsatzbereit ist, sonst False."""
        try:
            if not self._wait_for_browser():
                return False
            if allow_reuse and self.clf.trained:
                choice = self._menu(
                    ['Eyetracking ist bereits kalibriert.',
                     '',
                     '[ENTER]  Bestehende Kalibrierung verwenden',
                     '[T]      Live-Test durchführen',
                     '[K]      Neu kalibrieren (Augen- & Headunit-Punkte)',
                     '[ESC]    Ohne Eyetracking fortfahren'],
                    {pygame.K_RETURN: 'ok', pygame.K_KP_ENTER: 'ok',
                     pygame.K_t: 'test', pygame.K_k: 'cal'})
                if choice == 'ok':
                    return True
                if choice == 'test':
                    return self._live_test_loop()
            return self._full_calibration()
        except _Abort:
            print('[GAZE] Kalibrierung abgebrochen -> ohne Eyetracking.')
            return False
        finally:
            self.gaze.push_status(phase='Fahrt')

    # ============================================================ phases
    def _wait_for_browser(self):
        url = 'http://localhost:%d/gaze_tracker.html' % self.gaze.http_port
        while True:
            self._events()
            rate = self.gaze.sample_rate()
            ok = self.gaze.connected and rate >= 5
            self._screen(['Eyetracking (WebGazer)',
                          '',
                          'Browser-Tab öffnen: %s' % url,
                          'Verbunden: %s   |   Samplerate: %.0f Hz' %
                          ('ja' if self.gaze.connected else 'nein', rate),
                          '',
                          '[ENTER] weiter zur Kalibrierung' if ok else 'Warte auf Verbindung & Gesichtserkennung ...',
                          '[ESC] ohne Eyetracking fortfahren'])
            for k in self._keys():
                if k == pygame.K_ESCAPE:
                    raise _Abort()
                if k in (pygame.K_RETURN, pygame.K_KP_ENTER) and ok:
                    return True
            self.clock.tick(30)

    def _full_calibration(self):
        while True:
            self.gaze.push_status(phase='Kalibrierung')
            self.gaze.clear_model()
            time.sleep(0.2)

            # Schritt 1: Augen-Kalibrierung auf dem Simulator (Mitte, Links, Rechts, Oben)
            sim_samples = self._record_simulator_eyes()

            # Schritt 2: Blick auf die Headunit (Android Automotive)
            hu_samples = self._record_headunit()

            # Klassifikator trainieren (Augen-Gaze + Kopfpose)
            rep = self.clf.fit(sim_samples, hu_samples)
            if rep is None:
                retry = self._menu(
                    ['[WARNUNG] Zu wenige Blickdaten erfasst.',
                     'War das Gesicht während der Aufnahme gut sichtbar?',
                     '',
                     '[K]   Kalibrierung wiederholen',
                     '[ESC] Ohne Eyetracking fortfahren'],
                    {pygame.K_k: 'cal'})
                if retry == 'cal':
                    continue
                return False

            self.report = rep
            # Interaktiver Live-Test
            ok = self._live_test_loop()
            if ok:
                return True

    def _record_simulator_eyes(self):
        """Führt den Probanden über 4-5 Punkte auf dem Simulator, damit WebGazer

        die Augenbewegungen (Pupillenpositionen) über die gesamte Bildschirmbreite lernt.
        """
        # (Bezeichnung, X-Faktor, Y-Faktor, Dauer in Sekunden)
        points = [
            ("Mitte (Geradeaus / Fahrbahn)", 0.50, 0.50, 2.0),
            ("Links (Linke Spur / Spiegel)", 0.18, 0.50, 2.0),
            ("Rechts (Rechte Spur / Spiegel)", 0.82, 0.50, 2.0),
            ("Oben (Schilder / Horizont)", 0.50, 0.25, 1.8),
            ("Mitte (Zentriert)", 0.50, 0.50, 1.6),
        ]
        total_time = sum(p[3] for p in points)

        self._menu([
            "SCHRITT 1 VON 2: AUGEN-KALIBRIERUNG (SIMULATOR)",
            "",
            "Bitte den Kopf nach vorn zum Monitor gerichtet halten",
            "und dem roten Punkt NUR MIT DEN AUGEN folgen:",
            "  -> Mitte, Links (Spur), Rechts (Spur), Oben (Schilder)",
            "",
            "[LEERTASTE] Starten (ca. 9 Sekunden)"
        ], {pygame.K_SPACE: 'go'})

        self._play(self._beep_hi)
        collected = []
        self.gaze.drain()
        t_global = time.time()

        for pt_idx, (pt_name, fx, fy, dur) in enumerate(points):
            target_x = fx * self.geo.W
            target_y = fy * self.geo.H
            self.gaze.calib_start(target_x, target_y, label='sim_' + pt_name)
            t_pt = time.time()
            settle_s = 0.5  # Einschwingzeit vor jedem Punkt

            while True:
                self._events()
                dt = time.time() - t_pt
                collected.extend(self.gaze.drain())

                if dt >= dur:
                    break

                self.display.fill(BG)

                # Titel und Anweisung
                surf_t = self.font.render("SCHRITT 1: AUGEN-KALIBRIERUNG (SIMULATOR)", True, FG)
                self.display.blit(surf_t, (self.geo.W / 2 - surf_t.get_width() / 2, 45))

                surf_inst = self.font.render("Folgen Sie dem Punkt NUR MIT DEN AUGEN: %s" % pt_name, True, (200, 225, 255))
                self.display.blit(surf_inst, (self.geo.W / 2 - surf_inst.get_width() / 2, 85))

                # Animierter Fixationspunkt (schrumpft am Anfang leicht zur Fixation)
                cx, cy = int(target_x), int(target_y)
                r = int(22 - 12 * min(1.0, dt / settle_s))
                pygame.draw.circle(self.display, RED, (cx, cy), r)
                pygame.draw.circle(self.display, FG, (cx, cy), 3)

                # Fortschritt
                overall_elapsed = time.time() - t_global
                prog_str = "Punkt %d/%d  |  Dauer: %.1fs / %.1fs  |  %d Samples" % (
                    pt_idx + 1, len(points), overall_elapsed, total_time, len(collected))
                surf_prog = self.small.render(prog_str, True, (255, 230, 100))
                self.display.blit(surf_prog, (self.geo.W / 2 - surf_prog.get_width() / 2, self.geo.H - 55))

                pygame.display.flip()
                self.clock.tick(60)

            self.gaze.calib_stop()

        self._play(self._beep_lo)
        time.sleep(0.3)
        return collected

    def _record_headunit(self):
        """Erfasst den natürlichen Blick auf die Headunit (Android Automotive)."""
        self._menu([
            "SCHRITT 2 VON 2: HEADUNIT (ANDROID AUTOMOTIVE)",
            "",
            "Bitte jetzt ganz natürlich auf die Headunit (Android Automotive) schauen",
            "(wie beim Bedienen des Infotainment-Bildschirms).",
            "",
            "[LEERTASTE] Starten (3 Sekunden)"
        ], {pygame.K_SPACE: 'go'})

        self._play(self._beep_hi)
        t0 = time.time()
        collected = []
        self.gaze.drain()
        ref_x = self.geo.W * 0.85
        ref_y = self.geo.H * 1.25
        self.gaze.calib_start(ref_x, ref_y, label='headunit')

        while True:
            self._events()
            dt = time.time() - t0
            collected.extend(self.gaze.drain())

            if dt >= 3.0:
                break

            self.display.fill(BG)
            remaining = max(0.0, 3.0 - dt)

            surf_t = self.font.render("SCHRITT 2: HEADUNIT KALIBRIERUNG", True, FG)
            self.display.blit(surf_t, (self.geo.W / 2 - surf_t.get_width() / 2, 70))

            surf_inst = self.font.render("Blick auf das Headunit-Display gerichtet halten ...", True, (255, 180, 180))
            self.display.blit(surf_inst, (self.geo.W / 2 - surf_inst.get_width() / 2, 120))

            cd_text = "Aufnahme: %.1fs  |  %d Samples" % (remaining, len(collected))
            surf_cd = self.font.render(cd_text, True, (255, 230, 100))
            self.display.blit(surf_cd, (self.geo.W / 2 - surf_cd.get_width() / 2, self.geo.H / 2))

            pygame.display.flip()
            self.clock.tick(60)

        self.gaze.calib_stop()
        self._play(self._beep_lo)
        time.sleep(0.3)
        return collected

    def _live_test_loop(self):
        """Interaktiver Live-Test: Zeigt in Echtzeit an, wohin der Proband blickt."""
        self.gaze.push_status(phase='Live-Test')
        self._keys()  # Vorherige Tasten leeren
        current_aoi = AOI_SIM

        while True:
            self._events()
            for k in self._keys():
                if k in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    print('[GAZE] Kalibrierung durch Benutzer bestätigt!')
                    return True
                if k == pygame.K_k:
                    return False  # Erneut kalibrieren
                if k == pygame.K_ESCAPE:
                    raise _Abort()

            # Neueste Samples holen & klassifizieren
            samples = self.gaze.drain()
            if samples:
                last = samples[-1]
                aoi_raw, p = self.clf.classify(last)
                if aoi_raw in (AOI_SIM, AOI_HU, AOI_OFF, AOI_NOFACE):
                    current_aoi = aoi_raw
                self.gaze.push_status(aoi=current_aoi)

            self.display.fill(BG)

            # Titel
            t1 = self.font.render("KALIBRIERUNG ABGESCHLOSSEN - LIVE-TEST", True, FG)
            self.display.blit(t1, (self.geo.W / 2 - t1.get_width() / 2, 45))

            t2 = self.font.render("Testen Sie: Monitor (Straße), Headunit (AAOS) oder Blick abgewandt (weder noch):", True, (180, 200, 220))
            self.display.blit(t2, (self.geo.W / 2 - t2.get_width() / 2, 90))

            # Großes Status-Feld in der Mitte
            box_w, box_h = 700, 140
            box_x = int(self.geo.W / 2 - box_w / 2)
            box_y = int(self.geo.H / 2 - box_h / 2)

            if current_aoi == AOI_SIM:
                color = GREEN
                label = "BLICK AUF SIMULATOR (Straße)"
            elif current_aoi == AOI_HU:
                color = RED
                label = "BLICK AUF HEADUNIT"
            elif current_aoi == AOI_OFF:
                color = YELLOW
                label = "WEDER MONITOR NOCH HEADUNIT"
            elif current_aoi == AOI_NOFACE:
                color = GRAY
                label = "KEIN GESICHT ERKANNT"
            else:
                color = GRAY
                label = current_aoi

            pygame.draw.rect(self.display, color, (box_x, box_y, box_w, box_h), border_radius=12)
            lbl_surf = self.big_font.render(label, True, FG)
            self.display.blit(lbl_surf, (self.geo.W / 2 - lbl_surf.get_width() / 2, box_y + box_h / 2 - lbl_surf.get_height() / 2))

            # Fußzeile
            footer = self.font.render("[ENTER] Akzeptieren & Fahrt starten    |    [K] Neu kalibrieren", True, (255, 230, 100))
            self.display.blit(footer, (self.geo.W / 2 - footer.get_width() / 2, self.geo.H - 75))

            pygame.display.flip()
            self.clock.tick(60)

    # ================================================================= ui
    def _screen(self, lines):
        self.display.fill(BG)
        h = self.font.get_linesize() + 6
        y0 = self.geo.H / 2 - len(lines) * h / 2
        for i, line in enumerate(lines):
            surf = self.font.render(line, True, FG)
            self.display.blit(surf, (self.geo.W / 2 - surf.get_width() / 2, y0 + i * h))
        pygame.display.flip()

    def _menu(self, lines, keymap):
        self._keys()  # vorher gedrückte Tasten verwerfen
        while True:
            self._screen(lines)
            for k in self._keys():
                if k == pygame.K_ESCAPE:
                    raise _Abort()
                if k in keymap:
                    return keymap[k]
            self.gaze.drain()
            self.clock.tick(30)

    def _events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                raise _Abort()
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    raise _Abort()
                self._pending.append(ev.key)

    def _keys(self):
        self._events()
        keys, self._pending = self._pending, []
        return keys

    @staticmethod
    def _play(sound):
        if sound is not None:
            try:
                sound.play()
            except Exception:
                pass


def _make_beep(freq, dur):
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        rate, size, channels = pygame.mixer.get_init()
        if abs(size) != 16:
            return None
        n = int(rate * dur)
        buf = array.array('h')
        for i in range(n):
            v = int(12000 * math.sin(2 * math.pi * freq * i / rate))
            for _ in range(channels):
                buf.append(v)
        return pygame.mixer.Sound(buffer=buf.tobytes())
    except Exception:
        return None
