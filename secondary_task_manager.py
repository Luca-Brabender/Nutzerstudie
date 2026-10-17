#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Secondary Task Manager, HUD Kachel & AAOS Task Completion Tracker für LCT Studien
==================================================================================
Implementiert Idee 1:
- Multimodale Aufgabenankündigung für Android Automotive ("Ding"-Chime + obere HUD-Kachel)
- Vollautomatische Erfassung des App-Starts über ADB logcat (Sub-Millisekunden-Genauigkeit)
- Messung der Task Completion Time (TCT) vom Prompt bis zum Öffnen der App
- Visuelles Feedback im Simulator (grüne Kachel: "✓ Erledigt: Kamera (2.4s)")
- Automatisches Zurückkehren zur Menü-App nach 2 Sekunden
- Export einer wissenschaftlichen Auswertungstabelle in <run_prefix>_tasks.csv
- Baseline-Lauf (Referenzfahrt nach ISO 26022) bleibt standardmäßig aufgabenfrei.
"""

import csv
import datetime
import math
import os
import random
import re
import struct
import subprocess
import threading
import time
import wave

try:
    import pygame
except ImportError:
    pygame = None


# Zuordnung der deutschen App-Namen zu den tatsächlichen Android Automotive Paketnamen
APP_PACKAGE_MAP = {
    "kamera": "com.android.camera2",
    "kalender": "com.android.calendar",
    "radio": "com.android.car.radio",
    "telefon": "com.android.car.dialer",
    "kontakte": "com.android.contacts",
    "uhr": "com.android.deskclock",
    "wecker": "com.android.deskclock",
    "galerie": "com.android.gallery3d",
    "einstellungen": "com.android.car.settings",
    "nachrichten": "com.android.car.messenger",
    "medien": "com.android.car.media",
    "rechner": "com.android.calculator2",  # Nur falls nachträglich installiert
}

DEFAULT_SECONDARY_APPS = [
    "Kamera",
    "Kalender",
    "Radio",
    "Telefon",
    "Kontakte",
    "Uhr",
    "Galerie",
    "Einstellungen",
    "Nachrichten",
]

def generate_trigger_positions(start_x=225.0, interval=300.0, max_x=2750.0):
    """Generiert gleichmäßig verteilte Streckenpunkte zwischen den LCT-Schildern.

    Standard: start_x=225.0, interval=300.0 -> 9 Aufgaben bei [225, 525, 825, ..., 2625]m.
    Mit interval=450.0 -> 6 Aufgaben bei [225, 675, 1125, 1575, 2025, 2475]m.
    """
    positions = []
    x = float(start_x)
    while x <= max_x:
        positions.append(round(x, 1))
        x += float(interval)
    return positions


# Standard-Auslösepunkte auf der 3000m-LCT-Strecke (jeweils mittig zwischen den Schildern)
# Schilder befinden sich bei: 150, 300, 450, 600, 750, 900, 1050, 1200, 1350, 1500, 1650, 1800, 1950, 2100, 2250, 2400, 2550, 2700 m
# 9 Triggern bei je 300 m Abstand (ca. alle 18 Sekunden bei 60 km/h):
DEFAULT_TRIGGER_POSITIONS = generate_trigger_positions(225.0, 300.0)


def ensure_chime_asset(asset_path=None):
    """Stellt sicher, dass die Audio-Datei für den Benachrichtigungs-Ton ("Ding") existiert.
    Wird bei Bedarf on-the-fly synthetisiert (16-bit PCM WAV, 44.1 kHz, obertonreicher Glockenton)."""
    if asset_path is None:
        asset_path = os.path.join(os.path.dirname(__file__), 'assets', 'chime.wav')

    if os.path.exists(asset_path) and os.path.getsize(asset_path) > 1000:
        return asset_path

    os.makedirs(os.path.dirname(asset_path), exist_ok=True)
    sample_rate = 44100
    duration = 0.8  # Sekunden
    n_samples = int(sample_rate * duration)

    f_sub = 659.25   # E5
    f0 = 1318.51     # E6
    f1 = 1975.53     # B6
    f2 = 2637.02     # E7

    frames = bytearray()
    for i in range(n_samples):
        t = i / sample_rate
        att = min(1.0, t / 0.006)
        decay_sub = math.exp(-4.5 * t)
        decay0 = math.exp(-5.5 * t)
        decay1 = math.exp(-6.5 * t)
        decay2 = math.exp(-8.0 * t)

        val = (0.20 * math.sin(2 * math.pi * f_sub * t) * decay_sub +
               0.50 * math.sin(2 * math.pi * f0 * t) * decay0 +
               0.20 * math.sin(2 * math.pi * f1 * t) * decay1 +
               0.10 * math.sin(2 * math.pi * f2 * t) * decay2)

        sample = int(32000.0 * att * val)
        sample = max(-32767, min(32767, sample))
        frames.extend(struct.pack('<h', sample))

    with wave.open(asset_path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(frames)

    return asset_path


class TaskHUDCard(object):
    """Dezente Head-Up-Display-Kachel am oberen Bildschirmrand für Sekundäraufgaben."""

    def __init__(self, display_width, display_height):
        self.display_width = display_width
        self.display_height = display_height

        self.font_header = None
        self.font_task = None
        if pygame is not None:
            font_name = 'courier' if os.name == 'nt' else 'mono'
            fonts = [x for x in pygame.font.get_fonts() if font_name in x]
            default_font = 'ubuntumono'
            mono = default_font if default_font in fonts else (fonts[0] if fonts else None)
            mono_path = pygame.font.match_font(mono) if mono else pygame.font.get_default_font()
            self.font_header = pygame.font.Font(mono_path, 12)
            self.font_task = pygame.font.Font(pygame.font.get_default_font(), 22)

        self.card_w = 460
        self.card_h = 68
        self.pos_x = (display_width - self.card_w) // 2
        self.pos_y = 20

        self.app_name = None
        self.task_text = None
        self.total_duration = 0.0
        self.seconds_left = 0.0
        self.active = False
        self.is_completed = False
        self.is_success = True
        self.completion_duration = 0.0

    def show(self, app_name, duration=7.0):
        """Zeigt eine neue Aufgaben-Aufforderung an (z. B. 'Aufgabe: Kamera')."""
        self.app_name = app_name
        self.task_text = f"Aufgabe: {app_name}"
        self.total_duration = float(duration)
        self.seconds_left = float(duration)
        self.active = True
        self.is_completed = False
        self.is_success = True
        self.completion_duration = 0.0

    def show_completed(self, app_name, duration_s, success=True, display_duration=2.0):
        """Aktualisiert die Kachel sofort auf 'Erledigt' mit der gemessenen Bearbeitungszeit."""
        self.app_name = app_name
        self.is_completed = True
        self.is_success = success
        self.completion_duration = duration_s
        if success:
            self.task_text = f"✓ Erledigt: {app_name} ({duration_s:.1f}s)"
        else:
            self.task_text = f"⚠ Abweichende App ({duration_s:.1f}s)"
        self.total_duration = float(display_duration)
        self.seconds_left = float(display_duration)
        self.active = True

    def hide(self):
        self.active = False
        self.seconds_left = 0.0
        self.task_text = None
        self.app_name = None
        self.is_completed = False

    def tick(self, clock):
        if not self.active:
            return
        dt = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - dt)
        if self.seconds_left <= 0.0:
            self.hide()

    def render(self, display):
        if not self.active or self.seconds_left <= 0.0 or pygame is None:
            return

        elapsed = self.total_duration - self.seconds_left
        # Sanftes Einblenden (0.2s) und Ausblenden (letzte 0.5s)
        base_alpha = 230
        if elapsed < 0.2:
            alpha = int(base_alpha * (elapsed / 0.2))
        elif self.seconds_left < 0.5:
            alpha = int(base_alpha * (self.seconds_left / 0.5))
        else:
            alpha = base_alpha
        alpha = max(0, min(255, alpha))

        card_surf = pygame.Surface((self.card_w, self.card_h), pygame.SRCALPHA)

        # Farbschema: Grün bei Erfolg, Amber bei Fehler, Cyan während laufender Aufgabe
        if self.is_completed:
            if self.is_success:
                bg_color = (16, 34, 22, alpha)
                border_color = (46, 204, 113, alpha)   # Emerald Green
                header_text = "AUFGABE ABGESCHLOSSEN"
            else:
                bg_color = (36, 24, 14, alpha)
                border_color = (243, 156, 18, alpha)   # Amber
                header_text = "ABWEICHENDE APP GEÖFFNET"
        else:
            bg_color = (18, 22, 32, alpha)
            border_color = (0, 210, 255, alpha)       # Automotive Cyan
            header_text = "SEKUNDÄRAUFGABE (ANDROID AUTOMOTIVE)"

        # Abgerundete Kachel mit transluzentem dunklen Hintergrund
        pygame.draw.rect(card_surf, bg_color, (0, 0, self.card_w, self.card_h), border_radius=12)

        # Akzentrahmen
        pygame.draw.rect(card_surf, border_color, (0, 0, self.card_w, self.card_h), width=2, border_radius=12)

        # Linker Akzent-Pill-Streifen
        pygame.draw.rect(card_surf, border_color, (14, 14, 5, self.card_h - 28), border_radius=3)

        # Subtiler Header / Kategorie
        if self.font_header is not None:
            sub_surf = self.font_header.render(header_text, True, border_color[:3])
            sub_surf.set_alpha(alpha)
            card_surf.blit(sub_surf, (28, 12))

        # Hauptanweisung / Bestätigungstext
        if self.font_task is not None and self.task_text:
            text_color = (255, 255, 255)
            title_surf = self.font_task.render(self.task_text, True, text_color)
            title_surf.set_alpha(alpha)
            card_surf.blit(title_surf, (28, 31))

        # Dezenter Fortschrittsbalken an der Unterkante der Kachel
        if self.total_duration > 0:
            progress = max(0.0, min(1.0, self.seconds_left / self.total_duration))
            bar_w = int((self.card_w - 28) * progress)
            if bar_w > 0:
                bar_color = (border_color[0], border_color[1], border_color[2], int(alpha * 0.75))
                pygame.draw.rect(card_surf, bar_color, (14, self.card_h - 6, bar_w, 2), border_radius=1)

        display.blit(card_surf, (self.pos_x, self.pos_y))


class AAOSTaskTracker(object):
    """Überwacht über ADB das Android Automotive OS auf dem Raspberry Pi 4.
    Erkennt vollautomatisch den Start von Apps, misst die Bearbeitungszeit (TCT)
    und setzt nach kurzer Zeit den Bildschirm wieder auf die Menü-App zurück."""

    IGNORE_PACKAGES = {
        "com.example.menuapp",
        "com.android.systemui",
        "com.android.car.carlauncher",
    }

    def __init__(
        self,
        device_ip="192.168.0.103",
        port=5555,
        on_completed_callback=None,
        output_csv=None,
        method_name="Unknown",
        participant="P00",
        step=0,
    ):
        self.device_ip = device_ip
        self.port = int(port)
        self.device_serial = f"{self.device_ip}:{self.port}"
        self.on_completed_callback = on_completed_callback
        self.output_csv = output_csv
        self.method_name = method_name
        self.participant = participant
        self.step = step

        self.active_task = None
        self.active_task_idx = 0
        self.task_start_time = 0.0
        self.trigger_x = 0.0

        self.completed_tasks = []

        self._running = False
        self._thread = None
        self._adb_process = None
        self._lock = threading.Lock()

        self._csv_writer = None
        self._csv_file = None
        if self.output_csv:
            try:
                os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
                self._csv_file = open(self.output_csv, 'w', newline='', encoding='utf-8')
                self._csv_writer = csv.writer(self._csv_file)
                self._csv_writer.writerow([
                    'timestamp_unix', 'iso_time', 'participant', 'step', 'method',
                    'task_idx', 'target_app', 'target_pkg', 'opened_pkg',
                    'trigger_x', 'duration_s', 'success'
                ])
                self._csv_file.flush()
            except Exception as e:
                print(f"[AAOS TRACKER] Fehler beim Erstellen von {self.output_csv}: {e}")

    def start(self):
        """Startet den Logcat-Hintergrund-Listener."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._logcat_worker, name="AAOSTaskWatcher", daemon=True)
        self._thread.start()

    def stop(self):
        """Beendet den Listener und schließt offene Dateien."""
        self._running = False
        if self._adb_process is not None:
            try:
                self._adb_process.kill()
            except Exception:
                pass
            self._adb_process = None

        if self._csv_file is not None:
            try:
                self._csv_file.flush()
                self._csv_file.close()
            except Exception:
                pass
            self._csv_file = None

    def set_active_task(self, task_idx, app_name, trigger_x, start_time):
        """Wird aufgerufen, wenn in CARLA eine neue Sekundäraufgabe getriggert wird."""
        with self._lock:
            self.active_task = app_name
            self.active_task_idx = task_idx
            self.trigger_x = trigger_x
            self.task_start_time = start_time

    def clear_active_task(self):
        with self._lock:
            self.active_task = None
            self.task_start_time = 0.0

    def _extract_package(self, line):
        """Extrahiert den Paketnamen aus Android ActivityManager Logzeilen."""
        # Format 1: ActivityTaskManager: START u0 {cmp=com.android.camera2/.CameraActivity}
        m = re.search(r'cmp=([a-zA-Z0-9_\.]+)/', line)
        if m:
            return m.group(1)
        # Format 2: am_create_activity / am_resume_activity: [0,hash,com.android.camera2/.CameraActivity,...]
        m = re.search(r'am_(?:create|resume)_activity:\s*\[\d+,\s*\d+,\s*([a-zA-Z0-9_\.]+)/', line)
        if m:
            return m.group(1)
        return None

    def _logcat_worker(self):
        """Liest im Hintergrund den ADB Event-Stream auf Activity-Starts."""
        cmd = [
            "adb", "-s", self.device_serial, "shell",
            "logcat", "-v", "brief", "-b", "events", "-T", "1",
            "am_create_activity:I", "am_resume_activity:I", "*:S"
        ]
        try:
            self._adb_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1
            )
            for raw_line in iter(self._adb_process.stdout.readline, ''):
                if not self._running:
                    break
                line = raw_line.strip()
                if not line:
                    continue

                pkg = self._extract_package(line)
                if pkg and pkg not in self.IGNORE_PACKAGES:
                    self._on_package_started(pkg)
        except Exception as e:
            if self._running:
                print(f"[AAOS TRACKER] Logcat Worker beendet/Fehler: {e}")
        finally:
            if self._adb_process is not None:
                try:
                    self._adb_process.kill()
                except Exception:
                    pass

    def _on_package_started(self, pkg):
        """Wird aufgerufen, sobald auf Android Automotive eine App geöffnet wurde."""
        now = time.time()
        with self._lock:
            if not self.active_task or self.task_start_time <= 0:
                return

            target_app = self.active_task
            task_idx = self.active_task_idx
            t_start = self.task_start_time
            trig_x = self.trigger_x
            method = self.method_name

            # Aufgabe als aktiv beenden
            self.active_task = None
            self.task_start_time = 0.0

        duration_s = max(0.01, now - t_start)

        # Prüfung auf Korrektheit der geöffneten App
        expected_pkg = APP_PACKAGE_MAP.get(target_app.lower(), "").lower()
        success = (expected_pkg in pkg.lower()) or (target_app.lower() in pkg.lower())

        # In CSV schreiben
        iso_str = datetime.datetime.now().isoformat()
        if self._csv_writer and self._csv_file:
            try:
                self._csv_writer.writerow([
                    f"{now:.3f}", iso_str, self.participant, self.step, method,
                    task_idx, target_app, expected_pkg, pkg,
                    f"{trig_x:.1f}", f"{duration_s:.3f}", str(success).upper()
                ])
                self._csv_file.flush()
            except Exception as e:
                print(f"[AAOS TRACKER] Fehler beim Schreiben der CSV: {e}")

        # In Historie speichern
        self.completed_tasks.append({
            'task_idx': task_idx,
            'app': target_app,
            'target_pkg': expected_pkg,
            'opened_pkg': pkg,
            'duration_s': duration_s,
            'success': success,
            'time': now
        })

        status_symbol = "✓" if success else "⚠"
        print(f"\n🎯 [TASK COMPLETED] Aufgabe {task_idx} ({target_app}) in {duration_s:.2f}s abgeschlossen! {status_symbol} (App: {pkg}, Erfolg: {success})")

        # Callback an SecondaryTaskManager (HUD grün schalten + events.csv loggen)
        if self.on_completed_callback:
            try:
                self.on_completed_callback(task_idx, target_app, pkg, duration_s, success)
            except Exception as e:
                print(f"[AAOS TRACKER] Fehler im Callback: {e}")

        # Automatischer Reset zurück zur Menü-App nach 2.0 Sekunden
        threading.Thread(target=self._auto_return_to_menu, daemon=True).start()

    def _auto_return_to_menu(self, delay=2.0):
        """Bringt nach kurzer Zeit die Menü-App wieder in den Vordergrund,
        damit der Proband für die nächste Aufgabe bereit ist."""
        time.sleep(delay)
        if not self._running:
            return
        cmd = [
            "adb", "-s", self.device_serial, "shell",
            "am", "start", "-n", "com.example.menuapp/.MainActivity"
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=4)
        except Exception:
            pass


class SecondaryTaskManager(object):
    """Steuert die Auslösung und Verwaltung der Sekundäraufgaben während der LCT-Fahrt."""

    def __init__(
        self,
        method,
        hud=None,
        events=None,
        participant="P00",
        step=0,
        trigger_positions=None,
        app_list=None,
        duration=7.0,
        disabled=False,
        force_enable=False,
        device_ip="192.168.0.103",
        no_aaos=False,
        run_prefix=None,
        trigger_interval=300.0,
        trigger_start=225.0,
        locations_str=None,
    ):
        self.hud = hud
        self.events = events
        self.participant = participant
        self.step = step
        self.duration = float(duration)
        self.method_value = getattr(method, 'value', str(method))
        self.device_ip = device_ip
        self.no_aaos = no_aaos
        self.run_prefix = run_prefix

        # Baseline nach ISO 26022 ist standardmäßig ohne Sekundäraufgabe (außer bei force_enable für Tests)
        is_baseline = (self.method_value.lower() == 'baseline') and (not force_enable)
        self.is_active = (not disabled) and (not is_baseline)

        # Streckenpunkte und Apps vorbereiten
        if trigger_positions is not None:
            self.trigger_positions = list(trigger_positions)
        elif locations_str:
            self.trigger_positions = [float(x.strip()) for x in locations_str.split(',') if x.strip()]
        else:
            self.trigger_positions = generate_trigger_positions(start_x=trigger_start, interval=trigger_interval)

        base_apps = list(app_list or DEFAULT_SECONDARY_APPS)

        # Reproduzierbare Pseudozufalls-Reihenfolge der Apps pro Proband & Schritt
        rng = random.Random(f"{participant}_{step}_secondary_task_seed")
        shuffled_apps = list(base_apps)
        rng.shuffle(shuffled_apps)

        while len(shuffled_apps) < len(self.trigger_positions):
            extra = list(base_apps)
            rng.shuffle(extra)
            shuffled_apps.extend(extra)

        self.tasks = []
        for i, pos_x in enumerate(self.trigger_positions):
            app = shuffled_apps[i]
            self.tasks.append((float(pos_x), app))

        self.triggered_indices = set()
        self.current_task_idx = -1
        self.current_app = None

        # Audio-Initialisierung ("Ding"-Chime)
        self.chime_path = ensure_chime_asset()
        self.chime_sound = None
        if self.is_active and pygame is not None:
            try:
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                if os.path.exists(self.chime_path):
                    self.chime_sound = pygame.mixer.Sound(self.chime_path)
                    self.chime_sound.set_volume(0.85)
            except Exception as e:
                print(f"[SECONDARY TASK] Signalton konnte nicht geladen werden: {e}")

        # AAOS Task Tracker initialisieren (Idee 1: Vollautomatische Zeiterfassung via ADB)
        self.tracker = None
        if self.is_active and not self.no_aaos:
            task_csv = f"{run_prefix}_tasks.csv" if run_prefix else None
            self.tracker = AAOSTaskTracker(
                device_ip=self.device_ip,
                on_completed_callback=self._on_task_completed,
                output_csv=task_csv,
                method_name=self.method_value,
                participant=self.participant,
                step=self.step,
            )
            self.tracker.start()

        if self.is_active:
            task_preview = ", ".join([f"{x:.0f}m: {app}" for x, app in self.tasks[:4]])
            print(f"[SECONDARY TASK] Aktiv für '{self.method_value}': {len(self.tasks)} Aufgaben geplant (z. B. {task_preview}...)")
            if self.tracker is not None:
                print(f"[SECONDARY TASK] Automatische TCT-Zeiterfassung aktiv (Speicherung in: {run_prefix}_tasks.csv)")
        else:
            print(f"[SECONDARY TASK] Inaktiv für '{self.method_value}' (keine Aufgaben während dieser Fahrt).")

    def update(self, cur_x, now=None):
        """Wird in jedem Frame des Simulators aufgerufen.
        Prüft, ob ein Streckenpunkt erreicht wurde und löst Ton + HUD-Kachel aus."""
        if not self.is_active or cur_x < 0.0:
            return None

        for idx, (trig_x, app_name) in enumerate(self.tasks):
            if idx not in self.triggered_indices and cur_x >= trig_x:
                self.triggered_indices.add(idx)
                self.current_task_idx = idx
                self.current_app = app_name
                self._trigger_task(idx, trig_x, app_name, cur_x)
                return app_name

        return None

    def _trigger_task(self, idx, trig_x, app_name, cur_x):
        """Löst den Signalton ("Ding"), die HUD-Kachel und das Event-Logging aus."""
        t_start = time.time()

        # 1. Aufmerksamkeits-Ton ("Ding") abspielen
        if self.chime_sound is not None:
            try:
                self.chime_sound.play()
            except Exception as e:
                print(f"[SECONDARY TASK] Fehler beim Abspielen des Signaltons: {e}")

        # 2. Head-Up-Display Kachel oben im CARLA-Simulator einblenden
        if self.hud is not None and hasattr(self.hud, 'show_task_prompt'):
            self.hud.show_task_prompt(app_name, duration=self.duration)

        # 3. Tracker scharf schalten (startet die TCT-Messung bis zum App-Start)
        if self.tracker is not None:
            self.tracker.set_active_task(
                task_idx=idx + 1,
                app_name=app_name,
                trigger_x=trig_x,
                start_time=t_start
            )

        # 4. Wissenschaftliches Event-Logging in events.csv
        if self.events is not None:
            self.events.log(
                'secondary_task_prompt',
                f'task_idx={idx + 1}, app={app_name}, planned_x={trig_x:.1f}, cur_x={cur_x:.2f}'
            )

        # 5. Terminal-Ausgabe
        total_tasks = len(self.tasks)
        print(f"\n🔔 [AUFGABE {idx + 1}/{total_tasks}] Bei x={cur_x:.1f}m getriggert: 'Aufgabe: {app_name}'")

    def _on_task_completed(self, task_idx, app_name, opened_pkg, duration_s, success):
        """Callback: Wird von AAOSTaskTracker aufgerufen, sobald die App auf AAOS startet."""
        # 1. HUD Kachel sofort auf grün schalten ("✓ Erledigt: Kamera (2.4s)")
        if self.hud is not None and hasattr(self.hud, 'show_task_completed'):
            self.hud.show_task_completed(app_name, duration_s, success=success, display_duration=2.0)

        # 2. In events.csv protokollieren
        if self.events is not None:
            self.events.log(
                'secondary_task_completed',
                f'task_idx={task_idx}, app={app_name}, opened={opened_pkg}, tct_s={duration_s:.3f}, success={success}'
            )

    def get_active_task_status(self):
        """Gibt (task_active: int, task_target: str) für das 60-FPS-Fahr-Log zurück."""
        if not self.is_active:
            return 0, "None"
        if self.tracker is not None and getattr(self.tracker, 'active_task', None):
            return 1, self.tracker.active_task
        if self.hud is not None and hasattr(self.hud, 'task_card'):
            card = self.hud.task_card
            if getattr(card, 'active', False) and not getattr(card, 'is_completed', False):
                return 1, card.app_name or "Unknown"
        return 0, "None"

    def reset(self):
        """Setzt den Aufgabenstatus zurück und schließt den Tracker sauber ab."""
        self.triggered_indices.clear()
        self.current_task_idx = -1
        self.current_app = None
        if self.tracker is not None:
            self.tracker.stop()
            self.tracker = None
        if self.hud is not None and hasattr(self.hud, 'clear_task_prompt'):
            self.hud.clear_task_prompt()
