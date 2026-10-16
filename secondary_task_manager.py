#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Secondary Task Manager & Head-Up Display (HUD) Kachel für LCT Studien
====================================================================
Implementiert Idee A: Multimodale Aufgabenankündigung für Android Automotive
- Aufmerksamkeits-Signalton ("Ding"-Chime) beim Erreichen des Streckenpunkts
- Dezente Head-Up-Display-Kachel oben zentriert im CARLA-Simulator: "Aufgabe: <App>"
- Automatische Protokollierung aller Aufgabentrigger in der events.csv
- Baseline-Lauf (Referenzfahrt nach ISO 26022) bleibt garantiert aufgabenfrei.
"""

import math
import os
import random
import struct
import time
import wave

try:
    import pygame
except ImportError:
    pygame = None


# Standard-Apps auf Android Automotive (menuapp auf dem Raspberry Pi 4B)
DEFAULT_SECONDARY_APPS = [
    "Kamera",
    "Kalender",
    "Radio",
    "Telefon",
    "Kontakte",
    "Uhr",
    "Galerie",
    "Einstellungen",
    "Rechner",
]

# Standard-Auslösepunkte auf der 3000m-LCT-Strecke (jeweils mittig zwischen den Schildern)
# Schilder befinden sich bei: 150, 300, 450, 600, 750, 900, 1050, 1200, 1350, 1500, 1650, 1800, 1950, 2100, 2250, 2400, 2550, 2700 m
# 9 Triggern bei je 300 m Abstand (ca. alle 18 Sekunden bei 60 km/h):
DEFAULT_TRIGGER_POSITIONS = [
    225.0,
    525.0,
    825.0,
    1125.0,
    1425.0,
    1725.0,
    2025.0,
    2325.0,
    2625.0,
]


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

    # Akustisch harmonische Frequenzen (E5 Grundkörper + E6 Klarheit + B6 Quinte + E7 Brillanz)
    f_sub = 659.25   # E5
    f0 = 1318.51     # E6
    f1 = 1975.53     # B6
    f2 = 2637.02     # E7

    frames = bytearray()
    for i in range(n_samples):
        t = i / sample_rate
        # 6ms weicher Einschwingvorgang (verhindert Knacken)
        att = min(1.0, t / 0.006)
        # Physikalisch natürliches Abklingen höherer Obertöne
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

    def show(self, app_name, duration=7.0):
        self.app_name = app_name
        self.task_text = f"Aufgabe: {app_name}"
        self.total_duration = float(duration)
        self.seconds_left = float(duration)
        self.active = True

    def hide(self):
        self.active = False
        self.seconds_left = 0.0
        self.task_text = None
        self.app_name = None

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
        # Sanftes Einblenden (0.25s) und Ausblenden (letzte 0.75s)
        base_alpha = 230
        if elapsed < 0.25:
            alpha = int(base_alpha * (elapsed / 0.25))
        elif self.seconds_left < 0.75:
            alpha = int(base_alpha * (self.seconds_left / 0.75))
        else:
            alpha = base_alpha
        alpha = max(0, min(255, alpha))

        card_surf = pygame.Surface((self.card_w, self.card_h), pygame.SRCALPHA)

        # Abgerundete Kachel mit transluzentem dunklen Hintergrund
        bg_color = (18, 22, 32, alpha)
        pygame.draw.rect(card_surf, bg_color, (0, 0, self.card_w, self.card_h), border_radius=12)

        # Moderner Automotive-Cyan Akzentrahmen
        border_color = (0, 210, 255, alpha)
        pygame.draw.rect(card_surf, border_color, (0, 0, self.card_w, self.card_h), width=2, border_radius=12)

        # Linker Akzent-Pill-Streifen
        pygame.draw.rect(card_surf, border_color, (14, 14, 5, self.card_h - 28), border_radius=3)

        # Subtiler Header / Kategorie
        if self.font_header is not None:
            sub_surf = self.font_header.render("SEKUNDÄRAUFGABE (ANDROID AUTOMOTIVE)", True, (0, 210, 255))
            sub_surf.set_alpha(alpha)
            card_surf.blit(sub_surf, (28, 12))

        # Hauptanweisung ("Aufgabe: Kamera")
        if self.font_task is not None and self.task_text:
            title_surf = self.font_task.render(self.task_text, True, (255, 255, 255))
            title_surf.set_alpha(alpha)
            card_surf.blit(title_surf, (28, 31))

        # Dezenter schmaler Countdown-Balken an der Unterkante der Kachel
        if self.total_duration > 0:
            progress = max(0.0, min(1.0, self.seconds_left / self.total_duration))
            bar_w = int((self.card_w - 28) * progress)
            if bar_w > 0:
                bar_color = (0, 210, 255, int(alpha * 0.75))
                pygame.draw.rect(card_surf, bar_color, (14, self.card_h - 6, bar_w, 2), border_radius=1)

        display.blit(card_surf, (self.pos_x, self.pos_y))


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
    ):
        self.hud = hud
        self.events = events
        self.participant = participant
        self.step = step
        self.duration = float(duration)
        self.method_value = getattr(method, 'value', str(method))

        # Baseline nach ISO 26022 ist standardmäßig ohne Sekundäraufgabe (außer bei force_enable für Tests)
        is_baseline = (self.method_value.lower() == 'baseline') and (not force_enable)
        self.is_active = (not disabled) and (not is_baseline)

        # Streckenpunkte und Apps vorbereiten
        self.trigger_positions = list(trigger_positions or DEFAULT_TRIGGER_POSITIONS)
        base_apps = list(app_list or DEFAULT_SECONDARY_APPS)

        # Reproduzierbare Pseudozufalls-Reihenfolge der Apps pro Proband & Schritt
        # (Jede App kommt genau einmal vor, Reihenfolge ist pro Schritt permutiert)
        rng = random.Random(f"{participant}_{step}_secondary_task_seed")
        shuffled_apps = list(base_apps)
        rng.shuffle(shuffled_apps)

        # Falls mehr Triggern als Apps vorhanden sind, zyklisch auffüllen
        while len(shuffled_apps) < len(self.trigger_positions):
            extra = list(base_apps)
            rng.shuffle(extra)
            shuffled_apps.extend(extra)

        # Aufgabenliste zusammenstellen: [(x_trigger, app_name), ...]
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

        if self.is_active:
            task_preview = ", ".join([f"{x:.0f}m: {app}" for x, app in self.tasks[:4]])
            print(f"[SECONDARY TASK] Aktiv für '{self.method_value}': {len(self.tasks)} Aufgaben geplant (z. B. {task_preview}...)")
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
        # 1. Aufmerksamkeits-Ton ("Ding") abspielen
        if self.chime_sound is not None:
            try:
                self.chime_sound.play()
            except Exception as e:
                print(f"[SECONDARY TASK] Fehler beim Abspielen des Signaltons: {e}")

        # 2. Head-Up-Display Kachel oben im CARLA-Simulator einblenden
        if self.hud is not None and hasattr(self.hud, 'show_task_prompt'):
            self.hud.show_task_prompt(app_name, duration=self.duration)

        # 3. Wissenschaftliches Event-Logging in events.csv
        if self.events is not None:
            self.events.log(
                'secondary_task_prompt',
                f'task_idx={idx + 1}, app={app_name}, planned_x={trig_x:.1f}, cur_x={cur_x:.2f}'
            )

        # 4. Terminal-Ausgabe
        total_tasks = len(self.tasks)
        print(f"\n🔔 [AUFGABE {idx + 1}/{total_tasks}] Bei x={cur_x:.1f}m getriggert: 'Aufgabe: {app_name}'")

        # HINWEIS: Hier kann später direkt die Sprachausgabe (TTS / pyttsx3 / Audioclip)
        # angeschlossen werden ("Öffne <App>").

    def reset(self):
        """Setzt den Aufgabenstatus zurück."""
        self.triggered_indices.clear()
        self.current_task_idx = -1
        self.current_app = None
        if self.hud is not None and hasattr(self.hud, 'clear_task_prompt'):
            self.hud.clear_task_prompt()

