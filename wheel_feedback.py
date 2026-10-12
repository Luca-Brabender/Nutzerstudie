# wheel_feedback.py
"""Force-Feedback-Controller fuer Logitech Racing Wheels (G27 / G29) unter Linux.

Unterstuetzt:
1. Hardware-Zentrierfeder (Autocenter-Strength 0-100%) und Gesamt-Gain (0-100%)
   ueber den nativen Linux-Treiber (ffset & ioctl).
2. Lenkwinkel-Konfiguration (z. B. 900 Grad Lock-to-Lock) im Linux-Kernel-Sysfs.
3. Dynamische Kraftmodulation waehrend der Fahrt (Daempfung gegen Schwingungen,
   Kurvenwiderstand bei Spurwechseln, Anti-Clipping).
4. Kann direkt als Modul oder standalone ueber die Befehlszeile ausgefuehrt werden.
"""

import argparse
import glob
import math
import os
import subprocess
import sys
import time

try:
    import evdev
    from evdev import ecodes
except ImportError:
    evdev = None
    ecodes = None


def find_wheel_event_device():
    """Findet den /dev/input/eventX Pfad des Logitech-Lenkrads."""
    # 1. Bevorzugt ueber /dev/input/by-id/
    candidates = (
        glob.glob('/dev/input/by-id/*046d_G27*event*')
        + glob.glob('/dev/input/by-id/*046d_G29*event*')
        + glob.glob('/dev/input/by-id/*Logitech*event-joystick')
        + glob.glob('/dev/input/by-id/*Racing_Wheel*event*')
    )
    for c in candidates:
        if os.path.exists(c):
            return os.path.realpath(c)

    # 2. Suche ueber evdev
    if evdev is not None:
        for p in evdev.list_devices():
            try:
                d = evdev.InputDevice(p)
                name = d.name.lower()
                if 'g27' in name or 'g29' in name or ('wheel' in name and 'logitech' in name):
                    return p
            except Exception:
                continue

    # 3. Fallback
    if os.path.exists('/dev/input/event5'):
        return '/dev/input/event5'
    return None


def apply_hardware_ffb(device_path=None, autocenter_pct=65, gain_pct=85, range_deg=900):
    """Setzt die Hardware-Zentrierfeder und Gesamtkraft direkt im Logitech-Treiber."""
    dev_path = device_path or find_wheel_event_device()
    if not dev_path or not os.path.exists(dev_path):
        print("[WHEEL FFB] Kein Logitech-Lenkrad unter /dev/input gefunden.")
        return False

    ac = float(autocenter_pct)
    ac_int = int(ac * 100) if ac <= 1.0 else int(ac)
    ac_int = max(0, min(100, ac_int))

    g = float(gain_pct)
    g_int = int(g * 100) if g <= 1.0 else int(g)
    g_int = max(0, min(100, g_int))

    r_int = int(range_deg)

    success = False

    # 1. ffset aufrufen (setzt den internen Motor-Controller direkt)
    try:
        cmd = ['ffset', dev_path, '-a', str(ac_int), '-g', str(g_int)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            success = True
            print(f"[WHEEL FFB] Hardware-Motor konfiguriert: Zentrierfeder={ac_int}%, Gain={g_int}% ({dev_path})")
        else:
            print(f"[WHEEL FFB] ffset Meldung: {res.stderr.strip() or res.stdout.strip()}")
    except FileNotFoundError:
        print("[WHEEL FFB] ffset-Befehl nicht gefunden.")

    # 2. Lenkwinkel (Range) in sysfs setzen
    try:
        ev_name = os.path.basename(dev_path)
        dev_link = f"/sys/class/input/{ev_name}/device"
        if os.path.exists(dev_link):
            real_dev = os.path.realpath(dev_link)
            search_dirs = [
                os.path.dirname(real_dev),
                os.path.dirname(os.path.dirname(real_dev)),
                real_dev
            ]
            for d in search_dirs:
                range_path = os.path.join(d, 'range')
                if os.path.exists(range_path):
                    with open(range_path, 'r') as f:
                        cur_range = f.read().strip()
                    if cur_range != str(r_int):
                        try:
                            with open(range_path, 'w') as f:
                                f.write(str(r_int))
                            print(f"[WHEEL FFB] Lenkwinkel angepasst: {cur_range}° -> {r_int}°")
                        except PermissionError:
                            print(f"[WHEEL FFB] Lenkwinkel im Kernel: {cur_range}° (sysfs schreibgeschützt)")
                    else:
                        print(f"[WHEEL FFB] Lenkwinkel: {r_int}°")
                    break
    except Exception:
        # sysfs kann read-only sein
        pass

    return success


class WheelForceFeedback:
    """Dynamischer Force-Feedback Controller fuer CARLA."""

    def __init__(
        self,
        steering_range=900,
        overall_gain=85,
        autocenter=65,
        damping_strength=35,
        friction_strength=15,
        anti_clipping=True,
        **kwargs
    ):
        # Flexible Parameternamen aus ini-Datei oder Aufruf abfangen
        if 'centering_strength' in kwargs:
            autocenter = kwargs['centering_strength']
        if 'gain' in kwargs:
            overall_gain = kwargs['gain']
        if 'range' in kwargs:
            steering_range = kwargs['range']

        self.steering_range = int(steering_range)

        g = float(overall_gain)
        self.overall_gain = g if g <= 1.0 else g / 100.0
        self.gain_pct = int(self.overall_gain * 100)

        ac = float(autocenter)
        self.autocenter_pct = int(ac * 100) if ac <= 1.0 else int(ac)

        ds = float(damping_strength)
        self.damping_strength = ds if ds <= 1.0 else ds / 100.0

        fs = float(friction_strength)
        self.friction_strength = fs if fs <= 1.0 else fs / 100.0

        self.anti_clipping = bool(anti_clipping)

        self.dev_path = find_wheel_event_device()
        self.dev = None
        self.effect_id = None
        self.effect = None
        self.last_steer = 0.0
        self.last_time = time.time()
        self.is_active = False

        if not self.dev_path:
            print("[WHEEL FFB] Kein Logitech-Lenkrad gefunden.")
            return

        # Hardware-Motor sofort ueber ffset konfigurieren (Zentrierfeder + Gain)
        apply_hardware_ffb(
            device_path=self.dev_path,
            autocenter_pct=self.autocenter_pct,
            gain_pct=self.gain_pct,
            range_deg=self.steering_range
        )

        if evdev is None:
            return

        try:
            self.dev = evdev.InputDevice(self.dev_path)
            caps = self.dev.capabilities().get(ecodes.EV_FF, [])
            if ecodes.FF_CONSTANT in caps:
                self.effect = evdev.ff.Effect(
                    ecodes.FF_CONSTANT, -1, 0,
                    evdev.ff.Trigger(0, 0),
                    evdev.ff.Replay(0, 0),
                    evdev.ff.EffectType(ff_constant_effect=evdev.ff.Constant(0, evdev.ff.Envelope(0, 0, 0, 0)))
                )
                self.effect_id = self.dev.upload_effect(self.effect)
                self.dev.write(ecodes.EV_FF, self.effect_id, 1)
                self.is_active = True
                print(f"[WHEEL FFB] Dynamisches Feedback aktiv (Daempfung={int(self.damping_strength*100)}%, Reibung={int(self.friction_strength*100)}%)")
        except Exception as e:
            print(f"[WHEEL FFB] evdev FF_CONSTANT Initialisierung: {e} (Hardware-Zentrierung bleibt aktiv)")

    def update(self, steer, speed_kmh, lateral_accel=0.0):
        """Berechnet dynamische Zusatzkraefte (Daempfung, Kurvenlast bei 60 km/h)."""
        if not self.is_active or self.dev is None or self.effect_id is None:
            return

        now = time.time()
        dt = max(0.001, now - self.last_time)
        steer_velocity = (steer - self.last_steer) / dt
        self.last_steer = steer
        self.last_time = now

        # Dynamische Daempfung gegen Lenkrad-Pendeln / Aufschwingen
        t_damping = -steer_velocity * self.damping_strength * 0.03

        # Kurvenlast bei Spurwechseln (nur bei Vorwaertsfahrt)
        speed_factor = min(1.0, max(0.0, speed_kmh / 50.0))
        t_lateral = -steer * 0.25 * speed_factor

        t_total = t_damping + t_lateral

        if self.anti_clipping:
            compressed = math.tanh(t_total)
        else:
            compressed = max(-1.0, min(1.0, t_total))

        level = int(max(-32767, min(32767, compressed * 32767)))

        try:
            self.effect.u.ff_constant_effect.level = level
            self.dev.upload_effect(self.effect)
        except Exception:
            pass

    def close(self):
        """Setzt das Lenkrad sauber zurueck."""
        if self.dev and self.effect_id is not None:
            try:
                self.dev.write(ecodes.EV_FF, self.effect_id, 0)
                self.dev.erase_effect(self.effect_id)
            except Exception:
                pass
            self.effect_id = None
            self.dev = None

        # Moderate Standard-Zentrierfeder (50%) fuer Linux-Desktop beibehalten
        if self.dev_path:
            apply_hardware_ffb(device_path=self.dev_path, autocenter_pct=50, gain_pct=80)
        self.is_active = False


def main():
    parser = argparse.ArgumentParser(description="Logitech Wheel Force-Feedback Tester")
    parser.add_argument('-a', '--autocenter', type=int, default=60, help='Zentrierfeder 0-100%% (Standard: 60)')
    parser.add_argument('-g', '--gain', type=int, default=85, help='Gesamtkraft 0-100%% (Standard: 85)')
    parser.add_argument('-r', '--range', type=int, default=900, help='Lenkwinkel in Grad (Standard: 900)')
    args = parser.parse_args()

    print(f"Konfiguriere Lenkrad: Zentrierfeder={args.autocenter}%, Gain={args.gain}%, Lenkwinkel={args.range}°...")
    ok = apply_hardware_ffb(autocenter_pct=args.autocenter, gain_pct=args.gain, range_deg=args.range)
    if ok:
        print("Erfolgreich gesetzt! Bitte am Lenkrad drehen, um die Zentrierfeder zu testen.")
    else:
        print("Konnte Lenkrad nicht konfigurieren.")


if __name__ == '__main__':
    main()
