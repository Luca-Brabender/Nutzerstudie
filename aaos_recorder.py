#!/usr/bin/env python3
import os
import signal
import subprocess
import time
import shutil


class AAOSRecorder:
    """Zeichnet den Bildschirm des Android Automotive OS (Raspberry Pi 4) via scrcpy auf.
    
    Vorteile von scrcpy:
    - --show-touches: Zeichnet die Touch-Eingaben des Probanden als sichtbare Punkte auf
    - Direktes Streaming in eine lokale MP4-Datei auf dem Host (kein Speicherverbrauch auf dem RPi)
    - 60 FPS für flüssige Bewegungen und minimale Latenz
    - --no-playback: Kein störendes Fenster auf dem Host-Bildschirm während der Fahrt
    """

    def __init__(
        self,
        output_dir,
        device_ip="192.168.0.103",
        port=5555,
        max_fps=60,
        bit_rate="2M",
        max_size=800,
        encoder="c2.android.avc.encoder",
    ):
        self.output_dir = output_dir
        self.device_ip = device_ip
        self.port = int(port)
        self.device_serial = f"{self.device_ip}:{self.port}"
        self.max_fps = int(max_fps)
        self.bit_rate = str(bit_rate)
        self.max_size = int(max_size)
        self.encoder = encoder

        self.process = None
        self.target_file = None
        self.start_time = None
        self.start_line_time = None
        self.start_line_offset = 0.0

    def mark_start_line(self):
        """Wird exakt beim Überqueren der Startlinie (x >= 0.0m) aufgerufen.
        
        Nimmt den Zeitstempel und den Offset zur Video-Aufnahme auf (benötigt 0 ms, kein Ruckeln).
        """
        now = time.time()
        self.start_line_time = now
        if self.start_time is not None:
            self.start_line_offset = max(0.0, now - self.start_time)
        else:
            self.start_line_offset = 0.0
        return self.start_line_offset

    def connect(self):
        """Verbindet ADB mit dem Raspberry Pi über TCP/IP."""
        try:
            subprocess.run(["adb", "start-server"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
            res = subprocess.run(
                ["adb", "connect", self.device_serial],
                capture_output=True,
                text=True,
                timeout=8
            )
            out = res.stdout.strip()
            state = subprocess.run(
                ["adb", "-s", self.device_serial, "get-state"],
                capture_output=True,
                text=True,
                timeout=4
            ).stdout.strip()

            if state == "device":
                print(f"[AAOS RECORDER] ADB erfolgreich verbunden mit RPi 4 ({self.device_serial}).")
                self.enable_show_touches(True)
                self.configure_device_display()
                return True
            else:
                print(f"[AAOS RECORDER WARNUNG] ADB Status '{state}' für {self.device_serial}: {out}")
                return False
        except Exception as e:
            print(f"[AAOS RECORDER WARNUNG] Fehler beim Verbindungsaufbau zu {self.device_serial}: {e}")
            return False

    def enable_show_touches(self, enabled=True):
        """Aktiviert oder deaktiviert die Touch-Indikatoren direkt in Android (ohne Fenster/Control in scrcpy)."""
        try:
            val = "1" if enabled else "0"
            subprocess.run(
                ["adb", "-s", self.device_serial, "shell", "settings", "put", "system", "show_touches", val],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3
            )
        except Exception:
            pass

    def start_recording(self, run_prefix=None, run_idx=0):
        """Startet das Screenrecording auf dem Host via scrcpy."""
        if self.process and self.process.poll() is None:
            print("[AAOS RECORDER] Aufzeichnung läuft bereits.")
            return True

        if not shutil.which("scrcpy"):
            print("[AAOS RECORDER FEHLER] 'scrcpy' ist nicht installiert oder nicht im PATH gefunden.")
            return False

        # ADB Verbindung sicherstellen
        self.connect()

        os.makedirs(self.output_dir, exist_ok=True)
        if run_prefix:
            self.target_file = f"{run_prefix}_aaos.mp4"
        else:
            self.target_file = os.path.join(self.output_dir, f"aaos_ui_run_{run_idx + 1}.mp4")

        # Touch-Punkte in Android aktivieren
        self.enable_show_touches(True)

        # --no-window verhindert die Erzeugung eines Desktop-/Dummy-Fensters durch scrcpy/SDL vollständig.
        # Damit wird der Fokus von CARLA/Pygame nicht gestohlen und kein Android-Symbol angezeigt.
        cmd = [
            "scrcpy",
            "-s", self.device_serial,
            "--no-window",
            "--no-audio",
            "--video-codec=h264",
            f"--video-encoder={self.encoder}",
            f"-m{self.max_size}",
            f"-b{self.bit_rate}",
            f"--max-fps={self.max_fps}",
            f"--record={self.target_file}",
        ]

        print(f"\n[AAOS RECORDER] Starte UI-Aufzeichnung auf RPi 4 ({self.device_serial})...")
        print(f"                Datei:   {self.target_file}")
        print(f"                Format:  {self.max_size}p @ {self.max_fps} FPS (Bitrate: {self.bit_rate}, Encoder: {self.encoder})")

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True
            )
            self.start_time = time.time()
            return True
        except Exception as e:
            print(f"[AAOS RECORDER FEHLER] scrcpy konnte nicht gestartet werden: {e}")
            self.process = None
            return False

    def configure_device_display(self):
        """Verhindert Standby/Display-Timeout auf dem Raspberry Pi 4 während der Studie."""
        try:
            subprocess.run(
                ["adb", "-s", self.device_serial, "shell", "settings", "put", "system", "screen_off_timeout", "2147483647"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3
            )
            subprocess.run(
                ["adb", "-s", self.device_serial, "shell", "svc", "power", "stayon", "true"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3
            )
        except Exception:
            pass

    def stop(self):
        """Stoppt die scrcpy-Aufzeichnung sauber via SIGINT und synchronisiert das Video mit der Startlinie."""
        if not self.process:
            return None

        now = time.time()
        expected_total_duration = now - (self.start_time or now)
        start_offset = self.start_line_offset or 0.0
        if self.start_line_time is not None:
            drive_duration = max(1.0, now - self.start_line_time)
        else:
            drive_duration = expected_total_duration

        if self.process.poll() is None:
            print("[AAOS RECORDER] Stoppe UI-Aufzeichnung (SIGINT an scrcpy)...")
            try:
                self.process.send_signal(signal.SIGINT)
                try:
                    _, errs = self.process.communicate(timeout=6)
                except subprocess.TimeoutExpired:
                    print("[AAOS RECORDER] scrcpy reagiert nicht, beende Prozess erzwungen...")
                    self.process.kill()
                    _, errs = self.process.communicate()
            except Exception as e:
                print(f"[AAOS RECORDER] Fehler beim Beenden: {e}")

        self.process = None

        if self.target_file and os.path.exists(self.target_file):
            raw_file = self.target_file
            # Prüfen, ob das Rohvideo durch statischen Bildschirm am Ende kürzer als die Gesamtlaufzeit ist
            actual_duration = None
            try:
                probe_res = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", raw_file],
                    capture_output=True, text=True, timeout=5
                )
                if probe_res.returncode == 0 and probe_res.stdout.strip():
                    actual_duration = float(probe_res.stdout.strip())
            except Exception:
                pass

            # Standbild am Ende verlängern, falls die scrcpy-Aufnahme durch Inaktivität vorzeitig endete
            if actual_duration is not None and expected_total_duration > (actual_duration + 0.8):
                pad_sec = expected_total_duration - actual_duration
                print(f"[AAOS RECORDER] Statischer Bildschirm erkannt ({actual_duration:.1f}s Aufzeichnung vs. {expected_total_duration:.1f}s Gesamtlaufzeit).")
                print(f"[AAOS RECORDER] Verlängere letzten Frame um {pad_sec:.1f}s auf exakte Ziellinienzeit ({expected_total_duration:.1f}s, 60 FPS CFR)...")

                temp_pad = raw_file + ".pad.mp4"
                cmd_pad = [
                    "ffmpeg", "-y", "-i", raw_file,
                    "-vf", f"tpad=stop_mode=clone:stop_duration={pad_sec:.3f},fps={self.max_fps}",
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
                    temp_pad
                ]
                try:
                    res_pad = subprocess.run(cmd_pad, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=25)
                    if res_pad.returncode == 0 and os.path.exists(temp_pad) and os.path.getsize(temp_pad) > 1000:
                        os.replace(temp_pad, raw_file)
                except Exception as e:
                    print(f"[AAOS RECORDER] Hinweis beim Nachberechnen des Videos: {e}")

            # SYNCHRONISATION MIT CSV-DATEN:
            # Falls vor der Startlinie gestartet wurde (start_offset > 0.05s):
            # 1. Behalte die ungeschnittene Rohaufnahme als <name>_full.mp4
            # 2. Erstelle <target_file> so, dass Sekunde 00:00.00 EXAKT die Startlinie ist!
            #    Zusätzlich blenden wir in den ersten 3.5 Sekunden ein optisches Badge 'LCT STARTLINIE' ein!
            if start_offset > 0.05:
                full_backup_file = raw_file.replace("_aaos.mp4", "_aaos_full.mp4")
                if full_backup_file == raw_file:
                    full_backup_file = raw_file + ".full.mp4"

                # Backup des Rohvideos anlegen
                try:
                    shutil.copy2(raw_file, full_backup_file)
                except Exception:
                    pass

                sync_trimmed = raw_file + ".trimmed.mp4"
                filter_chain = (
                    f"trim=start={start_offset:.3f}:duration={drive_duration:.3f},"
                    f"setpts=PTS-STARTPTS,"
                    f"drawtext=text='LCT STARTLINIE':fontcolor=white:fontsize=24:box=1:boxcolor=0x228B22@0.85:boxborderw=6:x=(w-text_w)/2:y=16:enable='between(t,0,3.5)',"
                    f"fps={self.max_fps}"
                )
                cmd_trim = [
                    "ffmpeg", "-y", "-i", raw_file,
                    "-vf", filter_chain,
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
                    sync_trimmed
                ]
                print(f"[AAOS RECORDER] Erstelle synchronisiertes Video (Startlinie = 00:00.00, Offset: {start_offset:.2f}s, Fahrzeit: {drive_duration:.1f}s)...")
                try:
                    res_trim = subprocess.run(cmd_trim, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=25)
                    if res_trim.returncode == 0 and os.path.exists(sync_trimmed) and os.path.getsize(sync_trimmed) > 1000:
                        os.replace(sync_trimmed, raw_file)
                        print(f"[AAOS RECORDER OK] Synchronisiertes Video: {raw_file} (Sekunde 00:00 = Startlinie)")
                        print(f"[AAOS RECORDER OK] Vollständige Rohaufnahme: {full_backup_file}")
                except Exception as e:
                    print(f"[AAOS RECORDER] Fehler beim Schneiden: {e}")
            else:
                # Video begann direkt an der Startlinie -> nur Start-Badge einblenden
                badge_file = raw_file + ".badge.mp4"
                filter_chain = (
                    f"drawtext=text='LCT STARTLINIE':fontcolor=white:fontsize=24:box=1:boxcolor=0x228B22@0.85:boxborderw=6:x=(w-text_w)/2:y=16:enable='between(t,0,3.5)',"
                    f"fps={self.max_fps}"
                )
                cmd_badge = [
                    "ffmpeg", "-y", "-i", raw_file,
                    "-vf", filter_chain,
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
                    badge_file
                ]
                try:
                    res_badge = subprocess.run(cmd_badge, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
                    if res_badge.returncode == 0 and os.path.exists(badge_file) and os.path.getsize(badge_file) > 1000:
                        os.replace(badge_file, raw_file)
                except Exception:
                    pass

            size_mb = os.path.getsize(self.target_file) / (1024 * 1024)
            print(f"[AAOS RECORDER OK] Video gespeichert: {self.target_file} ({size_mb:.2f} MB, Fahrzeit: {drive_duration:.1f}s)")
            return self.target_file
        else:
            print(f"[AAOS RECORDER WARNUNG] Zieldatei wurde nicht erstellt: {self.target_file}")
            return None

    def stop_and_pull(self, run_idx=0):
        """Rückwärtskompatible Methode zu früheren Versionen."""
        return self.stop()


def setup_external_hardware(device_ip="192.168.0.103", port=5555):
    """Prüft die Verfügbarkeit der Pixel 6 V4L2-Kamera und verbindet ADB zum RPi 4."""
    print("\n=== [HARDWARE CHECK] Prüfe Pixel 6 Kamera & RPi 4 AAOS (scrcpy) ===")
    
    # 1. V4L2 Kamera-Check (Pixel 6 / v4l2loopback)
    if not os.path.exists("/dev/video0"):
        print("[WARNHINWEIS] /dev/video0 nicht gefunden! Bitte Pixel 6 anschließen.")
    else:
        print("[OK] Pixel 6 Kamera /dev/video0 ist bereit.")

    # 2. scrcpy prüfen
    if shutil.which("scrcpy"):
        print("[OK] scrcpy ist auf dem System installiert.")
    else:
        print("[WARNHINWEIS] scrcpy wurde nicht gefunden! Bitte installieren via apt oder snap.")

    # 3. ADB Server & RPi 4 Verbindung
    device_serial = f"{device_ip}:{port}"
    try:
        subprocess.run(["adb", "start-server"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        connect_res = subprocess.run(
            ["adb", "connect", device_serial],
            capture_output=True,
            text=True,
            timeout=8
        )
        print(f"[ADB] {connect_res.stdout.strip()}")

        devices_out = subprocess.run(["adb", "devices"], capture_output=True, text=True, timeout=4).stdout
        print(f"[OK] ADB Geräte-Liste:\n{devices_out.strip()}")

        # Touch-Punkte in Android aktivieren
        subprocess.run(
            ["adb", "-s", device_serial, "shell", "settings", "put", "system", "show_touches", "1"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3
        )
        # Standby / Display-Timeout deaktivieren
        subprocess.run(
            ["adb", "-s", device_serial, "shell", "settings", "put", "system", "screen_off_timeout", "2147483647"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3
        )
        subprocess.run(
            ["adb", "-s", device_serial, "shell", "svc", "power", "stayon", "true"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3
        )
        print("[OK] Android Touch-Indikatoren & Daueranzeige (kein Standby) aktiviert.")
    except Exception as e:
        print(f"[WARNHINWEIS] ADB konnte nicht gestartet oder verbunden werden: {e}")


def set_knob_service_mode(device_ip="192.168.0.103", port=5555, standard_rotary=False):
    """Konfiguriert den Betriebsmodus des MultiKnob in KnobService.kt auf Android Automotive via ADB Broadcast.
    
    standard_rotary = True  -> Standard-Drehknopf / Drehregler (Fingererkennung AUS)
    standard_rotary = False -> MultiKnob (Fingererkennung AN)
    """
    device_serial = f"{device_ip}:{port}"
    val_str = "true" if standard_rotary else "false"
    mode_name = "Standard-Drehregler (Fingererkennung AUS)" if standard_rotary else "MultiKnob (Fingererkennung AN)"
    cmd = [
        "adb", "-s", device_serial, "shell",
        "am", "broadcast", "--user", "0",
        "-a", "com.example.knobservice.SET_ROTARY_MODE",
        "--ez", "standard_rotary", val_str
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        print(f"[KNOB SERVICE] Betriebsmodus gesetzt: {mode_name} (standard_rotary={val_str})")
        return True
    except Exception as e:
        print(f"[KNOB SERVICE WARNHINWEIS] Konnte Modus nicht an KnobService senden: {e}")
        return False
