#!/usr/bin/env python3
import os
import subprocess
import time


class AAOSRecorder:
    def __init__(self, output_dir, device_ip="192.168.0.193"):
        self.output_dir = output_dir
        self.device_ip = device_ip
        self.process = None
        self.remote_file = "/data/local/tmp/aaos_ui_record.mp4" # Sicherer Systempfad

    def start_recording(self):
        """Startet das Screenrecording auf dem Raspberry Pi 4 via ADB im Hintergrund."""
        print("\n[AAOS RECORDER] Starte UI-Aufzeichnung auf Raspberry Pi 4...")
        
        # Alte Testdatei auf RPi löschen
        subprocess.run(f"adb shell rm -f {self.remote_file}", shell=True, stderr=subprocess.DEVNULL)
        
        # Screenrecord starten (--bugreport sorgt für breitere Kompatibilität)
        cmd = f"adb shell screenrecord --bugreport --time-limit 180 {self.remote_file}"
        self.process = subprocess.Popen(cmd, shell=True)

    def stop_and_pull(self, run_idx):
        """Stoppt die Aufnahme und lädt das Video in deinen Ziel-Ordner herunter."""
        if self.process and self.process.poll() is None:
            print("[AAOS RECORDER] Stoppe Screenrecording...")
            # Sendet Interrupt-Signal (SIGINT/Ctrl+C), damit screenrecord die MP4 sauber abschließt
            subprocess.run("adb shell pkill -INT screenrecord", shell=True)
            time.sleep(2) # Kurze Pause, damit RPi die MP4-Header schreiben kann

        os.makedirs(self.output_dir, exist_ok=True)
        local_file = os.path.join(self.output_dir, f"aaos_ui_run_{run_idx + 1}.mp4")
        
        print(f"[AAOS RECORDER] Lade Video vom RPi 4 herunter nach: {local_file}")
        
        # ADB Pull ausführen
        result = subprocess.run(
            ["adb", "pull", self.remote_file, local_file],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"[AAOS RECORDER FEHLER] Download fehlgeschlagen: {result.stderr.strip()}")
            if os.path.exists(local_file):
                os.remove(local_file) # Fehlerhafte Textdatei löschen
        else:
            print(f"[AAOS RECORDER OK] Video erfolgreich gespeichert ({os.path.getsize(local_file) / 1024 / 1024:.2f} MB)")


def setup_external_hardware():
    """Prüft die Verfügbarkeit der Pixel 6 V4L2-Kamera und startet den ADB-Server."""
    print("\n=== [HARDWARE CHECK] Prüfe Pixel 6 Kamera & RPi 4 ADB ===")
    
    # 1. V4L2 Kamera-Check (Pixel 6 / v4l2loopback)
    if not os.path.exists("/dev/video0"):
        print("[WARNHINWEIS] /dev/video0 nicht gefunden! Bitte Pixel 6 anschließen.")
    else:
        print("[OK] Pixel 6 Kamera /dev/video0 ist bereit.")

    # 2. ADB Server prüfen & RPi4 verbinden
    try:
        subprocess.run(["adb", "start-server"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        adb_devices = subprocess.run(["adb", "devices"], capture_output=True, text=True).stdout
        print(f"[OK] ADB Geräte-Liste:\n{adb_devices.strip()}")
    except Exception as e:
        print(f"[WARNHINWEIS] ADB konnte nicht gestartet werden: {e}")
