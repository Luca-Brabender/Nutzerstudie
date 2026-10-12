#!/usr/bin/env python3
import os
import subprocess
import time


class AAOSRecorder:
    def __init__(self, output_dir, device_ip="192.168.1.100"):
        self.output_dir = output_dir
        self.device_ip = device_ip
        self.process = None

    def start_recording(self):
        """Startet das Screenrecording auf dem Raspberry Pi 4 via ADB im Hintergrund."""
        remote_file = "/sdcard/aaos_ui_record.mp4"
        print("\n[AAOS RECORDER] Starte UI-Aufzeichnung auf Raspberry Pi 4...")
        
        # Aufzeichnung auf max. 180 Sekunden (3 Min) begrenzen
        cmd = f"adb shell screenrecord --time-limit 180 {remote_file}"
        self.process = subprocess.Popen(cmd, shell=True)

    def stop_and_pull(self, run_idx):
        """Stoppt die Aufnahme und lädt das Video in deinen Ziel-Ordner herunter."""
        if self.process and self.process.poll() is None:
            print("[AAOS RECORDER] Stoppe Screenrecording...")
            # Sendet Interrupt-Signal (SIGINT/Ctrl+C), damit screenrecord die MP4 sauber abschließt
            subprocess.run(["adb", "shell", "pkill", "-INT", "screenrecord"], stdout=subprocess.DEVNULL)
            time.sleep(1)

        os.makedirs(self.output_dir, exist_ok=True)
        local_file = os.path.join(self.output_dir, f"aaos_ui_run_{run_idx + 1}.mp4")
        
        print(f"[AAOS RECORDER] Lade Video vom RPi 4 herunter nach: {local_file}")
        subprocess.run(["adb", "pull", "/sdcard/aaos_ui_record.mp4", local_file], stdout=subprocess.DEVNULL)


def setup_external_hardware():
    """Prüft die Verfügbarkeit der Pixel 6 V4L2-Kamera und startet den ADB-Server."""
    print("\n=== [HARDWARE CHECK] Prüfe Pixel 6 Kamera & RPi 4 ADB ===")
    
    # 1. V4L2 Kamera-Check (Pixel 6 / v4l2loopback)
    if not os.path.exists("/dev/video0"):
        print("[WARNHINWEIS] /dev/video0 nicht gefunden! Bitte Pixel 6 anschließen.")
    else:
        print("[OK] Pixel 6 Kamera /dev/video0 ist bereit.")

    # 2. ADB Server prüfen/starten
    try:
        subprocess.run(["adb", "start-server"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("[OK] ADB Daemon läuft.")
    except Exception as e:
        print(f"[WARNHINWEIS] ADB konnte nicht gestartet werden: {e}")
