# lct_logger.py
import csv
import math
import time


def classify_aoi(gaze_x, gaze_y, y_threshold=600):
    """Klassifiziert den Fokus anhand der Y-Bildschirmkoordinaten.

    - Y > y_threshold: INFOTAINMENT (unterer Bildschirmbereich)
    - sonst: ROAD
    """
    if gaze_x is None or gaze_y is None or gaze_x < 0 or gaze_y < 0:
        return 'UNKNOWN'

    if gaze_y > y_threshold:
        return 'INFOTAINMENT'
    else:
        return 'ROAD'


class LCTLogger:

    def __init__(
        self,
        start_x=0.0,
        finish_x=3000.0,
        gaze_estimator=None,
        aoi_threshold_y=600,
    ):
        self.start_x = start_x
        self.finish_x = finish_x
        self.gaze_estimator = gaze_estimator
        self.aoi_threshold_y = aoi_threshold_y

        self.is_recording = False
        self.completed = False

        self.start_time = None
        self.total_time = 0.0

        self.csv_file = None
        self.csv_writer = None

    def update(self, world, signs_data, webcam_frame=None, lane_event='None', gaze_x=-1.0, gaze_y=-1.0):
        if world.player is None or self.completed:
            return

        p_trans = world.player.get_transform()
        p_vel = world.player.get_velocity()
        p_ctrl = world.player.get_control()

        cur_x = p_trans.location.x
        raw_y = p_trans.location.y
        y_norm = raw_y + 7.55  # Normalisiertes Y (Unterkante = 0.0m)
        speed_kmh = 3.6 * math.sqrt(p_vel.x**2 + p_vel.y**2 + p_vel.z**2)
        current_time = time.time()

        # --- 1. STARTLINIE ÜBERFAHREN (X >= 0.0m) ---
        if not self.is_recording and cur_x >= self.start_x:
            self.is_recording = True
            self.start_time = current_time

            filename = f'LCT_Log_{int(self.start_time)}.csv'
            self.csv_file = open(filename, mode='w', newline='')
            self.csv_writer = csv.writer(self.csv_file)

            # CSV Header schreiben
            self.csv_writer.writerow([
                'timestamp_sim',
                'elapsed_time_s',
                'pos_x',
                'pos_y_raw',
                'pos_y_norm',
                'speed_kmh',
                'steer',
                'active_sign_idx',
                'active_maneuver',
                'dist_to_next_sign',
                'gaze_x',
                'gaze_y',
                'aoi_focus',
                'line_type',
            ])

            world.hud.notification('=== STARTLINIE ÜBERFAHREN: LOGGING GESTARTET ===')
            print(f'[LCT LOG] Startlinie bei X={cur_x:.2f}m überfahren. Messung gestartet!')

        # --- 2. ZIELLINIE ÜBERFAHREN (X >= 3000.0m) ---
        elif self.is_recording and cur_x >= self.finish_x:
            self.is_recording = False
            self.completed = True
            self.total_time = current_time - self.start_time

            self.close()

            world.hud.notification(f'=== ZIEL ERREICHT! Zeit: {self.total_time:.2f}s ===')
            print(f'[LCT LOG] Ziellinie bei X={cur_x:.2f}m erreicht! Gesamtzeit: {self.total_time:.2f}s')
            return

        # --- 3. DATEN SCHREIBEN (Während Messung aktiv ist) ---
        if self.is_recording and self.csv_writer:
            active_idx = -1
            active_maneuver = 'None'
            dist_to_next = 999.0

            # Nächstes/Aktives Schild in Fahrtrichtung ermitteln
            for idx, item in enumerate(signs_data):
                if isinstance(item, str):
                    sign_x = (idx + 1) * 150.0
                    maneuver = item
                elif isinstance(item, dict):
                    sign_x = item.get('x_pos', item.get('x', (idx + 1) * 150.0))
                    maneuver = item.get('maneuver', 'Unknown')
                elif isinstance(item, (tuple, list)):
                    sign_x = item[1]
                    maneuver = item[2]
                else:
                    continue

                if sign_x > cur_x:
                    active_idx = idx
                    active_maneuver = maneuver
                    dist_to_next = sign_x - cur_x
                    break

            # AOI-Klassifizierung ausführen
            aoi_focus = classify_aoi(gaze_x, gaze_y, y_threshold=self.aoi_threshold_y)
            
            line_type = lane_event
            elapsed_s = current_time - self.start_time

            self.csv_writer.writerow([
                round(world.hud.simulation_time, 3),
                round(elapsed_s, 3),
                round(cur_x, 3),
                round(raw_y, 3),
                round(y_norm, 3),
                round(speed_kmh, 2),
                round(p_ctrl.steer, 4),
                active_idx,
                active_maneuver,
                round(dist_to_next, 2),
                round(gaze_x, 1) if gaze_x != -1.0 else -1,
                round(gaze_y, 1) if gaze_y != -1.0 else -1,
                aoi_focus,
                line_type,
            ])

    def close(self):
        """Schließt die CSV-Datei ordnungsgemäß."""
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None
            print('[LCT LOG] CSV-Datei erfolgreich gespeichert und geschlossen.')
