#!/usr/bin/env python3
# lct_analysis.py
"""Auswertung der LCT-Nutzerstudie (ISO 26022) inkl. Blickzuwendungen zur Headunit.

Kennwerte pro Fahrt
-------------------
* **MDEV** (mean deviation, ISO 26022): mittlere absolute Querabweichung [m]
  zwischen gefahrener Spur und normativem Referenzpfad, über die Strecke
  (auf ein 1-m-Raster interpoliert, damit Geschwindigkeitsschwankungen /
  Framerate keine Gewichtung erzeugen).
    - ``mdev_basic``    : Basis-Normmodell (feste Parameter, s.u.)
    - ``mdev_adapted``  : adaptiertes Modell – Spurwechselbeginn und -länge
                          werden je Proband aus seinen Baseline-Fahrten
                          geschätzt (kleinster MDEV) und auf alle Fahrten
                          dieses Probanden angewandt.
* **Spurwechselfehler**: falsche Zielspur kurz nach dem Schild.
* **Blickzuwendungen zur Headunit** (DIN EN ISO 15007): Anzahl, mittlere /
  maximale Dauer, Gesamtblickdauer (TGT), Anzahl > 2 s, Anteil
  Eyes-off-Road, Datenqualität (Samplerate, Gesichtserkennung).

!! WICHTIG: Die Parameter des Basis-Normmodells (``--lc-start`` = Abstand des
   Spurwechselbeginns VOR dem Schild, ``--lc-length`` = Länge des Spur-
   wechsels) bitte mit dem Normtext ISO 26022:2010 abgleichen und ggf.
   anpassen. Das adaptierte Modell ist davon unabhängig.

Aufruf
------
    python3 lct_analysis.py logs/                # alle Probanden
    python3 lct_analysis.py logs/P01 --plot      # mit Trajektorien-Plots
    python3 lct_analysis.py LCT_Log_1791552587.csv
"""

import argparse
import csv
import glob
import math
import os
import re
import statistics

# Querposition (pos_y_norm) der Spurmitten. pos_y_norm = carla_y + 7.55,
# Spurbreite 3.775 m. In CARLA zeigt +y nach RECHTS -> 'Left' = kleines y.
LANE_WIDTH = 3.775
LANE_CENTER = {'Left': 0.5 * LANE_WIDTH,
               'Middle': 1.5 * LANE_WIDTH,
               'Right': 2.5 * LANE_WIDTH}
START_LANE = 'Middle'
SIGN_SPACING = 150.0
N_SIGNS = 18

FNAME_RE = re.compile(
    r'^(?P<participant>.+?)_step(?P<step>\d+)_(?P<map>.+)_(?P<ts>\d{8}-\d{6})_lct\.csv$')


# ==============================================================================
# -- Einlesen ------------------------------------------------------------------
# ==============================================================================


def read_lct(path):
    xs, ys = [], []
    seq = {}
    with open(path, newline='') as fh:
        for row in csv.DictReader(fh):
            try:
                x = float(row['pos_x'])
                y = float(row['pos_y_norm'])
            except (KeyError, ValueError):
                continue
            xs.append(x)
            ys.append(y)
            idx = int(row.get('active_sign_idx', -1))
            man = row.get('active_maneuver', 'None')
            if idx >= 0 and man in LANE_CENTER and idx not in seq:
                seq[idx] = man
    # nach x sortieren, Duplikate entfernen (Stillstand)
    pts = sorted(zip(xs, ys))
    xs2, ys2 = [], []
    for x, y in pts:
        if xs2 and x <= xs2[-1]:
            continue
        xs2.append(x)
        ys2.append(y)
    sequence = [seq.get(i) for i in range(N_SIGNS)]
    return xs2, ys2, sequence


def resample(xs, ys, x_from, x_to, step=1.0):
    gx, gy = [], []
    j = 0
    x = x_from
    while x <= x_to + 1e-9:
        while j + 1 < len(xs) and xs[j + 1] < x:
            j += 1
        if xs and xs[0] <= x <= xs[-1] and j + 1 < len(xs):
            x0, x1 = xs[j], xs[j + 1]
            f = (x - x0) / (x1 - x0) if x1 > x0 else 0.0
            gx.append(x)
            gy.append(ys[j] + f * (ys[j + 1] - ys[j]))
        x += step
    return gx, gy


def read_glances(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh))


def read_gaze_quality(path):
    if not path or not os.path.exists(path):
        return {}
    n = face = 0
    t0 = t1 = None
    lat = []
    with open(path, newline='') as fh:
        for row in csv.DictReader(fh):
            t = float(row['unix_time'])
            t0 = t if t0 is None else t0
            t1 = t
            n += 1
            face += int(row['face'])
            try:
                lat.append(float(row['latency_ms']))
            except ValueError:
                pass
    dur = (t1 - t0) if (t0 is not None and t1 > t0) else float('nan')
    return {'gaze_samples': n, 'gaze_rate_hz': n / dur if dur == dur and dur > 0 else '',
            'face_detect_pct': 100.0 * face / n if n else '',
            'ws_latency_ms_median': statistics.median(lat) if lat else ''}


# ==============================================================================
# -- Normmodell & MDEV ---------------------------------------------------------
# ==============================================================================


def normative_path(gx, sequence, lc_start, lc_length):
    """Stückweise linearer Referenzpfad.

    Für Schild i (bei x_i) beginnt der Spurwechsel bei x_i - lc_start und endet
    lc_length Meter später in der Mitte der Zielspur.
    """
    events = []
    lane = START_LANE
    for i, target in enumerate(sequence):
        if target is None:
            continue
        sx = (i + 1) * SIGN_SPACING
        events.append((sx - lc_start, sx - lc_start + lc_length,
                       LANE_CENTER[lane], LANE_CENTER[target]))
        lane = target
    out = []
    k = 0
    cur = LANE_CENTER[START_LANE]
    for x in gx:
        while k < len(events) and x >= events[k][1]:
            cur = events[k][3]
            k += 1
        if k < len(events) and x >= events[k][0]:
            s0, s1, y0, y1 = events[k]
            f = (x - s0) / (s1 - s0) if s1 > s0 else 1.0
            out.append(y0 + f * (y1 - y0))
        else:
            out.append(cur)
    return out


def mdev(gy, ny):
    return sum(abs(a - b) for a, b in zip(gy, ny)) / len(gy) if gy else float('nan')


def lane_change_errors(gx, gy, sequence, check_after=30.0):
    """Zählt Schilder, nach denen (check_after m hinter dem Schild) nicht die
    Zielspur befahren wird."""
    errors = 0
    lookup = dict(zip([round(x) for x in gx], gy))
    for i, target in enumerate(sequence):
        if target is None:
            continue
        xc = round((i + 1) * SIGN_SPACING + check_after)
        if xc not in lookup:
            continue
        y = lookup[xc]
        lane = min(LANE_CENTER, key=lambda k: abs(LANE_CENTER[k] - y))
        if lane != target:
            errors += 1
    return errors


def fit_adapted(runs, starts=range(0, 101, 2), lengths=range(6, 101, 2)):
    best = None
    for s in starts:
        for l in lengths:
            vals = [mdev(r['gy'], normative_path(r['gx'], r['seq'], s, l)) for r in runs]
            m = sum(vals) / len(vals)
            if best is None or m < best[0]:
                best = (m, s, l)
    return best  # (mdev, start, length)


# ==============================================================================
# -- Blickkennwerte -----------------------------------------------------------
# ==============================================================================


def glance_metrics(glances, min_glance_s=0.12):
    if glances is None:
        return {}
    segs = [g for g in glances if g['aoi'] in ('SIMULATOR', 'HEADUNIT', 'OFFSCREEN', 'NO_FACE')]
    total = sum(float(g['duration_s']) for g in segs)
    hu = [float(g['duration_s']) for g in segs
          if g['aoi'] == 'HEADUNIT' and float(g['duration_s']) >= min_glance_s]
    off = sum(float(g['duration_s']) for g in segs if g['aoi'] != 'SIMULATOR')
    noface = sum(float(g['duration_s']) for g in segs if g['aoi'] == 'NO_FACE')
    return {
        'hu_glances_n': len(hu),
        'hu_glance_mean_s': statistics.mean(hu) if hu else 0.0,
        'hu_glance_sd_s': statistics.stdev(hu) if len(hu) > 1 else 0.0,
        'hu_glance_max_s': max(hu) if hu else 0.0,
        'hu_tgt_s': sum(hu),
        'hu_glances_over_2s': sum(1 for d in hu if d > 2.0),
        'hu_time_pct': 100.0 * sum(hu) / total if total else '',
        'eyes_off_sim_pct': 100.0 * off / total if total else '',
        'no_face_pct': 100.0 * noface / total if total else '',
    }


# ==============================================================================
# -- Plot ---------------------------------------------------------------------
# ==============================================================================


def plot_run(run, out_png, s_basic, l_basic, adapted):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  (matplotlib nicht installiert -> kein Plot)')
        return
    fig, ax = plt.subplots(figsize=(18, 4))
    for name, y in LANE_CENTER.items():
        ax.axhline(y, color='#ccc', lw=0.8, ls='--')
    ax.plot(run['gx'], normative_path(run['gx'], run['seq'], s_basic, l_basic),
            color='#888', lw=1, label='Normmodell (basic)')
    if adapted:
        ax.plot(run['gx'], normative_path(run['gx'], run['seq'], adapted[1], adapted[2]),
                color='#3a7', lw=1, label='Normmodell (adaptiert)')
    ax.plot(run['gx'], run['gy'], color='#1f4e9c', lw=1.4, label='Fahrspur')
    for g in run.get('glances') or []:
        if g['aoi'] == 'HEADUNIT' and g['start_pos_x'] not in ('', None):
            ax.axvspan(float(g['start_pos_x']), float(g['end_pos_x']), color='#e44', alpha=0.25)
    for i in range(N_SIGNS):
        ax.axvline((i + 1) * SIGN_SPACING, color='#f90', lw=0.6)
    ax.set_xlabel('Strecke x [m]  (rot = Blick auf Headunit, orange = Schild)')
    ax.set_ylabel('Querposition [m]')
    ax.set_title('%s  |  MDEV basic = %.3f m' % (run['label'], run['mdev_basic']))
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print('  Plot: %s' % out_png)


# ==============================================================================
# -- Main ---------------------------------------------------------------------
# ==============================================================================


def collect_files(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += glob.glob(os.path.join(p, '**', '*_lct.csv'), recursive=True)
        else:
            files.append(p)
    return sorted(set(files))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('paths', nargs='+', help='Log-Ordner oder *_lct.csv Dateien')
    ap.add_argument('--x-from', type=float, default=0.0)
    ap.add_argument('--x-to', type=float, default=3000.0)
    ap.add_argument('--lc-start', type=float, default=30.0,
                    help='Basis-Normmodell: Spurwechselbeginn [m] vor dem Schild')
    ap.add_argument('--lc-length', type=float, default=30.0,
                    help='Basis-Normmodell: Spurwechsellänge [m]')
    ap.add_argument('--min-glance', type=float, default=0.12,
                    help='Glances kürzer als dies [s] werden ignoriert')
    ap.add_argument('--out', default='lct_summary.csv')
    ap.add_argument('--plot', action='store_true')
    args = ap.parse_args()

    runs = []
    for f in collect_files(args.paths):
        base = os.path.basename(f)
        m = FNAME_RE.match(base)
        meta = m.groupdict() if m else {'participant': 'unbekannt', 'step': '',
                                        'map': '', 'ts': ''}
        xs, ys, seq = read_lct(f)
        if len(xs) < 10:
            print('Übersprungen (zu wenige Daten): %s' % f)
            continue
        gx, gy = resample(xs, ys, args.x_from, args.x_to)
        prefix = f[:-len('_lct.csv')] if f.endswith('_lct.csv') else None
        run = dict(meta, file=f, gx=gx, gy=gy, seq=seq,
                   label='%s step%s %s' % (meta['participant'], meta['step'], meta['map']),
                   glances=read_glances(prefix + '_glances.csv' if prefix else None),
                   quality=read_gaze_quality(prefix + '_gaze.csv' if prefix else None),
                   coverage_m=(xs[-1] - xs[0]))
        run['mdev_basic'] = mdev(gy, normative_path(gx, seq, args.lc_start, args.lc_length))
        run['lc_errors'] = lane_change_errors(gx, gy, seq)
        runs.append(run)

    if not runs:
        print('Keine LCT-Logs gefunden.')
        return

    # Adaptiertes Modell je Proband aus Baseline-Fahrten
    adapted = {}
    for p in sorted(set(r['participant'] for r in runs)):
        base_runs = [r for r in runs if r['participant'] == p and 'baseline' in r['map'].lower()]
        if not base_runs:
            base_runs = [r for r in runs if r['participant'] == p]
            print('Hinweis: %s ohne Baseline-Fahrt -> adaptiertes Modell aus allen Fahrten.' % p)
        adapted[p] = fit_adapted(base_runs)
        print('Adaptiertes Modell %s: Beginn %d m vor Schild, Länge %d m (Baseline-MDEV %.3f m)'
              % (p, adapted[p][1], adapted[p][2], adapted[p][0]))

    fields = ['participant', 'step', 'map', 'file', 'coverage_m', 'mdev_basic', 'mdev_adapted',
              'adapted_lc_start_m', 'adapted_lc_length_m', 'lc_errors',
              'hu_glances_n', 'hu_glance_mean_s', 'hu_glance_sd_s', 'hu_glance_max_s',
              'hu_tgt_s', 'hu_glances_over_2s', 'hu_time_pct', 'eyes_off_sim_pct',
              'no_face_pct', 'gaze_samples', 'gaze_rate_hz', 'face_detect_pct',
              'ws_latency_ms_median']
    with open(args.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        print('\n%-10s %-5s %-24s %8s %8s %4s %5s %7s %7s' % (
            'Proband', 'Step', 'Map', 'MDEVb', 'MDEVa', 'Err', 'nHU', 'TGT[s]', 'max[s]'))
        for r in runs:
            a = adapted[r['participant']]
            r['mdev_adapted'] = mdev(r['gy'], normative_path(r['gx'], r['seq'], a[1], a[2]))
            row = dict(r)
            row.update(adapted_lc_start_m=a[1], adapted_lc_length_m=a[2])
            gm = glance_metrics(r['glances'], args.min_glance)
            row.update(gm)
            row.update(r['quality'])
            for k, v in list(row.items()):
                if isinstance(v, float):
                    row[k] = round(v, 4)
            w.writerow(row)
            print('%-10s %-5s %-24s %8.3f %8.3f %4d %5s %7s %7s' % (
                r['participant'], r['step'], r['map'][:24], r['mdev_basic'], r['mdev_adapted'],
                r['lc_errors'], gm.get('hu_glances_n', '-'),
                ('%.2f' % gm['hu_tgt_s']) if gm else '-',
                ('%.2f' % gm['hu_glance_max_s']) if gm else '-'))
            if args.plot:
                plot_run(r, os.path.splitext(r['file'])[0] + '_plot.png',
                         args.lc_start, args.lc_length, a)
    print('\nZusammenfassung gespeichert: %s' % args.out)


if __name__ == '__main__':
    main()

