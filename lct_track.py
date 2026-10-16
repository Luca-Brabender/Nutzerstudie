#!/usr/bin/env python

# Copyright (c) 2019 Intel Labs
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Welcome to CARLA manual control with steering wheel Logitech G27/G29 for LCT."""

from __future__ import print_function

import argparse
import collections
import csv
import datetime
from enum import Enum
import logging
import math
import os
import random
import re
import sys
import time
import weakref
import asyncio
import json
import threading
try:
    import websockets
except ImportError:
    websockets = None

import carla
from carla import ColorConverter as cc
import cv2
from lct_logger import LCTLogger
from lct_sign_manager import ALL_SEQUENCES
from aaos_recorder import AAOSRecorder, setup_external_hardware
from wheel_feedback import WheelForceFeedback, apply_hardware_ffb

if sys.version_info >= (3, 0):
    from configparser import ConfigParser
else:
    from ConfigParser import RawConfigParser as ConfigParser

try:
    import pygame
    from pygame.locals import (
        KMOD_CTRL,
        KMOD_SHIFT,
        K_0,
        K_9,
        K_BACKQUOTE,
        K_BACKSPACE,
        K_COMMA,
        K_DOWN,
        K_ESCAPE,
        K_F1,
        K_LEFT,
        K_PERIOD,
        K_RIGHT,
        K_SLASH,
        K_SPACE,
        K_TAB,
        K_UP,
        K_a,
        K_c,
        K_d,
        K_h,
        K_m,
        K_p,
        K_q,
        K_r,
        K_s,
        K_w,
    )
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================
# Eyetracking (WebGazer): Server, AOI-Klassifikation und Kalibrierung liegen in
# eigenen Modulen. Der Server wird erst im game_loop gestartet (nicht beim Import).
from gaze_server import GazeServer
from gaze_aoi import (AOIClassifier, GazeRecorder, Geometry, load_config,
                      AOI_UNKNOWN)
from gaze_calibration import GazeCalibration

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[: truncate - 1] + '\u2026') if len(name) > truncate else name


def find_audi_blueprint(blueprint_library, actor_filter='vehicle.audi.*'):
    """Wählt bevorzugt den Audi A2 (in CARLA der A1/A2-Kompaktklasse), TT oder ein Audi-Modell."""
    candidates = ['vehicle.audi.a2', 'vehicle.audi.a1', 'vehicle.audi.tt', 'vehicle.audi.*']
    for pattern in candidates:
        matches = blueprint_library.filter(pattern)
        if matches:
            # Deterministisch nach ID sortiert: in jedem Durchlauf exakt das gleiche Fahrzeug
            return sorted(matches, key=lambda x: x.id)[0]
    matches = blueprint_library.filter(actor_filter)
    return matches[0] if matches else blueprint_library.filter('vehicle.*')[0]


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):

    def __init__(self, carla_world, hud, actor_filter):
        self.world = carla_world
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = actor_filter
        self.restart()
        self.world.on_tick(hud.on_world_tick)

    def restart(self):
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = (
            self.camera_manager.transform_index if self.camera_manager is not None else 1
        )

        # WICHTIG: Alle bestehenden/herumstehenden Fahrzeuge & Geister-Autos in der Map entfernen!
        # Verhindert zuverlässig, dass in der Baseline ein zweites Auto unter/im Spieler spawnt.
        for actor in self.world.get_actors().filter('vehicle.*'):
            if self.player is None or actor.id != self.player.id:
                actor.destroy()

        blueprint = find_audi_blueprint(self.world.get_blueprint_library(), self._actor_filter)
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            recommended = blueprint.get_attribute('color').recommended_values
            # Einheitliches klassisches Audi-Silber / Metallic-Grau
            color = '160,160,160' if '160,160,160' in recommended else (recommended[0] if recommended else '0,0,0')
            blueprint.set_attribute('color', color)

        if self.player is not None:
            spawn_point = self.player.get_transform()
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)

        spawn_points = self.world.get_map().get_spawn_points()
        sp_idx = 0
        while self.player is None and sp_idx < len(spawn_points):
            self.player = self.world.try_spawn_actor(blueprint, spawn_points[sp_idx])
            sp_idx += 1

        if self.player is None:
            self.player = self.world.try_spawn_actor(blueprint, carla.Transform())

        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)
        print(f"[CARLA WORLD] Spieler-Fahrzeug gespawnt: {blueprint.id} ({actor_type})")

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy(self):
        sensors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
        ]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()


# ==============================================================================
# -- DualControl ---------------------------------------------------------------
# ==============================================================================


class DualControl(object):

    def __init__(self, world, start_in_autopilot, args=None):
        self._autopilot_enabled = start_in_autopilot
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            if self._autopilot_enabled:
                try:
                    world.player.set_autopilot(True)
                except RuntimeError as e:
                    print(f"[WARNHINWEIS] Autopilot konnte nicht gebunden werden: {e}")
            else:
                try:
                    world.player.set_autopilot(False)
                except RuntimeError:
                    pass
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError('Actor type not supported')
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        if joystick_count > 1:
            raise ValueError('Please Connect Just One Joystick')

        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()

        self._parser = ConfigParser()
        possible_paths = [
            'wheel_config.ini',
            os.path.join(os.path.dirname(__file__), 'wheel_config.ini'),
            os.path.join(os.path.dirname(__file__), '..', '..', 'wheel_config.ini'),
            '/home/moxdlab/Dokumente/Bachelorarbeit_Brabender/carla/wheel_config.ini',
        ]

        loaded = False
        for path in possible_paths:
            if os.path.exists(path):
                self._parser.read(path)
                loaded = True
                print(f'wheel_config.ini gefunden unter: {path}')
                break

        device_name = self._joystick.get_name()
        print(f'Erkanntes Eingabegerät: {device_name}')

        wheel_section = None
        if loaded:
            if self._parser.has_section(device_name):
                wheel_section = device_name
            elif self._parser.has_section('G27 Racing Wheel'):
                wheel_section = 'G27 Racing Wheel'
            elif self._parser.has_section('G29 Racing Wheel'):
                wheel_section = 'G29 Racing Wheel'

        if wheel_section:
            print(f'Verwende Konfigurations-Profil: [{wheel_section}]')
            self._steer_idx = int(self._parser.get(wheel_section, 'steering_wheel'))
            self._throttle_idx = int(self._parser.get(wheel_section, 'throttle'))
            self._brake_idx = int(self._parser.get(wheel_section, 'brake'))
            self._reverse_idx = int(self._parser.get(wheel_section, 'reverse'))
            self._handbrake_idx = int(self._parser.get(wheel_section, 'handbrake'))
        else:
            print('Verwende Standard-Belegung für Logitech G27...')
            self._steer_idx = 0
            self._throttle_idx = 1
            self._brake_idx = 2
            self._reverse_idx = 4
            self._handbrake_idx = 5

        # Force-Feedback Initialisierung (Lenkwinkel, Daempfung, Reibung, dynamisches Rueckstellmoment)
        ffb_kwargs = {}
        if wheel_section:
            for k, conv in [
                ('steering_range', int),
                ('overall_gain', float),
                ('centering_strength', float),
                ('autocenter', float),
                ('center_spring', float),
                ('min_force', float),
                ('damping_strength', float),
                ('friction_strength', float),
                ('anti_clipping', lambda v: v.lower() in ('true', '1', 'yes')),
            ]:
                if self._parser.has_option(wheel_section, k):
                    try:
                        ffb_kwargs[k] = conv(self._parser.get(wheel_section, k))
                    except Exception:
                        pass

        # Befehlszeilen-Argumente haben Vorrang (falls explizit übergeben)
        if args is not None:
            if getattr(args, 'autocenter', None) is not None:
                ffb_kwargs['autocenter'] = args.autocenter
            if getattr(args, 'ffb_gain', None) is not None:
                ffb_kwargs['overall_gain'] = args.ffb_gain
            if getattr(args, 'wheel_range', None) is not None:
                ffb_kwargs['steering_range'] = args.wheel_range
            if getattr(args, 'center_spring', None) is not None:
                ffb_kwargs['center_spring'] = args.center_spring
            if getattr(args, 'min_force', None) is not None:
                ffb_kwargs['min_force'] = args.min_force

        try:
            self._ffb = WheelForceFeedback(**ffb_kwargs)
        except Exception as e:
            print(f"[WARNHINWEIS] Force-Feedback Motor konnte nicht initialisiert werden: {e}")
            self._ffb = None

    def destroy(self):
        """Beendet den FFB-Motor sauber und stellt Standardwerte wieder her."""
        if self._ffb is not None:
            self._ffb.close()
            self._ffb = None

    def parse_events(self, world, clock):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.JOYBUTTONDOWN:
                # Button 0 auf dem Lenkrad soll im Versuch KEIN unabsichtliches world.restart() ausloesen!
                if event.button == 1:
                    world.hud.toggle_info()
                elif event.button == 2:
                    world.camera_manager.toggle_camera()
                elif event.button == 3:
                    world.next_weather()
                elif event.button == self._reverse_idx:
                    self._control.gear = 1 if self._control.reverse else -1
                elif event.button == 23:
                    world.camera_manager.next_sensor()

            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_h or (
                    event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT
                ):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r:
                    world.camera_manager.toggle_recording()
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification(
                            '%s Transmission'
                            % ('Manual' if self._control.manual_gear_shift else 'Automatic')
                        )
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p:
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification(
                            'Autopilot %s' % ('On' if self._autopilot_enabled else 'Off')
                        )

        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self._parse_vehicle_wheel()
                self._control.reverse = self._control.gear < 0
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time())
            world.player.apply_control(self._control)

            # Dynamisches Force Feedback an den internen Lenkrad-Motor uebertragen
            if self._ffb is not None and isinstance(world.player, carla.Vehicle):
                vel = world.player.get_velocity()
                speed_kmh = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                acc = world.player.get_acceleration()
                lat_accel = acc.y
                self._ffb.update(self._control.steer, speed_kmh, lateral_accel=lat_accel)

    def _parse_vehicle_keys(self, keys, milliseconds):
        self._control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

    def _parse_vehicle_wheel(self):
        numAxes = self._joystick.get_numaxes()
        jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]
        jsButtons = [
            float(self._joystick.get_button(i))
            for i in range(self._joystick.get_numbuttons())
        ]

        K1 = 1.0
        steerCmd = K1 * math.tan(1.1 * jsInputs[self._steer_idx])

        K2 = 1.6
        throttleCmd = (
            K2
            + (2.05 * math.log10(-0.7 * jsInputs[self._throttle_idx] + 1.4) - 1.2) / 0.92
        )
        if throttleCmd <= 0:
            throttleCmd = 0
        elif throttleCmd > 1:
            throttleCmd = 1

        brakeCmd = (
            1.6
            + (2.05 * math.log10(-0.7 * jsInputs[self._brake_idx] + 1.4) - 1.2) / 0.92
        )
        if brakeCmd <= 0:
            brakeCmd = 0
        elif brakeCmd > 1:
            brakeCmd = 1

        self._control.steer = steerCmd
        self._control.brake = brakeCmd
        self._control.throttle = throttleCmd
        self._control.hand_brake = bool(jsButtons[self._handbrake_idx])

    def _parse_walker_keys(self, keys, milliseconds):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = 0.01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = 0.01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = 5.556 if pygame.key.get_mods() & KMOD_SHIFT else 2.778
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):

    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, world_snapshot):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = world_snapshot.frame
        self.simulation_time = world_snapshot.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.world.get_map().name.split('/')[-1],
            'Method:  % 20s' % getattr(self, 'current_method_name', 'None'),
            'Simulation time: % 12s'
            % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (t.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s'
            % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '',
        ]
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                ('Gear:       %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)),
            ]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [('Speed:', c.speed, 0.0, 5.556), ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles),
        ]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt(
                (l.x - t.location.x) ** 2
                + (l.y - t.location.y) ** 2
                + (l.z - t.location.z) ** 2
            )
            vehicles = [
                (distance(x.get_location()), x)
                for x in vehicles
                if x.id != world.player.id
            ]
            for d, vehicle in sorted(vehicles):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [
                            (x + 8, v_offset + 8 + (1.0 - y) * 30)
                            for x, y in enumerate(item)
                        ]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(
                            display, (255, 255, 255), rect, 0 if item[1] else 1
                        )
                    else:
                        rect_border = pygame.Rect(
                            (bar_h_offset, v_offset + 8), (bar_width, 6)
                        )
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + f * (bar_width - 6), v_offset + 8),
                                (6, 6),
                            )
                        else:
                            rect = pygame.Rect(
                                (bar_h_offset, v_offset + 8), (f * bar_width, 6)
                            )
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):

    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):

    def __init__(self, font, width, height):
        doc_text = __doc__ if __doc__ is not None else 'LCT Track Simulation'
        lines = doc_text.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (
            0.5 * width - 0.5 * self.dim[0],
            0.5 * height - 0.5 * self.dim[1],
        )
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):

    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CollisionSensor._on_collision(weak_self, event)
        )

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor (Mit Event-Puffer für den Logger) ----------------------
# ==============================================================================


class LaneInvasionSensor(object):

    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        self.last_event_text = 'None'  # Standard: Keine Berührung
        
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: LaneInvasionSensor._on_invasion(weak_self, event)
        )

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return

        crossed_types = [marking.type for marking in event.crossed_lane_markings]

        # Differenzierung: Durchgezogen vs. Gestrichelt
        if carla.LaneMarkingType.Solid in crossed_types:
            line_kind = 'Solid'
        else:
            line_kind = 'Broken'

        # Bestimme die Richtung (Links vs. Rechts) anhand der Querposition
        p_loc = self._parent.get_transform().location
        # In deinem CARLA-Koordinatensystem: Y > 0 ist Links, Y < 0 ist Rechts
        direction = 'Right' if p_loc.y > -1.775 else 'Left'

        # Z. B. "Left_Broken", "Right_Solid"
        event_str = f"{direction}_{line_kind}"
        
        self.last_event_text = event_str
        self.hud.notification(f"LANE INVASION: {event_str}")
        print(f"[LANE INVASION] Reifen hat Linie berührt: {event_str}")

    def pop_event(self):
        """Liest das letzte Event aus und setzt es für den nächsten Frame zurück."""
        event = self.last_event_text
        self.last_event_text = "None"
        return event


# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================


class GnssSensor(object):

    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(carla.Location(x=1.0, z=2.8)),
            attach_to=self._parent,
        )
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: GnssSensor._on_gnss_event(weak_self, event)
        )

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):

    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        self._camera_transforms = [
            carla.Transform(
                carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)
            ),
            carla.Transform(carla.Location(x=1.6, z=1.7)),
        ]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            [
                'sensor.camera.depth',
                cc.LogarithmicDepth,
                'Camera Depth (Logarithmic Gray Scale)',
            ],
            [
                'sensor.camera.semantic_segmentation',
                cc.Raw,
                'Camera Semantic Segmentation (Raw)',
            ],
            [
                'sensor.camera.semantic_segmentation',
                cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)',
            ],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)'],
        ]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                bp.set_attribute('fov', '105')  # FOV auf 105 Grad
            elif item[0].startswith('sensor.lidar'):
                bp.set_attribute('range', '50')
            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(
            self._camera_transforms
        )
        self.sensor.set_transform(self._camera_transforms[self.transform_index])

    def set_sensor(self, index, notify=True):
        index = index % len(self.sensors)
        needs_respawn = (
            True
            if self.index is None
            else self.sensors[index][0] != self.sensors[self.index][0]
        )
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index],
                attach_to=self._parent,
            )
            weak_self = weakref.ref(self)
            self.sensor.listen(
                lambda image: CameraManager._parse_image(weak_self, image)
            )
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)


# ==============================================================================
# -- LCT Multi-Run Sequenz, Interaktionsmethoden & Counterbalancing ------------
# ==============================================================================

# Exakte Ordner-/Map-Namen in deiner Unreal Engine Map-Struktur
RUN_MAPS = [
    'lct_track_Baseline',
    'lct_track_dualtask_1',
    'lct_track_dualtask_2',
    'lct_track_dualtask_3'
]

SIGN_SPACING_M = 150.0
N_SIGNS = 18


class TaskMethod(Enum):
    """Interaktionsmethoden für die Sekundäraufgabe auf Android Automotive."""
    BASELINE = "Baseline"
    TOUCH = "Touch"
    ROTARY = "Drehknopf"
    MULTIKNOB = "MultiKnob"

    def label(self):
        labels = {
            TaskMethod.BASELINE: "Baseline (Nur Fahren - ohne Nebenaufgabe)",
            TaskMethod.TOUCH: "Touch (Display-Toucheingabe)",
            TaskMethod.ROTARY: "Drehknopf (Standard-Drehregler)",
            TaskMethod.MULTIKNOB: "MultiKnob (Smart Rotary mit Fingererkennung)",
        }
        return labels.get(self, self.value)

    def description(self):
        descs = {
            TaskMethod.BASELINE: "Referenzfahrt nach ISO 26022: Volle Konzentration auf Fahr- und Spurwechselaufgabe.",
            TaskMethod.TOUCH: "Bedienung der Menü-App auf Android Automotive direkt über den Touchscreen.",
            TaskMethod.ROTARY: "Bedienung der Menü-App auf Android Automotive über den Standard-Drehknopf.",
            TaskMethod.MULTIKNOB: "Bedienung der Menü-App über den MultiKnob (erkennt Anzahl der Finger).",
        }
        return descs.get(self, "")


# Vollständiges Counterbalancing der 3 Sekundäraufgaben-Methoden (3! = 6 Permutationen)
# Balanciert Reihenfolge- und Ermüdungseffekte (Learning/Fatigue/Carryover Effects)
COUNTERBALANCING_PERMUTATIONS = [
    [TaskMethod.TOUCH, TaskMethod.ROTARY, TaskMethod.MULTIKNOB],
    [TaskMethod.TOUCH, TaskMethod.MULTIKNOB, TaskMethod.ROTARY],
    [TaskMethod.ROTARY, TaskMethod.TOUCH, TaskMethod.MULTIKNOB],
    [TaskMethod.ROTARY, TaskMethod.MULTIKNOB, TaskMethod.TOUCH],
    [TaskMethod.MULTIKNOB, TaskMethod.TOUCH, TaskMethod.ROTARY],
    [TaskMethod.MULTIKNOB, TaskMethod.ROTARY, TaskMethod.TOUCH],
]


def resolve_study_methods(participant_id, order_list, manual_methods_str=None):
    """Ermittelt für jeden Schritt der Versuchsreihe die zugewiesene Interaktionsmethode.

    - Schritte mit Map 'lct_track_Baseline' (Map-Index 0) sind immer TaskMethod.BASELINE.
    - Die 3 Dual-Task-Strecken werden nach Probanden-ID per Counterbalancing balanciert.
    - Manuelle Vorgabe über manual_methods_str (z.B. 'Touch,Drehknopf,MultiKnob') möglich.
    """
    if manual_methods_str:
        method_map = {m.value.lower(): m for m in TaskMethod}
        parsed = []
        for name in manual_methods_str.split(','):
            key = name.strip().lower()
            if key in method_map:
                parsed.append(method_map[key])
            else:
                raise ValueError(
                    f"Unbekannte Methode '{name}'. Erlaubt: Touch, Drehknopf, MultiKnob, Baseline"
                )
        dual_methods = parsed
    else:
        digits = re.findall(r'\d+', participant_id)
        p_num = int(digits[0]) if digits else 1
        perm_idx = (p_num - 1) % len(COUNTERBALANCING_PERMUTATIONS)
        dual_methods = COUNTERBALANCING_PERMUTATIONS[perm_idx]

    methods_for_steps = []
    dual_idx = 0
    for map_idx in order_list:
        if map_idx == 0:
            methods_for_steps.append(TaskMethod.BASELINE)
        else:
            methods_for_steps.append(dual_methods[dual_idx % len(dual_methods)])
            dual_idx += 1
    return methods_for_steps


def show_study_ready_screen(display, participant, step_num, total_steps, method, map_name):
    """Zeigt vor jedem Lauf einen übersichtlichen Anweisungs-Bildschirm im Pygame-Fenster,
    damit Versuchsleiter & Proband wissen, welche Methode ansteht, und die App vorbereiten können."""
    clock = pygame.time.Clock()
    font_large = pygame.font.Font(pygame.font.get_default_font(), 30)
    font_medium = pygame.font.Font(pygame.font.get_default_font(), 22)
    font_small = pygame.font.Font(pygame.font.get_default_font(), 17)

    bg_color = (20, 24, 33)
    header_color = (40, 80, 140)
    card_bg = (32, 38, 52)
    accent_color = (255, 204, 0)
    inst_color = (110, 220, 130)

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key in (pygame.K_SPACE, pygame.K_RETURN):
                    waiting = False
            elif event.type == pygame.JOYBUTTONDOWN:
                waiting = False

        w, h = display.get_size()
        display.fill(bg_color)

        # Header Leiste
        pygame.draw.rect(display, header_color, (0, 0, w, 70))
        title = font_large.render(
            f"LCT STUDIE: PROBAND {participant}  |  LAUF {step_num} VON {total_steps}",
            True, (255, 255, 255)
        )
        display.blit(title, (w // 2 - title.get_width() // 2, 18))

        y = 110
        # Box mit Methode
        box_w = min(920, w - 80)
        box_h = 135
        box_x = w // 2 - box_w // 2
        pygame.draw.rect(display, card_bg, (box_x, y, box_w, box_h), border_radius=10)
        pygame.draw.rect(display, accent_color, (box_x, y, box_w, box_h), width=2, border_radius=10)

        badge_txt = font_small.render("AKTUELLE INTERAKTIONSMETHODE (SECONDARY TASK):", True, (200, 200, 200))
        display.blit(badge_txt, (box_x + 25, y + 15))

        m_name = font_large.render(method.label(), True, accent_color)
        display.blit(m_name, (box_x + 25, y + 42))

        m_desc = font_small.render(method.description(), True, (230, 230, 230))
        display.blit(m_desc, (box_x + 25, y + 88))

        y += box_h + 35

        # Hinweise Box
        map_txt = font_medium.render(f"Strecke: {map_name} (ISO 26022, 18 Schild-Manöver, 60 km/h)", True, (220, 220, 220))
        display.blit(map_txt, (w // 2 - map_txt.get_width() // 2, y))
        y += 45

        if method == TaskMethod.BASELINE:
            act_txt = font_medium.render("-> Keine Nebenaufgabe. Bitte nur auf das Fahren & Spurwechseln konzentrieren.", True, inst_color)
        else:
            act_txt = font_medium.render(f"-> Versuchsleiter: Bitte Android Automotive App auf '{method.value}' vorbereiten!", True, inst_color)
        display.blit(act_txt, (w // 2 - act_txt.get_width() // 2, y))
        y += 55

        start_prompt = font_large.render("Drücke [ LEERTASTE ] oder [ LENKRAD-TASTE ] zum Losfahren", True, (255, 255, 255))
        display.blit(start_prompt, (w // 2 - start_prompt.get_width() // 2, y))

        pygame.display.flip()
        clock.tick(30)
    return True


def show_study_finished_screen(display, participant, total_steps):
    """Zeigt am Ende aller 4 Strecken einen Dankeschön-Bildschirm für den Probanden.
    Wird mit [ ENTER ] (oder Leertaste / Lenkrad-Taste) bestätigt und beendet die Anwendung sauber."""
    clock = pygame.time.Clock()
    font_large = pygame.font.Font(pygame.font.get_default_font(), 32)
    font_medium = pygame.font.Font(pygame.font.get_default_font(), 22)
    font_small = pygame.font.Font(pygame.font.get_default_font(), 17)

    bg_color = (20, 24, 33)
    header_color = (34, 139, 34)  # Grün
    card_bg = (32, 38, 52)
    accent_color = (255, 215, 0)  # Gold
    text_color = (240, 240, 240)

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_ESCAPE, pygame.K_SPACE):
                    waiting = False
            elif event.type == pygame.JOYBUTTONDOWN:
                waiting = False

        w, h = display.get_size()
        display.fill(bg_color)

        # Header Leiste
        pygame.draw.rect(display, header_color, (0, 0, w, 70))
        title = font_large.render("STUDIE ERFOLGREICH ABGESCHLOSSEN!", True, (255, 255, 255))
        display.blit(title, (w // 2 - title.get_width() // 2, 18))

        y = 110
        box_w = min(920, w - 80)
        box_h = 230
        box_x = w // 2 - box_w // 2
        pygame.draw.rect(display, card_bg, (box_x, y, box_w, box_h), border_radius=10)
        pygame.draw.rect(display, header_color, (box_x, y, box_w, box_h), width=2, border_radius=10)

        t1 = font_large.render("Vielen Dank für deine Teilnahme!", True, accent_color)
        display.blit(t1, (w // 2 - t1.get_width() // 2, y + 25))

        t2 = font_medium.render(
            f"Proband {participant}: Alle {total_steps} Streckenabschnitte wurden erfolgreich absolviert.",
            True, text_color
        )
        display.blit(t2, (w // 2 - t2.get_width() // 2, y + 80))

        t3 = font_medium.render(
            "Alle Fahrdaten, Eyetracking-Blicke und Bildschirmaufnahmen wurden sicher gespeichert.",
            True, (170, 230, 170)
        )
        display.blit(t3, (w // 2 - t3.get_width() // 2, y + 125))

        t4 = font_small.render(
            "Der Versuch ist damit beendet. Du kannst das Lenkrad nun loslassen.",
            True, (200, 200, 200)
        )
        display.blit(t4, (w // 2 - t4.get_width() // 2, y + 175))

        y += box_h + 45
        prompt = font_large.render("Drücke [ ENTER ] zum Beenden", True, (255, 255, 255))
        display.blit(prompt, (w // 2 - prompt.get_width() // 2, y))

        pygame.display.flip()
        clock.tick(30)


def next_sign_idx(x):
    """Index des nächsten Schildes (wie im LCTLogger: Schilder bei 150, 300, ... m)."""
    idx = int(max(0.0, x) // SIGN_SPACING_M)
    return idx if idx < N_SIGNS else -1


class StudyEvents(object):
    """Fortlaufendes Ereignisprotokoll pro Proband (Unix-Zeit -> Synchronisation
    von LCT-Log, Gaze-Log und AAOS-Bildschirmvideo)."""

    def __init__(self, path, participant, step, map_name, method=''):
        new = not os.path.exists(path)
        self._fh = open(path, 'a', newline='')
        self._w = csv.writer(self._fh)
        if new:
            self._w.writerow(['unix_time', 'iso_time', 'participant', 'step', 'method', 'map',
                              'event', 'detail'])
        self.participant, self.step, self.map_name, self.method = participant, step, map_name, method

    def log(self, event, detail=''):
        if self._fh is None:
            return
        now = time.time()
        self._w.writerow(['%.4f' % now,
                          datetime.datetime.fromtimestamp(now).isoformat(timespec='milliseconds'),
                          self.participant, self.step + 1, self.method, self.map_name, event, detail])
        self._fh.flush()

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def _restart_argv(next_step):
    """Argumente für den Neustart des Prozesses (nächster Schritt der Reihenfolge)."""
    clean_argv = []
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg == '--run_idx':
            skip_next = True
            continue
        if arg.startswith('--run_idx=') or arg == '--recalibrate':
            continue
        clean_argv.append(arg)
    return [sys.executable, sys.argv[0]] + clean_argv + ['--run_idx', str(next_step)]


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    lct_logger = None
    aaos_recorder = None
    gaze = None
    gaze_rec = None
    recording_triggered = False

    # Schritt innerhalb der (permutierten) Versuchsreihenfolge
    order = args.order_list
    step = args.run_idx
    if step >= len(order):
        print('[LCT STUDY] run_idx %d außerhalb der Reihenfolge %s' % (step, order))
        return
    map_idx = order[step]
    current_map_name = RUN_MAPS[map_idx]
    current_sequence = ALL_SEQUENCES[map_idx]

    # Interaktionsmethode für diesen Schritt (per Counterbalancing oder explizit)
    methods_for_run = resolve_study_methods(args.participant, order, getattr(args, 'methods', None))
    current_method = methods_for_run[step]

    # Ausgabe: logs/<Proband>/<Proband>_step<N>_<Methode>_<Map>_<Zeitstempel>_*.csv
    participant_dir = os.path.join(args.log_dir, args.participant)
    os.makedirs(participant_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    run_prefix = os.path.join(
        participant_dir,
        '%s_step%d_%s_%s_%s' % (args.participant, step + 1, current_method.value, current_map_name, stamp))
    events = StudyEvents(os.path.join(participant_dir, '%s_events.csv' % args.participant),
                         args.participant, step, current_map_name, method=current_method.value)
    events.log('process_start', 'order=%s, method=%s' % (args.order, current_method.value))
    print('\n' + '=' * 72)
    print('[LCT STUDY] Proband %s | Schritt %d/%d' % (args.participant, step + 1, len(order)))
    print('[LCT STUDY] Map:       %s' % current_map_name)
    print('[LCT STUDY] METHODE:   %s (%s)' % (current_method.value, current_method.label()))
    print('[LCT STUDY] Anleitung: %s' % current_method.description())
    print('=' * 72 + '\n')

    # Sofortige Hardware-Konfiguration fuer das Logitech-Lenkrad (Zentrierfeder, Gain, Lenkwinkel)
    # Damit hat das Lenkrad schon vor der Kalibrierung und dem Map-Laden das richtige Gefühl!
    ac_val = getattr(args, 'autocenter', None) or 75
    gain_val = getattr(args, 'ffb_gain', None) or 85
    range_val = getattr(args, 'wheel_range', None) or 900
    apply_hardware_ffb(autocenter_pct=ac_val, gain_pct=gain_val, range_deg=range_val)

    # Nur beim allerersten Schritt den Hardware-Check ausführen
    if step == 0 and not getattr(args, 'no_aaos', False):
        setup_external_hardware(device_ip=getattr(args, 'aaos_ip', '192.168.0.103'))

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)

        # Prüfen und Laden der jeweiligen Map für den aktuellen Run
        world_map = client.get_world().get_map().name.split('/')[-1]
        if world_map != current_map_name:
            print(f"\n[LCT STUDY] Lade Map {step + 1}/{len(order)}: {current_map_name}...")
            client.load_world(current_map_name)

        # Vor jedem Lauf: Bestehende Fahrzeuge und Sensoren in der Map rückstandslos bereinigen
        # (Verhindert Geister-Fahrzeuge, Doppel-Spawns oder alte Autos unter dem Spieler)
        existing_world = client.get_world()
        for a in existing_world.get_actors().filter('vehicle.*'):
            a.destroy()
        for s in existing_world.get_actors().filter('sensor.*'):
            s.destroy()

        flags = pygame.HWSURFACE | pygame.DOUBLEBUF
        if args.fullscreen:
            flags |= pygame.FULLSCREEN
        display = pygame.display.set_mode((args.width, args.height), flags)
        pygame.display.set_caption('LCT %s - Schritt %d/%d: %s [%s]' % (
            args.participant, step + 1, len(order), current_map_name, current_method.value))

        # ------------------------------------------------------------------
        # Eyetracking: Server starten, (Re-)Kalibrierung vor jeder Fahrt
        # ------------------------------------------------------------------
        clf = None
        gaze_cfg = None
        if not args.no_gaze:
            gaze = GazeServer()
            gaze.start()
            gaze_cfg = load_config(args.gaze_config)
            geo = Geometry(gaze_cfg, args.width, args.height)
            clf = AOIClassifier(gaze_cfg, geo)
            clf_path = os.path.join(participant_dir, '%s_gaze_classifier.json' % args.participant)
            if not args.recalibrate:
                clf.load(clf_path)
            calib = GazeCalibration(display, gaze, geo, clf, gaze_cfg)
            if calib.run(allow_reuse=not args.recalibrate):
                clf.save(clf_path)
                with open(run_prefix + '_calibration.json', 'w') as fh:
                    json.dump({'validation': calib.report,
                               'classifier': clf.train_report,
                               'headunit_rect_px': geo.hu_rect,
                               'window_px': [args.width, args.height],
                               'config': gaze_cfg}, fh, indent=2)
                events.log('gaze_ready', json.dumps(calib.report))
            else:
                events.log('gaze_disabled', 'Kalibrierung übersprungen/abgebrochen')
                gaze = None

        hud = HUD(args.width, args.height)
        hud.current_method_name = current_method.value
        world = World(client.get_world(), hud, args.filter)
        controller = DualControl(world, args.autopilot, args=args)

        # LCT Logger Instanz erzeugen
        START_X = 0.0
        FINISH_X = 3000.0  # Ziellinie nach dem 18. Schild
        lct_logger = LCTLogger(start_x=START_X, finish_x=FINISH_X,
                               output_path=run_prefix + '_lct.csv',
                               method=current_method.value)

        # AAOS Recorder initialisieren (Zeichnet RPi 4 Android Automotive via scrcpy auf)
        aaos_recorder = None
        if not getattr(args, 'no_aaos', False):
            aaos_recorder = AAOSRecorder(
                output_dir=participant_dir,
                device_ip=getattr(args, 'aaos_ip', '192.168.0.103'),
                max_fps=60,
                bit_rate="2M",
                max_size=800,
            )

        # Bereit-Bildschirm vor Fahrtbeginn anzeigen (Versuchsleiter/Proband bestätigen Start)
        if not show_study_ready_screen(display, args.participant, step + 1, len(order), current_method, current_map_name):
            events.log('abort', 'Abbruch vor Fahrtbeginn')
            return

        # AAOS Aufnahme VOR dem Losfahren starten (verhindert jegliche Ruckler an der Startlinie!)
        if aaos_recorder is not None:
            aaos_recorder.start_recording(run_prefix=run_prefix, run_idx=step)
            events.log('aaos_record_start')

        clock = pygame.time.Clock()
        gaze_x, gaze_y = -1.0, -1.0
        gaze_state = None
        last_status_push = 0.0

        # Variablen fuer Out-of-Bounds / Map-Absturz Recovery (Teleport zurueck auf die Fahrbahn)
        MIDDLE_LANE_Y = -1.8875  # Mitte der mittleren Spur in CARLA (ISO 26022, pos_y_norm = 5.6625 m)
        last_safe_x = 0.0
        last_safe_z = None
        last_safe_speed_kmh = 50.0
        last_recovery_time = 0.0

        while True:
            clock.tick_busy_loop(60)
            if controller.parse_events(world, clock):
                events.log('abort', 'Abbruch durch Benutzer')
                return

            now = time.time()
            player_loc = world.player.get_transform().location if world.player else None

            # 0. Eyetracking-Samples dieses Frames verarbeiten
            if gaze is not None:
                samples = gaze.drain()
                status_aoi = None
                if gaze_rec is not None:
                    if player_loc is not None:
                        gaze_rec.set_context(now, player_loc.x, next_sign_idx(player_loc.x))
                    gaze_rec.process(samples)
                    gaze_state = {
                        'aoi_state': gaze_rec.state,
                        'aoi_raw': gaze_rec.last_raw,
                        'p_headunit': gaze_rec.last_p,
                        'hu_glance_id': gaze_rec.glance_id if gaze_rec.in_hu_glance else 0,
                    }
                    status_aoi = gaze_rec.state
                elif samples:
                    status_aoi = clf.classify(samples[-1])[0]
                latest = gaze.latest
                if latest is not None and latest.x is not None:
                    gaze_x, gaze_y = latest.x, latest.y
                else:
                    gaze_x, gaze_y = -1.0, -1.0
                if status_aoi and now - last_status_push > 0.2:
                    gaze.push_status(aoi=status_aoi)
                    last_status_push = now

            # 1. Lane Invasion Event auslesen
            invasion_event = "None"
            if world is not None and getattr(world, 'lane_invasion_sensor', None) is not None:
                invasion_event = world.lane_invasion_sensor.pop_event()

            # 2. Logger genau EINMAL pro Frame updaten & Blickdaten direkt übergeben
            if lct_logger is not None:
                lct_logger.update(
                    world,
                    current_sequence,
                    lane_event=invasion_event,
                    gaze_x=gaze_x,
                    gaze_y=gaze_y,
                    gaze_state=gaze_state,
                )

            if player_loc is not None and world.player is not None:
                player_trans = world.player.get_transform()
                vel = world.player.get_velocity()
                current_speed_kmh = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

                # Initiale Fahrbahnhöhe ermitteln
                if last_safe_z is None:
                    last_safe_z = player_loc.z

                # A) Normalen Straßenzustand erfassen (Fahrzeug fährt regulär auf der Fahrbahn)
                is_on_road = (
                    abs(player_loc.y - MIDDLE_LANE_Y) < 8.0 and
                    player_loc.z >= (last_safe_z - 1.5) and
                    abs(player_trans.rotation.pitch) < 35.0 and
                    abs(player_trans.rotation.roll) < 35.0
                )
                if is_on_road:
                    last_safe_x = player_loc.x
                    last_safe_z = player_loc.z
                    last_safe_speed_kmh = max(25.0, current_speed_kmh)

                # B) Map-Absturz / Void-Fall / Out-of-Bounds erkennen & nahtlos retten
                has_fallen = (
                    (player_loc.z < (last_safe_z - 3.5)) or
                    (player_loc.z < -10.0) or
                    (abs(player_loc.y - MIDDLE_LANE_Y) > 25.0) or
                    (vel.z < -12.0 and player_loc.z < (last_safe_z - 2.0))
                )

                if has_fallen and (now - last_recovery_time > 2.0):
                    last_recovery_time = now
                    respawn_x = max(0.0, last_safe_x)
                    respawn_y = MIDDLE_LANE_Y
                    respawn_z = (last_safe_z if last_safe_z is not None else 0.5) + 0.35
                    respawn_trans = carla.Transform(
                        carla.Location(x=respawn_x, y=respawn_y, z=respawn_z),
                        carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)  # Exakt geradeaus in Fahrtrichtung
                    )
                    world.player.set_transform(respawn_trans)

                    # Vorwärts-Fahrgeschwindigkeit wiederherstellen (nahe 60 km/h Zielgeschwindigkeit)
                    target_vx = min(16.67, max(10.0, last_safe_speed_kmh / 3.6))
                    world.player.set_target_velocity(carla.Vector3D(x=target_vx, y=0.0, z=0.0))
                    world.player.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))

                    ctrl = carla.VehicleControl(throttle=0.35, steer=0.0, brake=0.0, hand_brake=False)
                    world.player.apply_control(ctrl)

                    world.hud.notification("Fahrbahn verlassen: Auto auf mittlere Spur zurückgesetzt!")
                    events.log('recovery_teleport', f'x={respawn_x:.2f}, y={respawn_y:.2f}')
                    print(f"\n[RECOVERY] Fahrzeug fiel aus der Map! Nahtlos wiederhergestellt bei x={respawn_x:.2f}m auf mittlerer Spur.")

                # TRIGGER: STARTLINIE ÜBERSCHRITTEN (X >= 0.0m)
                if not recording_triggered and player_loc.x >= START_X:
                    recording_triggered = True
                    start_offset = 0.0
                    if aaos_recorder is not None:
                        start_offset = aaos_recorder.mark_start_line()
                    events.log('start_line', f'x={player_loc.x:.2f}, video_offset={start_offset:.3f}s')
                    print(f"\n[TRIGGER] Startlinie überschritten (x={player_loc.x:.2f}m)! AAOS Video-Offset: {start_offset:.2f}s")
                    if gaze is not None:
                        gaze_rec = GazeRecorder(clf, gaze_cfg, run_prefix)
                        gaze_rec.set_context(now, player_loc.x, next_sign_idx(player_loc.x))
                        events.log('gaze_record_start')

                # HARD SPEED LIMIT: Max. 60 km/h (ISO 26022 Vorgabe)
                if current_speed_kmh >= 60.0:
                    ctrl = world.player.get_control()
                    ctrl.throttle = 0.0
                    world.player.apply_control(ctrl)

                # ZIELLINIE ERREICHT (3000m)
                if player_loc.x >= FINISH_X:
                    print(f"\n[LCT STUDY] Ziellinie für Schritt {step + 1} überquert!")
                    events.log('finish_line', 'x=%.2f' % player_loc.x)

                    # A) Gaze-Logs schließen (offene Blickzuwendung wird abgeschlossen)
                    if gaze_rec is not None:
                        gaze_rec.close(t_end=time.time())
                        gaze_rec = None

                    # B) AAOS Recording stoppen & MP4 abschließen
                    if aaos_recorder is not None:
                        aaos_recorder.stop()
                        events.log('aaos_record_stop')
                        aaos_recorder = None

                    # C) Logger stoppen & CSV sofort speichern
                    if lct_logger is not None:
                        lct_logger.close()
                        lct_logger = None
                        print("[LOGGER] CSV-Datei erfolgreich abgespeichert.")
                    events.log('run_complete')

                    # D) Nächster Schritt der Reihenfolge
                    next_step = step + 1
                    if next_step < len(order):
                        next_map_name = RUN_MAPS[order[next_step]]
                        print(f"[LCT STUDY] Wechsle zu Schritt {next_step + 1}: {next_map_name}...")

                        if world is not None:
                            world.destroy()
                        if controller is not None:
                            controller.destroy()
                        pygame.quit()
                        events.close()

                        args_list = _restart_argv(next_step)
                        print(f"[LCT STUDY] Starte Prozess neu mit: {args_list}")
                        os.execv(sys.executable, args_list)
                    else:
                        print("\n==================================================")
                        print("[LCT STUDY] ALLE %d FAHRTEN ERFOLGREICH ABGESCHLOSSEN!" % len(order))
                        print("==================================================\n")
                        events.log('study_complete')

                        # Dankeschön-Screen anzeigen, bis der Proband ENTER drückt
                        show_study_finished_screen(display, args.participant, len(order))
                        return

            pygame.display.flip()
            world.tick(clock)
            world.render(display)
    finally:
        # Falls das Skript vorzeitig abgebrochen wird: Aufzeichnung & Logger absichern
        if controller is not None:
            controller.destroy()

        if gaze_rec is not None:
            gaze_rec.close(t_end=time.time())

        if aaos_recorder is not None and recording_triggered:
            aaos_recorder.stop()

        if lct_logger is not None:
            lct_logger.close()

        events.close()

        if world is not None:
            world.destroy()

        pygame.quit()


def main():
    argparser = argparse.ArgumentParser(description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information',
    )
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)',
    )
    argparser.add_argument(
        '-p',
        '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)',
    )
    argparser.add_argument(
        '-a', '--autopilot', action='store_true', help='enable autopilot'
    )
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)',
    )
    argparser.add_argument(
        '--fullscreen', action='store_true',
        help='Vollbild (empfohlen; --res sollte dann der Monitorauflösung entsprechen)',
    )
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.audi.*',
        help='actor filter (default: "vehicle.audi.*")',
    )
    argparser.add_argument(
        '--run_idx',
        default=0,
        type=int,
        help='Schritt innerhalb von --order (0-basiert, wird beim Neustart gesetzt)',
    )
    argparser.add_argument(
        '--participant', default='P00',
        help='Probanden-ID, z.B. P07 (bestimmt den Log-Ordner)',
    )
    argparser.add_argument(
        '--order', default='0,1,2,3',
        help='Reihenfolge der Maps als Indizes in RUN_MAPS, z.B. "0,2,3,1,0" '
             '(Permutation der Bedingungen, Baseline am Anfang und Ende nach ISO 26022)',
    )
    argparser.add_argument(
        '--methods', default=None,
        help='Optionale manuelle Vorgabe der Dual-Task-Methoden (z.B. "Touch,Drehknopf,MultiKnob"). '
             'Standard: Automatisch aus --participant per Counterbalancing (Lateinisches Quadrat) abgeleitet.',
    )
    argparser.add_argument(
        '--log-dir', dest='log_dir', default=os.path.join(SCRIPT_DIR, 'logs'),
        help='Basisordner für alle Logs',
    )
    argparser.add_argument(
        '--no-gaze', dest='no_gaze', action='store_true',
        help='Ohne Eyetracking fahren',
    )
    argparser.add_argument(
        '--gaze-config', dest='gaze_config',
        default=os.path.join(SCRIPT_DIR, 'gaze_config.json'),
        help='JSON mit Monitor- und Headunit-Geometrie',
    )
    argparser.add_argument(
        '--recalibrate', action='store_true',
        help='Gespeicherte Kalibrierung verwerfen und neu kalibrieren',
    )
    argparser.add_argument(
        '--aaos-ip', default='192.168.0.103',
        help='IP-Adresse des Raspberry Pi 4 (Android Automotive) für ADB/scrcpy (Standard: 192.168.0.103)',
    )
    argparser.add_argument(
        '--no-aaos', action='store_true',
        help='AAOS-Bildschirmaufnahme via scrcpy deaktivieren',
    )
    argparser.add_argument(
        '--autocenter', type=int, default=None,
        help='Hardware-Zentrierfeder des Lenkrads in %% (0-100, Standard: 75 in wheel_config.ini)',
    )
    argparser.add_argument(
        '--center-spring', type=int, default=None,
        help='Progressive Zentrierkraft um die Nulllage in %% (0-100, Standard: 55)',
    )
    argparser.add_argument(
        '--min-force', type=int, default=None,
        help='Mindestkraft / Anti-Deadzone gegen Getriebespiel in %% (0-100, Standard: 12)',
    )
    argparser.add_argument(
        '--ffb-gain', type=int, default=None,
        help='Gesamtkraft (Gain) des Lenkradmotors in %% (0-100, Standard: 85 in wheel_config.ini)',
    )
    argparser.add_argument(
        '--wheel-range', type=int, default=None,
        help='Maximaler Lenkwinkel in Grad (Standard: 900 in wheel_config.ini)',
    )
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]
    args.order_list = [int(x) for x in args.order.split(',') if x.strip() != '']
    for idx in args.order_list:
        if not 0 <= idx < len(RUN_MAPS):
            argparser.error('--order enthält ungültigen Map-Index %d' % idx)

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)
    print(__doc__)

    try:
        game_loop(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
