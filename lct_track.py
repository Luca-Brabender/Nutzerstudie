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
import websockets

import carla
from carla import ColorConverter as cc
import cv2
from lct_logger import LCTLogger
from lct_sign_manager import ALL_SEQUENCES
from aaos_recorder import AAOSRecorder, setup_external_hardware

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
            self.camera_manager.transform_index if self.camera_manager is not None else 0
        )
        blueprint = random.choice(
            self.world.get_blueprint_library().filter(self._actor_filter)
        )
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        while self.player is None:
            spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

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

    def __init__(self, world, start_in_autopilot):
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

    def parse_events(self, world, clock):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.JOYBUTTONDOWN:
                if event.button == 0:
                    world.restart()
                elif event.button == 1:
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
# -- LCT Multi-Run Sequenz & Logik --------------------------------------------
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


def next_sign_idx(x):
    """Index des nächsten Schildes (wie im LCTLogger: Schilder bei 150, 300, ... m)."""
    idx = int(max(0.0, x) // SIGN_SPACING_M)
    return idx if idx < N_SIGNS else -1


class StudyEvents(object):
    """Fortlaufendes Ereignisprotokoll pro Proband (Unix-Zeit -> Synchronisation
    von LCT-Log, Gaze-Log und AAOS-Bildschirmvideo)."""

    def __init__(self, path, participant, step, map_name):
        new = not os.path.exists(path)
        self._fh = open(path, 'a', newline='')
        self._w = csv.writer(self._fh)
        if new:
            self._w.writerow(['unix_time', 'iso_time', 'participant', 'step', 'map',
                              'event', 'detail'])
        self.participant, self.step, self.map_name = participant, step, map_name

    def log(self, event, detail=''):
        if self._fh is None:
            return
        now = time.time()
        self._w.writerow(['%.4f' % now,
                          datetime.datetime.fromtimestamp(now).isoformat(timespec='milliseconds'),
                          self.participant, self.step + 1, self.map_name, event, detail])
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

    # Ausgabe: logs/<Proband>/<Proband>_step<N>_<Map>_<Zeitstempel>_*.csv
    participant_dir = os.path.join(args.log_dir, args.participant)
    os.makedirs(participant_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    run_prefix = os.path.join(
        participant_dir,
        '%s_step%d_%s_%s' % (args.participant, step + 1, current_map_name, stamp))
    events = StudyEvents(os.path.join(participant_dir, '%s_events.csv' % args.participant),
                         args.participant, step, current_map_name)
    events.log('process_start', 'order=%s' % args.order)
    print('\n[LCT STUDY] Proband %s | Schritt %d/%d | Map %s'
          % (args.participant, step + 1, len(order), current_map_name))

    # Nur beim allerersten Schritt den Hardware-Check ausführen
    if step == 0:
        setup_external_hardware()

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)

        # Prüfen und Laden der jeweiligen Map für den aktuellen Run
        world_map = client.get_world().get_map().name.split('/')[-1]
        if world_map != current_map_name:
            print(f"\n[LCT STUDY] Lade Map {step + 1}/{len(order)}: {current_map_name}...")
            client.load_world(current_map_name)

        flags = pygame.HWSURFACE | pygame.DOUBLEBUF
        if args.fullscreen:
            flags |= pygame.FULLSCREEN
        display = pygame.display.set_mode((args.width, args.height), flags)
        pygame.display.set_caption('LCT %s - Schritt %d/%d: %s' % (
            args.participant, step + 1, len(order), current_map_name))

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
        world = World(client.get_world(), hud, args.filter)
        controller = DualControl(world, args.autopilot)

        # LCT Logger Instanz erzeugen
        START_X = 0.0
        FINISH_X = 3000.0  # Ziellinie nach dem 18. Schild
        lct_logger = LCTLogger(start_x=START_X, finish_x=FINISH_X,
                               output_path=run_prefix + '_lct.csv')

        # AAOS Recorder initialisieren
        aaos_recorder = AAOSRecorder(output_dir=participant_dir)

        clock = pygame.time.Clock()
        gaze_x, gaze_y = -1.0, -1.0
        gaze_state = None
        last_status_push = 0.0

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

            if player_loc is not None:

                # TRIGGER: STARTLINIE ÜBERSCHRITTEN (X >= 0.0m) -> AAOS RECORDING STARTEN
                if not recording_triggered and player_loc.x >= START_X:
                    recording_triggered = True
                    events.log('start_line', 'x=%.2f' % player_loc.x)
                    print("\n[TRIGGER] Startlinie überschritten! Starte AAOS UI-Aufzeichnung...")
                    if aaos_recorder is not None:
                        aaos_recorder.start_recording()
                        events.log('aaos_record_start')
                    if gaze is not None:
                        gaze_rec = GazeRecorder(clf, gaze_cfg, run_prefix)
                        gaze_rec.set_context(now, player_loc.x, next_sign_idx(player_loc.x))
                        events.log('gaze_record_start')

                # HARD SPEED LIMIT: Max. 60 km/h (ISO 26022 Vorgabe)
                vel = world.player.get_velocity()
                current_speed_kmh = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
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

                    # B) AAOS Recording stoppen & MP4 herunterladen
                    if aaos_recorder is not None:
                        aaos_recorder.stop_and_pull(step)
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
                        return

            pygame.display.flip()
            world.tick(clock)
            world.render(display)
    finally:
        # Falls das Skript vorzeitig abgebrochen wird: Aufzeichnung & Logger absichern
        if gaze_rec is not None:
            gaze_rec.close(t_end=time.time())

        if aaos_recorder is not None and recording_triggered:
            aaos_recorder.stop_and_pull(step)

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
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")',
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
