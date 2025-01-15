import numpy as np

MAX_AGENTS = 40

FPS = 50
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

MOTORS_TORQUE = 20 / 10
SPEED_HIP = 4
SPEED_KNEE = 6
LIDAR_RANGE = 160 / SCALE

INITIAL_RANDOM = 5

# (+34, +1),
HULL_POLY = [(-30, +9), (+6, +9), (+34, 1), (+34, -8), (-30, -8)]
LEG_DOWN = -8 / SCALE
LEG_W, LEG_H = 8 / SCALE, 34 / SCALE

PACKAGE_POLY = [(-120, 5), (120, 5), (120, -5), (-120, -5)]

PACKAGE_LENGTH = 240

VIEWPORT_W = 600
VIEWPORT_H = 400

TERRAIN_STEP = 14 / SCALE
TERRAIN_LENGTH = 200  # in steps
TERRAIN_HEIGHT = (VIEWPORT_H - 200) / SCALE / 5
TERRAIN_GRASS = 10  # low long are grass spots, in steps
TERRAIN_STARTPAD = 10  # in steps
FRICTION = 2.5

WALKER_SEPERATION = 10  # in steps

MW_COLORS = {
    "hull": [np.array([127, 51, 229]), np.array([76, 76, 127])],
    "leg:L": [np.array([178, 101, 152]), np.array([127, 76, 101])],
    "leg:R": [np.array([153, 76, 127]), np.array([102, 51, 76])],
}
