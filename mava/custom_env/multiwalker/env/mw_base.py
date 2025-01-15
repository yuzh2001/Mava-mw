import os

import imageio
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax2d.engine import PhysicsEngine, create_empty_sim
from jax2d.scene import (
    add_polygon_to_scene,
    add_rectangle_to_scene,
    add_revolute_joint_to_scene,
)
from jax2d.sim_state import SimParams, SimState, StaticSimParams

from mava.custom_env.multiwalker.env.mw_constants import (
    HULL_POLY,
    LEG_DOWN,
    LEG_H,
    LEG_W,
    PACKAGE_POLY,
    SCALE,
    TERRAIN_HEIGHT,
    TERRAIN_STARTPAD,
    TERRAIN_STEP,
)
from mava.custom_env.multiwalker.env.mw_render import (
    make_render_pixels,
    render_bridge,
)

os.environ["SDL_VIDEODRIVER"] = "dummy"


@struct.dataclass
class MW_StaticSimParams(StaticSimParams):
    # State size
    num_polygons: int = 12
    num_circles: int = 12
    num_joints: int = 12
    num_thrusters: int = 12
    max_polygon_vertices: int = 5

    # Compute amount
    num_solver_iterations: int = 10
    solver_batch_size: int = 16
    do_warm_starting: bool = True
    num_static_fixated_polys: int = 1


@struct.dataclass
class MW_SimParams(SimParams):
    # Timestep size
    dt: float = 1 / 60

    # Collision and joint coefficients
    slop: float = 0.01
    baumgarte_coefficient_joints_v: float = 2.0
    baumgarte_coefficient_joints_p: float = 0.7
    baumgarte_coefficient_fjoint_av: float = 2.0
    baumgarte_coefficient_rjoint_limit_av: float = 5.0
    baumgarte_coefficient_collision: float = 0.2
    joint_stiffness: float = 0.6

    # State clipping
    clip_position: float = 200
    clip_velocity: float = 100
    clip_angular_velocity: float = 50

    # Motors and thrusters
    base_motor_speed: float = 6.0  # rad/s
    base_motor_power: float = 900.0
    base_thruster_power: float = 10.0
    motor_decay_coefficient: float = 0.1
    motor_joint_limit: float = 0.1  # rad

    # Other defaults
    base_friction: float = 2.5


class BipedalWalker:
    def __init__(
        self,
        world,
        static_sim_params: StaticSimParams,
        init_x=TERRAIN_STEP * TERRAIN_STARTPAD / 2,
        init_y=TERRAIN_HEIGHT + 2 * LEG_H,
        n_walkers=3,
        seed=None,
    ):
        self.world = world
        self.static_sim_params = static_sim_params
        self._n_walkers = n_walkers
        self.init_x = init_x
        self.init_y = init_y
        self._seed(seed)

        # all parts
        self.hull_index = None
        self.leg_indexes = []
        self.joint_indexes = []

    def _seed(self, seed=None):
        self.key = jax.random.PRNGKey(seed if seed is not None else 0)
        self.uniform_fn = lambda min_v, max_v, **kwargs: jax.random.uniform(
            self.key, shape=(), minval=min_v, maxval=max_v, **kwargs
        )
        return [seed]

    def _reset(self, scene: SimState):
        # self._destroy()
        init_x = self.init_x
        init_y = self.init_y

        # add a hull
        hull_vertices = jnp.array([[x / SCALE, y / SCALE] for x, y in HULL_POLY])
        scene, (_, self.hull_index) = add_polygon_to_scene(
            scene,
            self.static_sim_params,
            position=jnp.array([init_x, init_y]),
            vertices=hull_vertices,
            n_vertices=len(HULL_POLY),
            density=1.0,
            friction=1,
            restitution=0.0,
        )
        # add the legs
        self.leg_indexes = []
        self.joint_indexes = []
        for i in [-1, +1]:
            scene, (_, leg_index) = add_rectangle_to_scene(
                scene,
                self.static_sim_params,
                position=jnp.array([init_x, init_y - LEG_H / 2 - LEG_DOWN]),
                dimensions=jnp.array([LEG_W, LEG_H]),
                density=1.0,
                restitution=0.0,
                rotation=i * 0.05,
            )
            motor_speed = 3
            motor_power = 2
            scene, leg_hull_joint_index = add_revolute_joint_to_scene(
                scene,
                self.static_sim_params,
                a_index=self.hull_index,
                b_index=leg_index,
                a_relative_pos=jnp.array([0.0, LEG_DOWN]),
                b_relative_pos=jnp.array([0.0, LEG_H / 2]),
                motor_on=True,
                has_joint_limits=True,
                min_rotation=-0.8,
                max_rotation=1.1,
                motor_speed=motor_speed,
                motor_power=motor_power,
            )

            self.leg_indexes.append(leg_index)
            self.joint_indexes.append(leg_hull_joint_index)

            scene, (_, leg_lower_index) = add_rectangle_to_scene(
                scene,
                self.static_sim_params,
                position=jnp.array([init_x, init_y - LEG_H * 3 / 2 - LEG_DOWN]),
                dimensions=jnp.array([0.8 * LEG_W, LEG_H]),
                density=1.0,
                restitution=0.0,
                rotation=i * 0.05,
            )

            scene, leg_lower_joint_index = add_revolute_joint_to_scene(
                scene,
                self.static_sim_params,
                a_index=leg_index,
                b_index=leg_lower_index,
                a_relative_pos=jnp.array([0.0, -LEG_H / 2]),
                b_relative_pos=jnp.array([0.0, LEG_H / 2]),
                motor_on=True,
                motor_speed=motor_speed,
                motor_power=motor_power,
                has_joint_limits=True,
                min_rotation=-1.6,
                max_rotation=-0.1,
            )
            self.leg_indexes.append(leg_lower_index)
            self.joint_indexes.append(leg_lower_joint_index)

        return scene

    def setup(self, scene: SimState):
        return self._reset(scene)


class MultiWalkerWorld:
    def __init__(
        self,
        sim_params: MW_SimParams,
        static_sim_params: MW_StaticSimParams,
        n_walkers: int = 3,
    ):
        self.static_sim_params = static_sim_params
        self.sim_params = sim_params
        self.scene: SimState = None
        self._n_walkers = n_walkers

        self.walkers: list[BipedalWalker] = []
        self.package_index = None

    def _generate_walkers(self):
        self.start_x = jnp.array([5 * i + 3 for i in range(self._n_walkers)])
        for i in range(self._n_walkers):
            walker = BipedalWalker(
                self,
                self.static_sim_params,
                init_x=self.start_x[i],
            )
            self.scene = walker.setup(self.scene)
            self.walkers.append(walker)

    def _generate_terrain(self):
        self.scene, (_, self.terrain_index) = add_rectangle_to_scene(
            self.scene,
            self.static_sim_params,
            position=jnp.array([0, 0]),
            dimensions=jnp.array([200, TERRAIN_HEIGHT]),
            density=1.0,
            restitution=0.0,
            friction=1.0,
            fixated=True,
        )

    def _generate_package(self):
        package_x = jnp.mean(self.start_x)
        # jax.debug.print("package_x={package_x}", package_x=package_x)
        package_y = TERRAIN_HEIGHT + 2.5 * LEG_H
        # jax.debug.print("package_y={package_y}", package_y=package_y)
        package_scale = self._n_walkers / 1.75
        # jax.debug.print("package_scale={package_scale}", package_scale=package_scale)
        self.scene, (_, self.package_index) = add_polygon_to_scene(
            self.scene,
            self.static_sim_params,
            position=jnp.array([package_x, package_y]),
            vertices=jnp.array([(x * package_scale / SCALE, y / SCALE) for x, y in PACKAGE_POLY]),
            n_vertices=len(PACKAGE_POLY),
            density=1,
            restitution=0.0,
            friction=1,
        )

    def reset(self) -> SimState:
        self.scene = create_empty_sim(
            self.static_sim_params,
            add_floor=False,
            add_walls_and_ceiling=False,
        )

        self._generate_walkers()
        self._generate_terrain()
        self._generate_package()
        return self.scene

    def get_state(self):
        return self.scene


def main():
    screen_dim = (1200, 400)

    # 设定基础参数
    n_walkers = 2
    n_package = 1
    static_sim_params = MW_StaticSimParams(
        num_polygons=n_walkers * 5 + n_package + 2,
        num_circles=2,
        num_thrusters=n_walkers * 1,
        num_joints=n_walkers * 4,
    )
    sim_params = MW_SimParams()
    engine = PhysicsEngine(static_sim_params)

    world = MultiWalkerWorld(sim_params, static_sim_params, n_walkers=n_walkers)
    world.reset()

    # Renderer
    renderer = make_render_pixels(static_sim_params, screen_dim)

    # Step scene
    step_fn = jax.jit(engine.step)

    frames = []
    max_step = 35
    step = 0
    while True:
        actions = -jnp.zeros(static_sim_params.num_joints + static_sim_params.num_thrusters)
        world.scene, _ = step_fn(world.scene, sim_params, actions)

        # Render
        pixels = render_bridge(world, renderer, step)
        frames.append(pixels.astype(np.uint8))

        # step
        step += 1
        if step >= max_step:
            imageio.mimsave("test.gif", frames, fps=20)
            return True


if __name__ == "__main__":
    debug = False

    if debug:
        print("JIT disabled")
        with jax.disable_jit():
            main()
    else:
        main()
