import os
from functools import partial
from typing import Dict, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax.struct import dataclass
from jax2d.engine import PhysicsEngine, RigidBody
from jax2d.sim_state import SimState
from jaxmarl.environments import spaces
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.wrappers.baselines import LogWrapper

from mava.custom_env.multiwalker.env.mw_base import (
    MultiWalkerWorld,
    MW_SimParams,
    MW_StaticSimParams,
)

# from mava.custom_env.multiwalker.env import (
#     PACKAGE_LENGTH,
#     SCALE,
#     MultiWalkerWorld,
#     MW_SimParams,
#     MW_StaticSimParams,
#     _extract_joint,
#     _extract_polygon,
#     _is_ground_contact,
#     make_render_pixels,
#     render_bridge,
# )
from mava.custom_env.multiwalker.env.mw_constants import (
    PACKAGE_LENGTH,
    SCALE,
)
from mava.custom_env.multiwalker.env.mw_render import (
    make_render_pixels,
    render_bridge,
)
from mava.custom_env.multiwalker.env.mw_utils import (
    _extract_joint,
    _extract_polygon,
    _is_ground_contact,
)

os.environ["SDL_VIDEODRIVER"] = "dummy"


@dataclass
class StateWithStep:
    state: SimState
    step: int


class MultiWalkerEnv(MultiAgentEnv):
    def __init__(
        self,
        n_walkers: int = 3,
        position_noise: float = 1e-3,
        angle_noise: float = 1e-3,
    ):
        self.n_walkers = n_walkers
        self.position_noise = position_noise
        self.angle_noise = angle_noise
        self.screen_dim = (1200, 400)

        self.terrain_index = self.n_walkers * 5
        self.package_index = self.n_walkers * 5 + 1

        self.key = jax.random.PRNGKey(42)
        self.package_scale = self.n_walkers / 1.75
        self.package_length = PACKAGE_LENGTH / SCALE * self.package_scale

        # jax2d参数
        self.static_sim_params = MW_StaticSimParams(
            num_polygons=n_walkers * 5 + 1 + 3,
            num_circles=2,
            num_thrusters=2,
            num_joints=n_walkers * 4,
        )
        self.sim_params = MW_SimParams()

        # jax2d引擎
        self.engine = PhysicsEngine(self.static_sim_params)
        self.step_fn = jax.jit(self.engine.step)

        # 环境
        self.env = MultiWalkerWorld(self.sim_params, self.static_sim_params, n_walkers=n_walkers)

        # agents
        self.num_agents = n_walkers
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]

        # 观察空间和动作空间
        self.observation_spaces = {
            agent: spaces.Box(
                -jnp.inf,
                jnp.inf,
                shape=(21,),
            )
            for agent in self.agents
        }
        self.action_spaces = {
            agent: spaces.Box(
                -1.0,
                1.0,
                shape=(4,),
            )
            for agent in self.agents
        }

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], StateWithStep]:
        state = self.env.reset()
        state_with_step = StateWithStep(state=state, step=0)
        return self.get_obs(state_with_step), state_with_step

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        key: chex.PRNGKey,
        state: StateWithStep,
        actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], StateWithStep, Dict[str, float], Dict[str, bool], Dict]:
        step = state.step
        state = state.state

        # 环境步进
        actions_as_array = jnp.array([actions[agent] for agent in self.agents]).flatten()
        actions_as_array = jnp.concatenate(
            [actions_as_array, jnp.zeros(self.static_sim_params.num_thrusters)]
        )
        next_state, _ = self.step_fn(state, self.sim_params, actions_as_array)
        next_state: SimState = next_state
        # 获取观测
        next_state_with_step = StateWithStep(state=next_state, step=step + 1)
        observations = self.get_obs(next_state_with_step)

        def _calculate_reward(state: SimState, next_state: SimState):
            """
            - forward * 1
            - package contact ground: -100[done]
            - walker contact ground: -10[done]
            - hull angle change: -5*rotation
            """
            done = False
            overall_reward = 0.0

            # forward
            forward_reward = (
                next_state.polygon.position[self.package_index][0]
                - state.polygon.position[self.package_index][0]
            ) * 40.0

            overall_reward += forward_reward

            agents_reward = jnp.zeros(self.n_walkers)
            for i in range(self.n_walkers):
                agent_reward = 0.0
                # walker contact ground
                walker_contact_ground = _is_ground_contact(next_state, self.terrain_index, i * 5)
                walker_contact_ground_reward = -10.0 * walker_contact_ground
                agent_reward += walker_contact_ground_reward
                done = done | walker_contact_ground

                # hull angle change
                hull_angle_change_reward = -5.0 * jnp.abs(
                    next_state.polygon.rotation[i * 5]
                ) + 5.0 * jnp.abs(state.polygon.rotation[i * 5])
                agent_reward += hull_angle_change_reward
                agents_reward = agents_reward.at[i].set(agent_reward)

            # package contact ground
            package_contact_ground = _is_ground_contact(
                next_state, self.terrain_index, self.package_index
            )
            done = done | package_contact_ground
            overall_reward += -100.0 * done
            overall_reward += jnp.mean(agents_reward)
            return overall_reward, done

        # 计算奖励
        rwd, done = _calculate_reward(state, next_state)
        rewards = {agent: rwd for agent in self.agents}
        rewards["__all__"] = rwd * 3

        # 计算终止
        done = done | jnp.where(step > 500, True, False)
        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done

        info = {}
        return (
            observations,
            next_state_with_step,  # type: ignore
            rewards,
            dones,
            info,
        )

    def world_state_size(self) -> int:
        spaces = [self.observation_spaces[agent] for agent in self.agents]
        return sum([space.shape[-1] for space in spaces]) + 3

    @partial(jax.jit, static_argnums=0)
    def get_world_obs(self, env_state: StateWithStep, obs: Dict[str, chex.Array]):
        env_state = env_state.state
        package_index = self._env.num_agents * 5 + 1
        package_obs = jnp.array(
            [
                env_state.polygon.position[package_index][0],
                env_state.polygon.position[package_index][1],
                env_state.polygon.rotation[package_index],
            ]
        )
        all_obs = jnp.concatenate(
            [
                jnp.array([obs[agent] for agent in self._env.agents]).flatten(),
                package_obs,
            ]
        ).flatten()
        all_obs = jnp.expand_dims(all_obs, axis=0).repeat(self._env.num_agents, axis=0)
        return all_obs

    def get_obs(self, state: StateWithStep) -> Dict[str, chex.Array]:
        step = state.step
        state = state.state
        # 通过取出self.env里存放的walker里的对应index，从state里取出观测
        package_index = self.n_walkers * 5 + 1
        terrain_index = self.n_walkers * 5

        def _get_obs(walker_idx: int) -> chex.Array:
            # hull
            hull_index = walker_idx * 5
            hull_state: RigidBody = _extract_polygon(state, hull_index)

            # joints
            joint_states = [
                _extract_joint(state, index) for index in range(walker_idx * 4, walker_idx * 4 + 4)
            ]
            # legs
            legs_on_floor = jnp.array(
                [
                    _is_ground_contact(state, terrain_index, index)
                    for index in range(walker_idx * 4, walker_idx * 4 + 4)
                ]
            )

            def _get_ang_vel(pid):
                return _extract_polygon(state, pid).angular_velocity

            def get_motor_speed(jid):
                a_index = joint_states[jid].a_index
                b_index = joint_states[jid].b_index
                return _get_ang_vel(b_index) - _get_ang_vel(a_index)

            full_obs = jnp.array(
                [
                    hull_state.rotation,
                    hull_state.angular_velocity,
                    hull_state.velocity[0],
                    hull_state.velocity[1],
                    joint_states[0].rotation,
                    get_motor_speed(0),
                    joint_states[1].rotation,
                    get_motor_speed(1),
                    legs_on_floor[1],
                    joint_states[2].rotation,
                    get_motor_speed(2),
                    joint_states[3].rotation,
                    get_motor_speed(3),
                    legs_on_floor[3],
                    # above are the first 14 observations
                    # however, LiDAR hasn't been added yet, so jumped 10 observations
                ]
            )
            return full_obs

        agent_obs = {
            agent: _get_obs(i) for agent, i in zip(self.agents, range(self.n_walkers), strict=False)
        }

        # 获取邻居观测
        for i in range(self.n_walkers):
            neighbor_obs = []
            x = _extract_polygon(state, i * 5).position[0]
            y = _extract_polygon(state, i * 5).position[1]
            for j in [i - 1, i + 1]:
                # if no neighbor (for edge walkers)
                if j < 0 or j == self.n_walkers:
                    neighbor_obs.append(0.0)
                    neighbor_obs.append(0.0)
                else:
                    xm = _extract_polygon(state, j * 5).position[0] - x
                    ym = _extract_polygon(state, j * 5).position[1] - y
                    xm = xm / self.package_length
                    ym = ym / self.package_length
                    neighbor_obs.append(xm + jax.random.normal(self.key, ()) * self.position_noise)
                    neighbor_obs.append(ym + jax.random.normal(self.key, ()) * self.position_noise)
            # 28 29, 与包裹的相对位置
            package_state = _extract_polygon(state, package_index)
            xd = package_state.position[0] - x
            yd = package_state.position[1] - y
            xd = xd / self.package_length
            yd = yd / self.package_length
            neighbor_obs.append(xd + jax.random.normal(self.key, ()) * self.position_noise)
            neighbor_obs.append(yd + jax.random.normal(self.key, ()) * self.position_noise)
            neighbor_obs.append(
                package_state.rotation + jax.random.normal(self.key, ()) * self.angle_noise
            )
            agent_obs[f"agent_{i}"] = jnp.concatenate(
                [agent_obs[f"agent_{i}"], jnp.array(neighbor_obs)]
            )

        return agent_obs

    def render(self, state: StateWithStep, step: int):
        self.renderer = make_render_pixels(self.static_sim_params, self.screen_dim)
        pixels = render_bridge(self.env, state.state, self.renderer, step, self.num_agents).astype(
            np.uint8
        )
        return pixels


class MultiWalkerLogWrapper(LogWrapper):
    def __init__(self, env: MultiWalkerEnv, replace_info: bool = True):
        super().__init__(env, replace_info)


def main():
    key = jax.random.PRNGKey(0)
    key, key_reset, key_act, key_step = jax.random.split(key, 4)

    # Initialise environment.
    env = MultiWalkerEnv(n_walkers=3)

    # Reset the environment.
    obs, state = env.reset(key_reset)
    jax.debug.print("{s}", s=obs)

    # Sample random actions.
    key_act = jax.random.split(key_act, env.num_agents)
    actions = {
        agent: env.action_space(agent).sample(key_act[i]) for i, agent in enumerate(env.agents)
    }

    # Perform the step transition.
    obs, state, reward, done, infos = env.step(key_step, state, actions)


if __name__ == "__main__":
    main()
