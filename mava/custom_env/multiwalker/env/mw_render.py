from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax2d.maths import rmat
from jax2d.sim_state import SimState
from jaxgl.renderer import clear_screen, make_renderer
from jaxgl.shaders import (
    add_mask_to_shader,
    make_fragment_shader_convex_dynamic_ngon_with_edges,
)

# from mava.custom_env.multiwalker.env.mw_base import MW_StaticSimParams
from mava.custom_env.multiwalker.env.mw_constants import MW_COLORS


def make_render_pixels(
    static_sim_params, screen_dim: Tuple[int, int]
) -> Callable[[SimState, int, int], list]:
    ppud = 12
    patch_size_x = 2000
    patch_size_y = 400
    screen_padding = 400
    full_screen_size = (
        screen_dim[0] + 2 * screen_padding,
        screen_dim[1] + 2 * screen_padding,
    )

    cleared_screen = clear_screen(full_screen_size, jnp.ones(3) * 200.0)

    polygon_shader = add_mask_to_shader(make_fragment_shader_convex_dynamic_ngon_with_edges(5))
    quad_renderer = make_renderer(
        full_screen_size, polygon_shader, (patch_size_x, patch_size_y), batched=True
    )

    def calculate_color_table(n: int) -> list:
        num_agents = 3
        num_polygons = num_agents * 5 + 3
        color_table = jax.numpy.zeros((num_polygons, 3), dtype=jax.numpy.float32)

        # hull
        for i in range(num_agents):
            color_table = color_table.at[i * 5].set(MW_COLORS["hull"][0])

        # legs
        for i in range(num_agents):
            color_table = color_table.at[i * 5 + 1].set(MW_COLORS["leg:L"][0])
            color_table = color_table.at[i * 5 + 2].set(MW_COLORS["leg:L"][0])
            color_table = color_table.at[i * 5 + 3].set(MW_COLORS["leg:R"][0])
            color_table = color_table.at[i * 5 + 4].set(MW_COLORS["leg:R"][0])

        return color_table

    @jax.jit
    def render_pixels(state: SimState, num_agents: int, step: int) -> list:
        pixels = cleared_screen
        color_table = calculate_color_table(num_agents)

        def _world_space_to_pixel_spacestep(x):
            return jnp.array(
                [[t[0] * ppud + screen_padding, t[1] * ppud + screen_padding] for t in x]
            )

        def _world_space_to_pixel_space(x):
            return x * ppud + screen_padding

        # Rectangles

        rectangle_rmats = jax.vmap(rmat)(state.polygon.rotation)
        rectangle_rmats = jnp.repeat(
            rectangle_rmats[:, None, :, :],
            repeats=static_sim_params.max_polygon_vertices,
            axis=1,
        )

        # vertices to pixel space
        rectangle_vertices_pixel_space = _world_space_to_pixel_space(
            state.polygon.position[:, None, :]
            + jax.vmap(jax.vmap(jnp.matmul))(rectangle_rmats, state.polygon.vertices)
        )

        # calculate patch positions
        rect_positions_pixel_space = _world_space_to_pixel_spacestep(state.polygon.position)
        rect_patch_positions = (rect_positions_pixel_space - 100 / 2).astype(jnp.int32)
        rect_patch_positions = jnp.maximum(rect_patch_positions, 0)
        # jax.debug.print("state.collision_matrix: {x}", x=state.acc_rr_manifolds.active)
        # jax.debug.print(
        #     "state.collision_matrix:collision_point: {x}",
        #     x=state.acc_rr_manifolds.penetration[0],
        # )
        # jax.debug.print("state.polygon.position[0]: {x}", x=state.polygon.position[0])
        # jax.debug.print("state.polygon.rotation: {x}", x=state.polygon.rotation)
        # jax.debug.print("state.polygon.velocity: {x}", x=state.polygon.velocity)
        # jax.debug.print("rect_positions_pixel_space: {x}", x=rect_positions_pixel_space)
        # jax.debug.print("rect_patch_positions: {x}", x=rect_patch_positions)
        # jax.debug.print("rect_patch_positions max: {x}", x=rect_patch_positions)

        rect_colours = jnp.array(
            [color_table[idx] for idx in range(static_sim_params.num_polygons)]
        )
        rect_uniforms = (
            rectangle_vertices_pixel_space,
            rect_colours,
            rect_colours,
            state.polygon.n_vertices,
            state.polygon.active,
        )

        pixels = quad_renderer(pixels, rect_patch_positions, rect_uniforms)

        # Crop out the sides
        return jnp.rot90(pixels[screen_padding:-screen_padding, screen_padding:-screen_padding])

    return render_pixels


def render_bridge(world, state, renderer, step, num_agents):
    return renderer(state, num_agents, step)
