#!/usr/bin/env python3
"""Spawn scaled COCO in the Isaac Rivermark scene and drive it along roads."""

from __future__ import annotations

import argparse
import copy
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from isaaclab.app import AppLauncher


RIVERMARK_USD_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/5.1/Isaac/Environments/Outdoor/Rivermark/rivermark.usd"
)

WHEEL_RADIUS = 0.3
WHEEL_BASE = 1.5
TRACK_WIDTH = 1.8
MAX_STEER = 0.45
MAX_SPEED = 3.0

DEFAULT_ROAD_KEYWORDS = ("road", "street", "asphalt", "lane", "driveway")
DEFAULT_OBSTACLE_KEYWORDS = (
    "prop",
    "house",
    "building",
    "lamp",
    "light",
    "grass",
    "tree",
    "bush",
    "curb",
    "sidewalk",
    "pole",
    "fence",
)

ROAD_SURFACE_EXCLUDE_KEYWORDS = (
    "building",
    "house",
    "lamp",
    "light",
    "grass",
    "tree",
    "bush",
    "hedge",
    "prop",
    "foliage",
    "pole",
    "fence",
    "table",
    "chair",
)


@dataclass
class Bounds2D:
    path: str
    minimum: np.ndarray
    maximum: np.ndarray

    @property
    def center(self) -> np.ndarray:
        return 0.5 * (self.minimum + self.maximum)

    @property
    def size(self) -> np.ndarray:
        return self.maximum - self.minimum

    @property
    def area_xy(self) -> float:
        size = self.size
        return float(max(0.0, size[0]) * max(0.0, size[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Drive scaled COCO in Isaac Rivermark")
    parser.add_argument("--asset-url", default=RIVERMARK_USD_URL, help="USD file for the Rivermark environment.")
    parser.add_argument("--num-steps", type=int, default=6000, help="Simulation steps to run. Use <=0 for infinite.")
    parser.add_argument("--sim-dt", type=float, default=0.005, help="Physics time-step.")
    parser.add_argument("--render-every", type=int, default=8, help="Render interval in physics steps.")
    parser.add_argument("--control-decimation", type=int, default=20, help="Physics steps per control update.")
    parser.add_argument("--coco-scale", type=float, default=0.55, help="Uniform scale applied to the COCO USD.")
    parser.add_argument("--spawn-z-offset", type=float, default=0.45, help="Robot root z above the selected road.")
    parser.add_argument("--target-speed", type=float, default=1.2, help="Nominal forward speed in m/s.")
    parser.add_argument("--waypoint-radius", type=float, default=2.0, help="Distance for advancing to the next waypoint.")
    parser.add_argument("--lookahead-waypoint", type=int, default=1, help="Waypoint offset used by the pure-pursuit target.")
    parser.add_argument("--avoid-distance", type=float, default=8.0, help="Forward obstacle reaction distance.")
    parser.add_argument("--avoid-width", type=float, default=3.0, help="Lateral obstacle reaction width.")
    parser.add_argument("--avoid-gain", type=float, default=0.55, help="Steering gain for obstacle avoidance.")
    parser.add_argument("--max-obstacle-extent", type=float, default=28.0, help="Ignore huge bbox obstacles above this XY extent.")
    parser.add_argument("--log-every", type=int, default=120, help="Print status every N sim steps.")
    parser.add_argument(
        "--close-app",
        action="store_true",
        help="Call SimulationApp.close() on exit. Disabled by default because large remote stages can hang during Kit shutdown.",
    )
    parser.add_argument(
        "--road-collider",
        default="auto",
        choices=("auto", "off"),
        help="Spawn a hidden high-friction static collider under the selected road surface, like random_env terrain.",
    )
    parser.add_argument("--road-collider-thickness", type=float, default=0.20, help="Thickness of the hidden road collider.")
    parser.add_argument("--road-collider-padding", type=float, default=1.00, help="XY padding added around the selected road bbox.")
    parser.add_argument("--road-static-friction", type=float, default=1.0, help="Static friction for the hidden road collider.")
    parser.add_argument("--road-dynamic-friction", type=float, default=1.0, help="Dynamic friction for the hidden road collider.")
    parser.add_argument(
        "--gui-lite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "In GUI mode, hide the densest Rivermark foliage/grass instancers and force performance rendering "
            "unless --rendering_mode is provided. This keeps the scene usable on 8GB GPUs."
        ),
    )
    parser.add_argument(
        "--gui-lite-hide-paths",
        default="/World/Rivermark/main/foliage,/World/Rivermark/main/grass",
        help="Comma-separated prim paths hidden by --gui-lite after route/obstacle bounds are collected.",
    )
    parser.add_argument(
        "--waypoints",
        default=None,
        help="Optional route as 'x,y;x,y;...'. If omitted, a route is inferred from road-like prim names.",
    )
    parser.add_argument(
        "--road-keywords",
        default=",".join(DEFAULT_ROAD_KEYWORDS),
        help="Comma-separated prim-name hints used to infer road surfaces.",
    )
    parser.add_argument(
        "--obstacle-keywords",
        default=",".join(DEFAULT_OBSTACLE_KEYWORDS),
        help="Comma-separated prim-name hints used for reactive avoidance.",
    )
    parser.add_argument(
        "--collision-mode",
        default="none",
        choices=("matching", "all", "none"),
        help=(
            "Add CollisionAPI to matching meshes, all meshes, or none. "
            "The reactive obstacle avoidance still uses Rivermark bounding boxes when this is none."
        ),
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not getattr(args, "device_explicit", False):
        args.device = "cpu"
    if args.gui_lite and not args.headless and not getattr(args, "rendering_mode_explicit", False):
        args.rendering_mode = "performance"
    return args


def main() -> None:
    args = parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        _run(args)
    except KeyboardInterrupt:
        print("[INFO] Simulation interrupted by user.", file=sys.stderr)
    finally:
        if args.close_app:
            simulation_app.close()
        else:
            print("[INFO] Skipping SimulationApp.close(); process exit will release Isaac Sim resources.", flush=True)


def _run(args: argparse.Namespace) -> None:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.utils.configclass import configclass
    import omni.usd

    from urbansim.primitives.robot.coco import COCO_CFG
    from urbansim.scene.urban_scene import UrbanScene
    from urbansim.scene.urban_scene_cfg import UrbanSceneCfg

    robot_cfg = copy.deepcopy(COCO_CFG)
    robot_cfg.spawn.scale = (args.coco_scale, args.coco_scale, args.coco_scale)
    robot_cfg.init_state.pos = (0.0, 0.0, args.spawn_z_offset)

    @configclass
    class RivermarkSceneCfg(UrbanSceneCfg):
        num_envs: int = 1
        env_spacing: float = 1.0
        replicate_physics: bool = False
        filter_collisions: bool = False
        scenario_generation_method: str = "predefined"

        rivermark = AssetBaseCfg(
            prim_path="/World/Rivermark",
            spawn=sim_utils.UsdFileCfg(usd_path=args.asset_url),
        )
        sky_light = AssetBaseCfg(
            prim_path="/World/RivermarkDomeLight",
            spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(1.0, 1.0, 1.0)),
        )
        robot = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")

    sim_cfg = SimulationCfg(
        dt=args.sim_dt,
        render_interval=max(1, args.render_every),
        device=args.device,
        enable_scene_query_support=True,
    )
    sim = SimulationContext(sim_cfg)
    _ensure_physics_scene(omni.usd.get_context().get_stage(), "/physicsScene")

    scene = UrbanScene(RivermarkSceneCfg())
    scene.generate_scene()

    stage = omni.usd.get_context().get_stage()
    patched_textures = _patch_known_texture_paths(stage)
    if patched_textures:
        print(f"[INFO] Patched {patched_textures} missing texture reference(s) to local asset paths.")

    road_keywords = _split_keywords(args.road_keywords)
    obstacle_keywords = _split_keywords(args.obstacle_keywords)
    if args.collision_mode != "none":
        patched = _patch_collisions(stage, "/World/Rivermark", road_keywords, obstacle_keywords, args.collision_mode)
        print(f"[INFO] CollisionAPI patched on {patched} Rivermark mesh prim(s).")

    road_bounds, obstacle_bounds = _collect_bounds(
        stage,
        "/World/Rivermark",
        road_keywords,
        obstacle_keywords,
        max_obstacle_extent=args.max_obstacle_extent,
    )
    print(f"[INFO] Found {len(road_bounds)} road-like bounds and {len(obstacle_bounds)} obstacle-like bounds.")

    route, spawn_z, route_source, route_surface = _choose_route(args, road_bounds)
    print(f"[INFO] Route source: {route_source}")
    print(f"[INFO] Route has {len(route)} waypoint(s). spawn_z={spawn_z:.2f}")
    if args.road_collider == "auto" and route_surface is not None:
        road_collider_path = _spawn_route_collider(stage, route_surface, args)
        print(
            f"[INFO] Spawned hidden road traction collider at {road_collider_path} "
            f"top_z={route_surface.maximum[2]:.2f} size=({route_surface.size[0]:.2f}, {route_surface.size[1]:.2f})."
        )
    hidden_paths = _apply_gui_lite_mode(stage, args)
    if hidden_paths:
        print("[INFO] GUI lite mode hid heavy visual prims: " + ", ".join(hidden_paths))

    sim.reset()
    scene.reset()
    scene.write_data_to_sim()
    scene.update(0.0)

    robot = scene["robot"]
    wheel_joint_ids, wheel_joint_names = robot.find_joints(".*wheel_joint")
    axle_joint_ids, axle_joint_names = robot.find_joints("base_to_front_axle_joint")
    if len(wheel_joint_ids) == 0 or len(axle_joint_ids) == 0:
        raise RuntimeError(f"COCO joints not found. wheel={wheel_joint_names}, axle={axle_joint_names}")

    left_wheel_ids = _filter_joint_ids(wheel_joint_ids, wheel_joint_names, "left")
    right_wheel_ids = _filter_joint_ids(wheel_joint_ids, wheel_joint_names, "right")
    if not left_wheel_ids or not right_wheel_ids:
        raise RuntimeError(f"Unable to split wheel joints into left/right groups: {wheel_joint_names}")

    _teleport_robot_to_route_start(robot, route, spawn_z, scene.device)

    wheel_radius = WHEEL_RADIUS * args.coco_scale
    wheel_base = WHEEL_BASE * args.coco_scale
    track_width = TRACK_WIDTH * args.coco_scale
    max_wheel_rate = MAX_SPEED / max(1e-6, wheel_radius)

    dt = sim.get_physics_dt()
    sim_step = 0
    waypoint_index = 0
    speed_cmd = 0.0
    steer_cmd = 0.0

    while args.num_steps <= 0 or sim_step < args.num_steps:
        state = _get_robot_state(robot, axle_joint_ids)
        waypoint_index = _advance_waypoint(route, state["position"], waypoint_index, args.waypoint_radius)
        target_index = min(len(route) - 1, waypoint_index + max(0, args.lookahead_waypoint))
        target = route[target_index]

        steer_cmd, speed_cmd = _compute_drive_command(state, target, obstacle_bounds, args)
        left_speed, right_speed = _ackermann_wheel_speeds(speed_cmd, steer_cmd, wheel_base, track_width, wheel_radius)
        _set_wheel_velocity(robot, left_wheel_ids, left_speed, max_wheel_rate)
        _set_wheel_velocity(robot, right_wheel_ids, right_speed, max_wheel_rate)
        _set_steering(robot, axle_joint_ids, steer_cmd)

        for _ in range(max(1, args.control_decimation)):
            scene.write_data_to_sim()
            should_render = bool(sim.has_gui() and sim_step % max(1, args.render_every) == 0)
            sim.step(render=should_render)
            scene.update(dt)
            sim_step += 1
            if args.num_steps > 0 and sim_step >= args.num_steps:
                break

        if args.log_every > 0 and sim_step % args.log_every == 0:
            state = _get_robot_state(robot, axle_joint_ids, wheel_joint_ids)
            print(
                f"step={sim_step:06d} waypoint={waypoint_index:02d}/{len(route)-1:02d} "
                f"speed={speed_cmd:4.2f} steer={math.degrees(steer_cmd):6.2f}deg "
                f"pos=({state['position'][0]:7.2f},{state['position'][1]:7.2f}) "
                f"z={state['height']:5.2f} vxy={state['speed']:4.2f} wheel_actual={state['wheel_speed']:5.2f}",
                flush=True,
            )

    print("[INFO] Completed requested Rivermark simulation steps.", flush=True)


def _split_keywords(value: str) -> tuple[str, ...]:
    return tuple(keyword.strip().lower() for keyword in value.split(",") if keyword.strip())


def _split_paths(value: str) -> tuple[str, ...]:
    return tuple(path.strip() for path in value.split(",") if path.strip())


def _patch_known_texture_paths(stage) -> int:
    from pxr import Sdf

    repo_root = Path(__file__).resolve().parents[3]
    texture_paths = {
        "coco-white.png": repo_root / "assets" / "robots" / "coco_one" / "materials" / "coco-white.png",
        "coco-white.png.001.png": (
            repo_root / "assets" / "robots" / "coco_one" / "materials" / "coco-white.png.001.png"
        ),
        "wood_planks_01_basecolor.png": (
            repo_root / "assets" / "materials" / "Wood" / ".thumbs" / "Wood_Tiles_Pine.Wood_Tiles_Pine.png"
        ),
    }

    patched = 0
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            try:
                value = attr.Get()
            except Exception:
                continue
            asset_path = value.path if isinstance(value, Sdf.AssetPath) else value if isinstance(value, str) else None
            if asset_path is None:
                continue
            basename = Path(asset_path.strip("@")).name
            local_path = texture_paths.get(basename)
            if local_path is None or not local_path.exists():
                continue
            if asset_path == str(local_path):
                continue
            if isinstance(value, Sdf.AssetPath):
                attr.Set(Sdf.AssetPath(str(local_path)))
            else:
                attr.Set(str(local_path))
            patched += 1
    return patched


def _apply_gui_lite_mode(stage, args: argparse.Namespace) -> list[str]:
    if args.headless or not args.gui_lite:
        return []

    from pxr import UsdGeom

    hidden_paths: list[str] = []
    for prim_path in _split_paths(args.gui_lite_hide_paths):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            continue
        UsdGeom.Imageable(prim).MakeInvisible()
        hidden_paths.append(prim_path)
    return hidden_paths


def _ensure_physics_scene(stage, prim_path: str) -> None:
    from pxr import Gf, PhysxSchema, Sdf, UsdPhysics

    scene_prim = stage.GetPrimAtPath(Sdf.Path(prim_path))
    if not scene_prim or not scene_prim.IsValid():
        physics_scene = UsdPhysics.Scene.Define(stage, prim_path)
        physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
        physics_scene.CreateGravityMagnitudeAttr(9.81)
        scene_prim = physics_scene.GetPrim()
    if not scene_prim.HasAPI(PhysxSchema.PhysxSceneAPI):
        PhysxSchema.PhysxSceneAPI.Apply(scene_prim)


def _patch_collisions(stage, root_path: str, road_keywords: tuple[str, ...], obstacle_keywords: tuple[str, ...], mode: str) -> int:
    from pxr import UsdGeom, UsdPhysics

    patched = 0
    keywords = road_keywords + obstacle_keywords
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if not path.startswith(root_path) or not prim.IsA(UsdGeom.Mesh):
            continue
        path_l = path.lower()
        if mode == "matching" and not any(keyword in path_l for keyword in keywords):
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(prim)
            patched += 1
    return patched


def _collect_bounds(
    stage,
    root_path: str,
    road_keywords: tuple[str, ...],
    obstacle_keywords: tuple[str, ...],
    max_obstacle_extent: float,
) -> tuple[list[Bounds2D], list[Bounds2D]]:
    from pxr import Usd, UsdGeom

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    road_bounds: list[Bounds2D] = []
    obstacle_bounds: list[Bounds2D] = []

    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if not path.startswith(root_path) or not prim.IsA(UsdGeom.Boundable):
            continue
        path_l = path.lower()
        if not any(keyword in path_l for keyword in road_keywords + obstacle_keywords):
            continue

        bounds = _world_bounds_2d(bbox_cache, prim)
        if bounds is None or bounds.area_xy <= 0.02:
            continue

        is_road = any(keyword in path_l for keyword in road_keywords)
        is_obstacle = any(keyword in path_l for keyword in obstacle_keywords)
        if is_road:
            road_bounds.append(bounds)
        if is_obstacle and float(max(bounds.size[0], bounds.size[1])) <= max_obstacle_extent:
            obstacle_bounds.append(bounds)

    road_bounds.sort(key=lambda item: item.area_xy, reverse=True)
    obstacle_bounds.sort(key=lambda item: item.area_xy, reverse=True)
    return road_bounds, obstacle_bounds


def _world_bounds_2d(bbox_cache, prim) -> Bounds2D | None:
    try:
        aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        minimum_gf = aligned.GetMin()
        maximum_gf = aligned.GetMax()
    except Exception:
        return None

    minimum = np.array([minimum_gf[0], minimum_gf[1], minimum_gf[2]], dtype=np.float32)
    maximum = np.array([maximum_gf[0], maximum_gf[1], maximum_gf[2]], dtype=np.float32)
    if not np.isfinite(minimum).all() or not np.isfinite(maximum).all():
        return None
    if np.any(maximum <= minimum):
        return None
    return Bounds2D(prim.GetPath().pathString, minimum, maximum)


def _choose_route(args: argparse.Namespace, road_bounds: list[Bounds2D]) -> tuple[np.ndarray, float, str, Bounds2D | None]:
    if args.waypoints:
        route = _parse_waypoints(args.waypoints)
        return route, args.spawn_z_offset, "manual --waypoints", None

    if road_bounds:
        road = _select_route_surface(road_bounds)
        center = road.center
        size = road.size
        axis = 0 if size[0] >= size[1] else 1
        start = road.minimum[axis] + min(8.0, max(1.0, 0.1 * size[axis]))
        end = road.maximum[axis] - min(8.0, max(1.0, 0.1 * size[axis]))
        if end < start:
            start, end = road.minimum[axis], road.maximum[axis]
        samples = np.linspace(start, end, 10, dtype=np.float32)
        route = np.zeros((len(samples), 2), dtype=np.float32)
        route[:, axis] = samples
        route[:, 1 - axis] = center[1 - axis]
        spawn_z = float(road.maximum[2] + args.spawn_z_offset)
        return route, spawn_z, road.path, road

    route = np.array([[-30.0, 0.0], [-15.0, 0.0], [0.0, 0.0], [15.0, 0.0], [30.0, 0.0]], dtype=np.float32)
    return route, args.spawn_z_offset, "fallback straight route", None


def _spawn_route_collider(stage, bounds: Bounds2D, args: argparse.Namespace) -> str:
    import isaaclab.sim as sim_utils
    from pxr import UsdGeom

    prim_path = "/World/RivermarkRoadTractionCollider"
    thickness = max(0.02, float(args.road_collider_thickness))
    padding = max(0.0, float(args.road_collider_padding))
    size = bounds.size
    center = bounds.center.copy()
    center[2] = float(bounds.maximum[2] - 0.5 * thickness)

    collider_cfg = sim_utils.CuboidCfg(
        size=(float(size[0] + 2.0 * padding), float(size[1] + 2.0 * padding), thickness),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=float(args.road_static_friction),
            dynamic_friction=float(args.road_dynamic_friction),
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.2, 0.8), opacity=0.0),
    )
    collider_cfg.func(prim_path, collider_cfg, translation=tuple(float(v) for v in center))

    prim = stage.GetPrimAtPath(prim_path)
    if prim and prim.IsValid():
        UsdGeom.Imageable(prim).MakeInvisible()
    return prim_path


def _select_route_surface(road_bounds: list[Bounds2D]) -> Bounds2D:
    candidates = [bounds for bounds in road_bounds if _is_route_surface_candidate(bounds)]
    if candidates:
        return max(candidates, key=lambda bounds: (bounds.area_xy, -float(bounds.size[2])))
    return road_bounds[0]


def _is_route_surface_candidate(bounds: Bounds2D) -> bool:
    path_l = bounds.path.lower()
    if any(keyword in path_l for keyword in ROAD_SURFACE_EXCLUDE_KEYWORDS):
        return False
    size = bounds.size
    if bounds.area_xy < 25.0:
        return False
    if max(float(size[0]), float(size[1])) < 8.0 or min(float(size[0]), float(size[1])) < 1.5:
        return False
    return float(size[2]) <= 2.0


def _parse_waypoints(value: str) -> np.ndarray:
    points = []
    for item in value.split(";"):
        if not item.strip():
            continue
        x_str, y_str = item.split(",")
        points.append([float(x_str), float(y_str)])
    if len(points) < 2:
        raise ValueError("--waypoints must contain at least two 'x,y' pairs")
    return np.asarray(points, dtype=np.float32)


def _teleport_robot_to_route_start(robot, route: np.ndarray, spawn_z: float, device: str) -> None:
    yaw = math.atan2(route[1, 1] - route[0, 1], route[1, 0] - route[0, 0])
    quat = _quat_from_yaw(yaw)
    root_pose = torch.tensor([[route[0, 0], route[0, 1], spawn_z, *quat]], dtype=torch.float32, device=device)
    root_velocity = torch.zeros((1, 6), dtype=torch.float32, device=device)
    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(root_velocity)


def _get_robot_state(robot, axle_joint_ids: Sequence[int], wheel_joint_ids: Sequence[int] | None = None) -> dict:
    root_pos = robot.data.root_pos_w[0].cpu().numpy()
    root_vel = robot.data.root_vel_w[0].cpu().numpy()
    root_quat = robot.data.root_quat_w[0].cpu().numpy()
    yaw = _quat_to_yaw(root_quat)
    speed = float(np.linalg.norm(root_vel[:2]))
    steering = float(robot.data.joint_pos[0, axle_joint_ids].mean().item())
    wheel_speed = float("nan")
    if wheel_joint_ids is not None and len(wheel_joint_ids) > 0:
        wheel_speed = float(robot.data.joint_vel[0, wheel_joint_ids].abs().mean().item())
    return {
        "position": root_pos[:2].copy(),
        "height": float(root_pos[2]),
        "yaw": yaw,
        "speed": speed,
        "steering": steering,
        "wheel_speed": wheel_speed,
    }


def _compute_drive_command(state: dict, target: np.ndarray, obstacle_bounds: list[Bounds2D], args: argparse.Namespace) -> tuple[float, float]:
    dx = float(target[0] - state["position"][0])
    dy = float(target[1] - state["position"][1])
    local_x, local_y = _world_to_local_xy(dx, dy, state["yaw"])
    heading_error = math.atan2(local_y, max(0.3, local_x))
    steer = 0.75 * heading_error

    avoid_steer, brake_factor = _obstacle_avoidance(state, obstacle_bounds, args)
    steer = max(-MAX_STEER, min(MAX_STEER, steer + avoid_steer))

    steer_slowdown = 1.0 - 0.45 * min(1.0, abs(steer) / MAX_STEER)
    speed = args.target_speed * steer_slowdown * brake_factor
    speed = max(0.15, min(args.target_speed, speed))
    return steer, speed


def _obstacle_avoidance(state: dict, obstacle_bounds: list[Bounds2D], args: argparse.Namespace) -> tuple[float, float]:
    steer = 0.0
    closest_front = args.avoid_distance
    position = state["position"]

    for bounds in obstacle_bounds:
        closest = np.clip(position, bounds.minimum[:2], bounds.maximum[:2])
        dx = float(closest[0] - position[0])
        dy = float(closest[1] - position[1])
        local_x, local_y = _world_to_local_xy(dx, dy, state["yaw"])
        if local_x < -0.5 or local_x > args.avoid_distance:
            continue
        lateral_margin = args.avoid_width + 0.5 * min(bounds.size[0], bounds.size[1])
        if abs(local_y) > lateral_margin:
            continue
        side = -1.0 if local_y >= 0.0 else 1.0
        distance_weight = 1.0 - max(0.0, local_x) / max(1e-6, args.avoid_distance)
        lateral_weight = 1.0 - min(1.0, abs(local_y) / max(1e-6, lateral_margin))
        steer += side * args.avoid_gain * distance_weight * lateral_weight
        closest_front = min(closest_front, max(0.0, local_x))

    brake_factor = 0.35 + 0.65 * min(1.0, closest_front / max(1e-6, args.avoid_distance))
    return max(-MAX_STEER, min(MAX_STEER, steer)), brake_factor


def _world_to_local_xy(dx: float, dy: float, yaw: float) -> tuple[float, float]:
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    local_x = cos_yaw * dx + sin_yaw * dy
    local_y = -sin_yaw * dx + cos_yaw * dy
    return local_x, local_y


def _advance_waypoint(route: np.ndarray, position: np.ndarray, current_index: int, radius: float) -> int:
    index = current_index
    while index < len(route) - 1 and np.linalg.norm(route[index] - position) < radius:
        index += 1
    return index


def _filter_joint_ids(joint_ids: Sequence[int], joint_names: Sequence[str], keyword: str) -> list[int]:
    return [jid for jid, name in zip(joint_ids, joint_names) if keyword in name]


def _set_wheel_velocity(robot, joint_ids: Sequence[int], wheel_speed: float, max_wheel_rate: float) -> None:
    clipped_speed = float(max(-max_wheel_rate, min(max_wheel_rate, wheel_speed)))
    device = robot.data.root_pos_w.device
    targets = torch.full((robot.num_instances, len(joint_ids)), clipped_speed, device=device)
    robot.set_joint_velocity_target(targets, joint_ids=joint_ids)


def _set_steering(robot, joint_ids: Sequence[int], steering_angle: float) -> None:
    clipped_angle = float(max(-MAX_STEER, min(MAX_STEER, steering_angle)))
    device = robot.data.root_pos_w.device
    targets = torch.full((robot.num_instances, len(joint_ids)), clipped_angle, device=device)
    robot.set_joint_position_target(targets, joint_ids=joint_ids)


def _ackermann_wheel_speeds(
    forward_speed: float,
    steering_angle: float,
    wheel_base: float,
    track_width: float,
    wheel_radius: float,
) -> tuple[float, float]:
    if abs(steering_angle) < 1e-5:
        angular_velocity = forward_speed / wheel_radius
        return angular_velocity, angular_velocity

    turning_radius = wheel_base / math.tan(steering_angle)
    inner_radius = turning_radius - 0.5 * track_width
    outer_radius = turning_radius + 0.5 * track_width
    if abs(inner_radius) < 1e-5 or abs(outer_radius) < 1e-5:
        angular_velocity = forward_speed / wheel_radius
        return angular_velocity, angular_velocity

    inner_speed = forward_speed * (inner_radius / turning_radius)
    outer_speed = forward_speed * (outer_radius / turning_radius)
    inner_angular = inner_speed / wheel_radius
    outer_angular = outer_speed / wheel_radius

    if steering_angle > 0:
        return inner_angular, outer_angular
    return outer_angular, inner_angular


def _quat_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    half = 0.5 * yaw
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def _quat_to_yaw(quat: np.ndarray) -> float:
    w, x, y, z = quat
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


if __name__ == "__main__":
    main()
