#!/usr/bin/env python3
"""Spawn the COCO robot in an empty UrbanSim scene for controller debugging."""

import argparse
import math
import sys
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from isaaclab.app import AppLauncher


MPC_HORIZON = 20
MPC_DT = 0.2
WHEEL_RADIUS = 0.3
WHEEL_BASE = 1.5
TRACK_WIDTH = 1.8
MAX_SPEED = 3.0
MAX_STEER = 0.45
MAX_WHEEL_RATE = MAX_SPEED / WHEEL_RADIUS


class MPCTrajectoryFollower:
    def __init__(self, horizon: int, dt: float, device: str, iterations: int):
        self.horizon = horizon
        self.dt = dt
        self.device = device
        self.max_speed = MAX_SPEED
        self.max_steer = MAX_STEER
        self.iterations = max(1, iterations)
        self.prev_controls = torch.zeros(self.horizon, 2, device=self.device)

    def solve(self, state: dict, target_points_world: np.ndarray) -> tuple[float, float, np.ndarray]:
        target = torch.as_tensor(target_points_world, dtype=torch.float32, device=self.device)
        controls = self.prev_controls.clone().detach()
        controls[:, 0].clamp_(0.0, self.max_speed)
        controls[:, 1].clamp_(-self.max_steer, self.max_steer)
        controls.requires_grad_(True)

        optimizer = torch.optim.Adam([controls], lr=0.35)

        rollout_state = _state_to_tensors(state, self.device)

        for _ in range(self.iterations):
            optimizer.zero_grad()
            positions = self._rollout(rollout_state, controls)
            if torch.isnan(positions).any():
                break

            position_error = positions - target
            cost = (position_error.pow(2).sum(dim=1)).mean()
            smoothness = (controls[1:] - controls[:-1]).pow(2).sum(dim=1).mean()
            steering_reg = controls[:, 1].pow(2).mean()
            speed_reg = (controls[:, 0] - self.max_speed * 0.5).pow(2).mean()
            total_cost = cost + 0.15 * smoothness + 0.05 * steering_reg + 0.02 * speed_reg
            total_cost.backward()
            optimizer.step()

            with torch.no_grad():
                controls[:, 0].clamp_(0.0, self.max_speed)
                controls[:, 1].clamp_(-self.max_steer, self.max_steer)

        with torch.no_grad():
            predicted_positions = self._rollout(rollout_state, controls).detach()
        self.prev_controls = controls.detach()

        speed_command = float(controls[0, 0].item())
        steering_command = float(controls[0, 1].item())
        predicted_np = predicted_positions.cpu().numpy()
        return speed_command, steering_command, predicted_np

    def _rollout(self, rollout_state: dict, controls: torch.Tensor) -> torch.Tensor:
        x = rollout_state["x"]
        y = rollout_state["y"]
        yaw = rollout_state["yaw"]
        speed = rollout_state["speed"]
        steer = rollout_state["steer"]

        positions = []
        for u in controls:
            target_speed = u[0]
            target_steer = u[1]
            speed = speed + 0.4 * (target_speed - speed)
            steer = steer + 0.3 * (target_steer - steer)
            yaw = yaw + self.dt * speed * torch.tan(steer) / WHEEL_BASE
            x = x + self.dt * speed * torch.cos(yaw)
            y = y + self.dt * speed * torch.sin(yaw)
            positions.append(torch.stack([x, y]))

        return torch.stack(positions)


class TrajectoryVisualizer:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_title("MPC Trajectory Tracking")

    def update(
        self,
        position_history: list[np.ndarray],
        target_points: np.ndarray,
        predicted_points: np.ndarray,
        robot_state: dict,
    ) -> None:
        self.ax.clear()
        self.ax.set_title("MPC Trajectory Tracking")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.grid(True, alpha=0.3)

        if position_history:
            history_array = np.stack(position_history)
            self.ax.plot(history_array[:, 0], history_array[:, 1], label="Actual path", color="tab:blue")

        if target_points is not None and len(target_points) > 0:
            self.ax.plot(
                target_points[:, 0],
                target_points[:, 1],
                label="Target waypoints",
                color="tab:green",
                linestyle="--",
                marker="o",
                markersize=3,
            )

        if predicted_points is not None and len(predicted_points) > 0:
            self.ax.plot(
                predicted_points[:, 0],
                predicted_points[:, 1],
                label="MPC prediction",
                color="tab:orange",
                linestyle="-.",
            )

        position = robot_state["position"]
        self.ax.scatter(position[0], position[1], color="tab:red", marker="x", label="Robot")

        x_center = position[0]
        y_center = position[1]
        self.ax.set_xlim(x_center - 10.0, x_center + 20.0)
        self.ax.set_ylim(y_center - 10.0, y_center + 10.0)
        self.ax.legend(loc="upper right")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)


def _state_to_tensors(state: dict, device: str) -> dict:
    return {
        "x": torch.tensor(state["position"][0], dtype=torch.float32, device=device),
        "y": torch.tensor(state["position"][1], dtype=torch.float32, device=device),
        "yaw": torch.tensor(state["yaw"], dtype=torch.float32, device=device),
        "speed": torch.tensor(state["speed"], dtype=torch.float32, device=device),
        "steer": torch.tensor(state["steering"], dtype=torch.float32, device=device),
    }


def _get_robot_state(robot, axle_joint_ids: Sequence[int]) -> dict:
    root_pos = robot.data.root_pos_w[0].cpu().numpy()
    root_vel = robot.data.root_vel_w[0].cpu().numpy()
    root_quat = robot.data.root_quat_w[0].cpu().numpy()
    yaw = _quat_to_yaw(root_quat)
    speed = float(np.linalg.norm(root_vel[:2]))
    steering = float(robot.data.joint_pos[0, axle_joint_ids].mean().item())
    return {
        "position": root_pos[:2].copy(),
        "yaw": yaw,
        "speed": speed,
        "steering": steering,
    }


def _quat_to_yaw(quat: np.ndarray) -> float:
    w, x, y, z = quat
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _sample_ego_waypoints(
    step: int,
    horizon: int,
    dt: float,
    base_speed: float,
    lateral_amplitude: float,
    lateral_frequency: float,
) -> np.ndarray:
    t = (np.arange(1, horizon + 1) * dt).astype(np.float32)
    forward = np.clip(base_speed * t + 0.5, 0.3, None)
    oscillation = lateral_amplitude * np.sin(lateral_frequency * (t + step * dt))
    curvature = 0.3 * np.sin(0.2 * step * dt)
    lateral = oscillation + curvature * t
    return np.stack([forward, lateral], axis=1)


def _ego_to_world(ego_points: np.ndarray, state: dict) -> np.ndarray:
    cos_yaw = math.cos(state["yaw"])
    sin_yaw = math.sin(state["yaw"])
    rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float32)
    world_points = ego_points @ rotation.T
    world_points += state["position"]
    return world_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("COCO robot empty-scene debugger")
    parser.add_argument("--num-steps", type=int, default=2000, help="How many simulation steps to run")
    parser.add_argument("--sim-dt", type=float, default=0.005, help="Physics time-step in seconds")
    parser.add_argument(
        "--render-every",
        type=int,
        default=1,
        help="Render every N physics steps (only used when a viewer is available)",
    )
    parser.add_argument("--forward-speed", type=float, default=1.0, help="Target forward speed in m/s")
    parser.add_argument(
        "--steer-amplitude",
        type=float,
        default=0.2,
        help="Maximum steering command in radians for the sample controller",
    )
    parser.add_argument(
        "--steer-frequency",
        type=float,
        default=0.2,
        help="Steering oscillation frequency in Hz for the sample controller",
    )
    parser.add_argument("--log-every", type=int, default=120, help="Print robot pose every N steps")
    parser.add_argument(
        "--mpc-iters",
        type=int,
        default=15,
        help="Gradient descent iterations per MPC solve (higher = smoother but slower).",
    )
    parser.add_argument(
        "--sim-decimation",
        type=int,
        default=20,
        help="Number of physics steps to run per control tick (>=1).",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        _run_sim(args)
    except KeyboardInterrupt:
        print("[INFO] Simulation interrupted by user", file=sys.stderr)
    finally:
        simulation_app.close()


def _run_sim(args: argparse.Namespace) -> None:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.utils.configclass import configclass

    from urbansim.scene.urban_scene import UrbanScene
    from urbansim.scene.urban_scene_cfg import UrbanSceneCfg
    from urbansim.primitives.robot.coco import COCO_CFG

    COCO_CFG.init_state.pos = (0.0, 0.0, 0.6)

    @configclass
    class EmptySceneCfg(UrbanSceneCfg):
        num_envs: int = 1
        env_spacing: float = 20.0
        replicate_physics: bool = False
        filter_collisions: bool = True
        scenario_generation_method: str = "predefined"

        ground = AssetBaseCfg(
            prim_path="/World/GroundPlane",
            spawn=sim_utils.GroundPlaneCfg(size=(80.0, 80.0), color=(0.15, 0.15, 0.15)),
        )
        sky_light = AssetBaseCfg(
            prim_path="/World/DebugDomeLight",
            spawn=sim_utils.DomeLightCfg(intensity=8000.0, color=(1.0, 1.0, 1.0)),
        )
        robot = COCO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    sim_cfg = SimulationCfg(dt=args.sim_dt, render_interval=args.render_every)
    sim = SimulationContext(sim_cfg)

    scene_cfg = EmptySceneCfg()
    scene = UrbanScene(scene_cfg)
    scene.generate_scene()

    sim.reset()
    scene.reset()
    scene.write_data_to_sim()
    scene.update(0.0)

    robot = scene["robot"]

    wheel_joint_ids, wheel_joint_names = robot.find_joints(".*wheel_joint")
    axle_joint_ids, axle_joint_names = robot.find_joints("base_to_front_axle_joint")

    if len(wheel_joint_ids) == 0:
        raise RuntimeError("Could not find wheel joints on COCO robot")
    if len(axle_joint_ids) == 0:
        raise RuntimeError("Could not find front axle joint on COCO robot")

    left_wheel_ids = _filter_joint_ids(wheel_joint_ids, wheel_joint_names, "left")
    right_wheel_ids = _filter_joint_ids(wheel_joint_ids, wheel_joint_names, "right")

    if not left_wheel_ids or not right_wheel_ids:
        raise RuntimeError(f"Unable to separate wheel joints into left/right groups: {wheel_joint_names}")

    device = scene.device
    dt_sim = sim.get_physics_dt()

    mpc = MPCTrajectoryFollower(horizon=MPC_HORIZON, dt=MPC_DT, device=device, iterations=args.mpc_iters)
    visualizer = TrajectoryVisualizer()

    position_history: list[np.ndarray] = []
    last_world_waypoints = np.zeros((MPC_HORIZON, 2), dtype=np.float32)
    predicted_world = np.zeros_like(last_world_waypoints)
    current_speed_cmd = 0.0
    current_steer_cmd = 0.0
    sim_decimation = max(1, args.sim_decimation)
    mpc_update_interval = max(1, int(round(MPC_DT / dt_sim)))
    last_mpc_step = -mpc_update_interval

    sim_step = 0
    mpc_tick = 0
    while args.num_steps <= 0 or sim_step < args.num_steps:
        robot_state = _get_robot_state(robot, axle_joint_ids)
        position_history.append(robot_state["position"].copy())
        if len(position_history) > 1200:
            position_history.pop(0)

        if sim_step - last_mpc_step >= mpc_update_interval:
            mpc_tick += 1
            ego_waypoints = _sample_ego_waypoints(
                mpc_tick,
                MPC_HORIZON,
                MPC_DT,
                base_speed=args.forward_speed,
                lateral_amplitude=args.steer_amplitude,
                lateral_frequency=args.steer_frequency,
            )
            last_world_waypoints = _ego_to_world(ego_waypoints, robot_state)
            current_speed_cmd, current_steer_cmd, predicted_world = mpc.solve(
                robot_state, last_world_waypoints
            )
            last_mpc_step = sim_step

        left_wheel_speed, right_wheel_speed = _ackermann_wheel_speeds(
            current_speed_cmd, current_steer_cmd, WHEEL_BASE, TRACK_WIDTH, WHEEL_RADIUS
        )

        _set_wheel_velocity(robot, left_wheel_ids, left_wheel_speed)
        _set_wheel_velocity(robot, right_wheel_ids, right_wheel_speed)
        _set_steering(robot, axle_joint_ids, current_steer_cmd)

        for _ in range(sim_decimation):
            scene.write_data_to_sim()
            should_render = bool(sim.has_gui() and (sim_step % max(1, args.render_every) == 0))
            sim.step(render=should_render)
            scene.update(dt_sim)
            sim_step += 1
            if args.num_steps > 0 and sim_step >= args.num_steps:
                break

        robot_state = _get_robot_state(robot, axle_joint_ids)
        position_history[-1] = robot_state["position"].copy()

        visualizer.update(position_history, last_world_waypoints, predicted_world, robot_state)

        if args.log_every > 0 and sim_step % args.log_every == 0:
            print(
                f"step={sim_step:05d} speed_cmd={current_speed_cmd:4.2f}m/s "
                f"steer_cmd={math.degrees(current_steer_cmd):5.2f}deg "
                f"pos=({robot_state['position'][0]:5.2f},{robot_state['position'][1]:5.2f})",
                flush=True,
            )

    print("[INFO] Completed requested simulation steps.")


def _filter_joint_ids(joint_ids: Sequence[int], joint_names: Sequence[str], keyword: str) -> list[int]:
    return [jid for jid, name in zip(joint_ids, joint_names) if keyword in name]


def _set_wheel_velocity(robot, joint_ids: Sequence[int], wheel_speed: float) -> None:
    if not joint_ids:
        return
    clipped_speed = float(max(-MAX_WHEEL_RATE, min(MAX_WHEEL_RATE, wheel_speed)))
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
    else:
        return outer_angular, inner_angular


if __name__ == "__main__":
    main()
