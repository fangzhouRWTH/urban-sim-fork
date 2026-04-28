# URBAN-SIM — Codebase Conclusion

This document summarizes the repository structure, main components, and the purpose of key modules in the `urban-sim` project.

## High-level overview

- **Goal:** URBAN-SIM is a large-scale urban robot learning platform built on NVIDIA Omniverse / IsaacSim, designed for scalable RL training, simulation, and data collection in procedurally generated urban scenes.
- **Primary capabilities:** procedural scene generation, multi-env (sync/async) simulation, robot embodiments (COCO, Unitree, etc.), asset conversion and management, RL training and evaluation pipelines.

## Top-level layout (key folders)

- `urbansim/` — Core Python package containing environment code, scene and asset configuration, primitives (robots, navigation), utilities, and entry points for running environments.
- `assets/` — All 3D assets, materials, USDS, checkpoints, and additional data used by simulations.
- `apps/`, `_isaac_sim/` — Isaac/Omniverse app-related kits and runtime links.
- `documentation/` — Project docs, examples, and build artifacts for the website.
- `isaac_source/`, `meta_source/`, `rl_source/` — Source code and examples for IsaacLab/rl integrations and task-specific modules (RL algos, envs, tools).
- `scripts/` — Utility scripts (asset collectors, converters, controllers, tools) used during setup and content pipeline.
- `configs/` — Predefined YAML configs for environments, training, and experiments.
- `urbansim.egg-info`, `pyproject.toml`, `setup.py`, `VERSION`, license files — packaging and metadata.

## Important code components and roles

- `urbansim/envs/` — Environment definitions and wrappers.
  - `separate_envs/random_env.py` — Example entry that composes a `BaseEnvCfg` which sets up a navigation environment using `UrbanScene`, COCO robot, and procedural generation parameters; it registers a Gym environment `UrbanSim-Base` and provides a `main()` loop to run the env (used for debugging / local runs / data collection).
  - `abstract_env.py` — Environment glue that instantiates `UrbanScene`, applies fixes (e.g., material path corrections), connects sensors and cameras, and ties IsaacLab managers to Gym API.

- `urbansim/primitives/` — Reusable building blocks.
  - `primitives/robot/coco.py` — COCO robot `ArticulationCfg` and default spawn/usd paths; used by several envs.
  - `primitives/navigation/` — Scene configuration classes, observation/reward/termination definitions.

- `scripts/tools/converters/convert_asset.py` — Asset conversion utility that batch-converts source models (e.g., `.glb`) to USD using IsaacLab's `MeshConverter` and per-asset parameter JSONs (scale, collision, mass). Helpful during dataset preparation and onboarding new assets.

- `urbansim/learning/RL/` — Training and evaluation entrypoints.
  - `train.py`, `play.py` — wrappers around RL backends for large-scale and local experiments; accept config files located under `configs/env_configs/`.

- `assets/` organization
  - `assets/objects/` — raw model inputs (.glb/.obj/.fbx). `scripts/tools/converters` processes these into `assets/usds/`.
  - `assets/robots/` — robot models (e.g., `coco_one`) and their material folders.
  - `assets/ckpts/` — model checkpoints for pretrained policies and experiments.

## Workflow highlights

- Development and testing: run example env scripts such as `python urbansim/envs/separate_envs/random_env.py` (options: `--num_envs`, `--use_async`, `--scenario_type`, `--enable_cameras`).
- Asset pipeline: gather raw models with `scripts/tools/collectors/collect_asset.py` and convert them with `scripts/tools/converters/convert_asset.py` (this handles mass, collision, scale via JSON parameter files).
- Training pipeline: use `urbansim/learning/RL/train.py` with environment configs in `configs/env_configs/navigation/` to start RL experiments.

## Troubleshooting notes (common pain points)

- IsaacSim / Omniverse paths: users must correctly set `ISAACSIM_PATH` and `ISAACSIM_PYTHON_EXE`; many runtime errors are caused by incorrect IsaacSim installation or missing kit versions.
- Asset paths in USD may reference absolute developer paths; `abstract_env.py` contains a post-load pass that attempts to fix shader texture paths (search for `coco-white.png`). If textures are missing, check `assets/robots/*/materials` and USD references.
- Converting assets: keep the JSON param files in `assets/adj_parameter_folder/` matched to filenames so `convert_asset.py` picks correct scale/collision settings.

## Suggested next steps for new contributors

- Read `README.md` and `documentation/` to set up IsaacSim and the `urbansim` conda environment.
- Run `scripts/tools/converters/convert_asset.py` on a small model and verify the generated USD in `assets/usds/`.
- Launch a single env with `random_env.py` using `--num_envs 1 --enable_cameras` to inspect robot and sensor setup.

---
Generated on: 2026-04-23

## Detailed module mapping

Below is a more detailed mapping of important modules to files and their responsibilities. Click paths to open files in the workspace.

- **Environments**
  - [urbansim/envs/abstract_env.py](urbansim/envs/abstract_env.py): Core environment glue; stage creation, sensor setup, material fixes, and Gym adapter.
  - [urbansim/envs/abstract_rl_env.py](urbansim/envs/abstract_rl_env.py): RL-specialized environment base (training utilities, vectorization helpers).
  - [urbansim/envs/separate_envs/random_env.py](urbansim/envs/separate_envs/random_env.py): Example navigation env entry (procedural generation, COCO robot, main loop).
  - [urbansim/envs/separate_envs/pg_env.py](urbansim/envs/separate_envs/pg_env.py): Procedural generation environment entry for large-scale PG experiments.
  - [urbansim/envs/separate_envs/predefined_env.py](urbansim/envs/separate_envs/predefined_env.py): Prebuilt scene setups and deterministic scenarios.

- **Scene & procedural generation**
  - [urbansim/scene/urban_scene.py](urbansim/scene/urban_scene.py): Scene manager that assembles environment prims and entities.
  - [urbansim/scene/urban_scene_cfg.py](urbansim/scene/urban_scene_cfg.py): Configuration classes for procedural generation and scene composition.
  - [urbansim/scene/utils.py](urbansim/scene/utils.py): Helpers for scene construction and asset placement.

- **Primitives (robots, navigation, locomotion)**
  - [urbansim/primitives/robot/coco.py](urbansim/primitives/robot/coco.py): COCO robot articulation configuration and USD path references.
  - [urbansim/primitives/robot/unitree_g1.py](urbansim/primitives/robot/unitree_g1.py): Unitree G1 robot config.
  - [urbansim/primitives/robot/unitree_go2.py](urbansim/primitives/robot/unitree_go2.py): Unitree GO2 config.
  - [urbansim/primitives/robot/anymal_c.py](urbansim/primitives/robot/anymal_c.py): Anymal-C robot config.
  - [urbansim/primitives/navigation/random_env_cfg.py](urbansim/primitives/navigation/random_env_cfg.py): Observation/Reward/Termination/Command definitions for navigation tasks.
  - [urbansim/primitives/navigation/pg_env_cfg.py](urbansim/primitives/navigation/pg_env_cfg.py): PG-specific scene config for procedural generation experiments.
  - [urbansim/primitives/navigation/predefined.py](urbansim/primitives/navigation/predefined.py): Deterministic scene presets and maps.
  - [urbansim/primitives/locomotion/general.py](urbansim/primitives/locomotion/general.py): Locomotion helpers and configs.

- **RL training & evaluation**
  - [urbansim/learning/RL/train.py](urbansim/learning/RL/train.py): Training entrypoint; parses experiment configs and launches training backends.
  - [urbansim/learning/RL/play.py](urbansim/learning/RL/play.py): Evaluation / interactive play entrypoint for pretrained policies.

- **Utilities & CLI**
  - [urbansim/cli_args.py](urbansim/cli_args.py): Centralized CLI argument helpers used across scripts.
  - [urbansim/utils/config.py](urbansim/utils/config.py): Config helpers and dataclass utilities.
  - [urbansim/utils/utils.py](urbansim/utils/utils.py): General-purpose helpers.
  - [urbansim/utils/geometry.py](urbansim/utils/geometry.py): Geometry utilities used by scene and navigation modules.
  - [urbansim/utils/map_manager.py](urbansim/utils/map_manager.py): Map and region management utilities.
  - [urbansim/utils/random_utils.py](urbansim/utils/random_utils.py): Randomization helpers for procedural generation.
  - [urbansim/utils/coordinates_shift.py](urbansim/utils/coordinates_shift.py): Coordinate transforms (local <-> global)
  - [urbansim/utils/math.py](urbansim/utils/math.py): Math helpers.

- **Asset & data pipeline scripts**
  - [scripts/tools/converters/convert_asset.py](scripts/tools/converters/convert_asset.py): Batch conversion from `.glb`/`.obj` to `.usd` with per-asset parameters.
  - [scripts/tools/collectors/collect_asset.py](scripts/tools/collectors/collect_asset.py): Asset collector/downloader used to gather raw meshes and metadata.

- **Other important files**
  - [urbansim/version.py](urbansim/version.py): Package version.

If you want, I can now expand any single module entry into a sub-section showing key classes, functions, and example usage (e.g., `UrbanScene` methods, `COCO_CFG` fields, or `MeshConverterCfg` parameters). Which module should I expand first?
