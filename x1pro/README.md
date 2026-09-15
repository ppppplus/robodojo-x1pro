# X1 Pro integration for RoboDojo

This fork adds the EX001/X1 Pro fixed-base dual-arm robot with two selectable end effectors:

| `ROBODOJO_EX001_GRIPPER` | Geometry | Joint motion |
| --- | --- | --- |
| `fx001_h_evt1` (default) | FX001 parallel gripper | Prismatic fingers |
| `rm001_g1_dvt1` | RM001 rotary gripper | Revolute fingers |

The robot is registered through `env/robot_manager/robot_manager.py`; the configs are `env_cfg/robot/ex001.yml` and `env_cfg/scene/ex001.yml`. The visual and collision model loads from `x1pro/assets/<gripper>/appearance/x1pro_appearance.usd`, which references `../usd_control/ex001.usd` and its configuration layers. The `kinematics.urdf` alongside it contains the joints, limits and fixed camera frames used by IK and the simulator configuration. Its visual/collision mesh elements were removed because the USD is the rendered and collided model. This URDF alone is **not** a complete mesh-based robot model.

The initial model matches the uploaded DVT2 base/PVT1 arm geometry, and camera intrinsics are nominal D435 simulation values. The model geometry, contact coefficients and camera calibration have not been verified against an individual physical robot.

## Use with an existing RoboDojo installation

Install the upstream RoboDojo environment and `Assets/` as described by the upstream project. X1 Pro model files are **not** in this code repository: both gripper variants, their USD layers, textures and kinematics URDFs are installed separately under the ignored `x1pro/assets/` directory. The model bundle must have this layout:

```text
x1pro/assets/
  fx001_h_evt1/{kinematics.urdf,appearance/,usd_control/}
  rm001_g1_dvt1/{kinematics.urdf,appearance/,usd_control/}
```

If you already have the model bundle locally, link it into the fork without copying the large files:

```bash
bash x1pro/init_assets.sh --from-dir /path/to/x1pro/assets
```

The public bundle can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1yjbJ4eWfCZwoM0ejFrjkhuAvDq_T4YAK?usp=sharing). Its root contains the two variant directories expected by the installer. For a command-line installation:

```bash
python3 -m pip install gdown
mkdir -p .cache
python3 -m gdown --folder \
  "https://drive.google.com/drive/folders/1yjbJ4eWfCZwoM0ejFrjkhuAvDq_T4YAK?usp=sharing" \
  -O .cache/x1pro-assets
bash x1pro/init_assets.sh --from-dir .cache/x1pro-assets
python3 x1pro/check_install.py
```

For a separately hosted Hugging Face **dataset** repository containing the same `x1pro/assets/` layout, use:

```bash
bash x1pro/init_assets.sh --hf-repo OWNER/DATASET
```

This fetches only `x1pro/assets/` into the ignored `.cache/` directory using sparse checkout and Git LFS, then links it into the code tree. The X1 Pro examples also use the bowl, bamboo texture and HDR from upstream `Assets/`. Check the local files after installing both asset sets:

```bash
python x1pro/check_install.py
```

Select a gripper **before** launching Isaac Sim:

```bash
export ROBODOJO_EX001_GRIPPER=fx001_h_evt1
# or: export ROBODOJO_EX001_GRIPPER=rm001_g1_dvt1
```

Use `env_cfg/robot/ex001.yml` for the RoboDojo robot configuration. To preview the full X1 Pro noodle workstation with the FX001 parallel gripper:

```bash
python x1pro/noodle_scene/noodle_expert/collect.py \
  --output data/x1pro_scene_preview --scene-only \
  --headless --enable_cameras --device cuda:0
```

This scene uses a bamboo table, two independent rigid noodle bundles on one plate, a white bowl, two-basket cooker and three seasoning bottles. The chili and salt bottles have visible granular contents and real cap apertures. `--scene-only` previews the environment; omitting it runs the scripted physical grasp/pour collector. The collector is a custom X1 Pro example, not an official RoboDojo DataGen task. Its pour-only mode accepts `--initial-episode PATH` to initialize from a separate earlier grasp episode. See `x1pro/noodle_scene/noodle_expert/README.md` for the data schema and limitations.

`x1pro/noodle_scene/chili_pour/simulate.py` makes a separate PhysX particle pour video. The bottle uses a prescribed kinematic motion, so this is not a robot grasp demonstration. `x1pro/collect_stack.py` is the older X1 Pro block-stacking example.

## Portability and distribution

The sample code finds the repository root from its own file path. It does not use the original workspace location. RoboDojo's downloaded `Assets/` remain provided by the upstream asset installer. Existing USD scene exports and recorded HDF5 datasets may contain paths from the machine that produced them; regenerate those on the destination machine. The collector previously ran in the source workspace. This isolated fork passed static asset and syntax checks; a fresh runtime render in the fork is still pending because Isaac Sim stalled during extension startup on this machine. An independent install also needs validation.

The X1 Pro USD and URDF model files are derived from user-provided X1 Pro URDF/mesh archives. Their right to redistribution has not been established in this workspace. The code-only branch can be shared independently; publishing the model bundle, even in a separate dataset repository, requires checking the asset owner's terms. RoboDojo's [license](../LICENSE) allows non-commercial research, education and evaluation, and requires separate permission for commercial use.

## Replay a real X1 Pro trajectory

The noodle collector can replay a robot-bridge JSON trajectory without teleporting the robot or objects:

```bash
python x1pro/noodle_scene/noodle_expert/collect.py \
  --output data/replay_result \
  --replay-json /path/to/episode.json \
  --fast --headless --enable_cameras --device cuda:0
```

The JSON uses the bridge `follow_*_joint_position`, `follow_*_gripper`, `head_yaw`, and `head_pitch` fields. The replay accepts both the small FX001 joint-unit encoding and the historical raw follower/master gripper encoder. For the latter it calibrates the trajectory range so the low signal is closed and the high signal is open, and prints a `REPLAY_GRIPPER` diagnostic. Outputs include `summary.json`, `episode.hdf5`, and `overview.mp4`; replay runs are marked `replay_complete` rather than as generated expert demonstrations.
