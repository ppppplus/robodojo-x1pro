# X1 Pro photo-based noodle workstation

This custom Isaac Sim example uses the FX001 parallel gripper, the two-basket cooker, two independent rigid noodle bundles on one plate, a bowl and three seasoning bottles. The source layout is estimated from a user-provided photo. The camera mounts are taken from the packaged kinematics URDF, with nominal D435 intrinsics.

`scene_builder.py` defines the workcell, `seasoning.py` defines the visible salt/chili grains and perforated shaker caps, `collect.py` drives the robot and records physical contact-driven demonstrations, and `validate.py` checks a recorded episode. The static contents in the noodle task are visual approximations. For freely moving granular chili, use `../chili_pour/simulate.py`.

From the fork root, after installing RoboDojo and its Assets:

```bash
export ROBODOJO_EX001_GRIPPER=fx001_h_evt1
python x1pro/noodle_scene/noodle_expert/collect.py \
  --output data/x1pro_preview --scene-only \
  --headless --enable_cameras --device cuda:0
```

Omit `--scene-only` for the scripted manipulation collector. `--pour-only --initial-episode PATH` initializes a separate pour episode from a successful grasp episode's terminal poses; it is not an uninterrupted full rollout. The current collector explicitly selects FX001 and is not an RM001 pour expert.

The collector records 23 absolute joint position targets at 240 Hz, 23 joint states, object poses and three mounted RGB camera streams at 10 FPS. The noodles are rigid items with strand visuals. Robot gravity is compensated, self-collision is disabled, and food/contact coefficients are estimates. There is no flexible noodle, boiling or water simulation. Re-running in a new simulator build may change contact outcomes.

Previously generated successful episodes were separately initialized: grasp into cooker (~38.1 s) and pour into bowl (~36.8 s). They are not shipped in this source fork; supply your own output directory when running.

## OpenPI policy rollout

The collector can receive actions from the matching X1 Pro `smp2smp` OpenPI
checkpoint over its MsgPack WebSocket server. Start
`serve_x1pro_checkpoint.py` from the OpenPI `x1pro_smp2smp` branch, then run:

```bash
export ROBODOJO_PYTHON=/path/to/robodojo-env/bin/python
export ROBODOJO_GPU=1
export OPENPI_X1PRO_PORT=8010
bash x1pro/noodle_scene/noodle_expert/run_openpi_rollout.sh \
  data/openpi_rollout 90 place_noodles_in_pot
```

The third argument accepts `place_noodles_in_pot`,
`sprinkle_chili_seasoning`, `sprinkle_green_onions`, `sprinkle_salt`, or
`transfer_noodles_to_bowl`. Do not paraphrase the language instruction unless
you intentionally want to test prompt sensitivity: the collector selects the
exact training prompt for each task by default. `--openpi-prompt` remains
available as an explicit override when calling `collect.py` directly.

The deployed observation is a `7x29` sequence: 14 follower values, 14 virtual
master values, and one phase value for three history rows, the current row, and
three future rows. Each prediction supplies a 20-row action horizon; the bridge
skips rows 0-2 for measured latency and executes rows 3-12 at 15 Hz before
requesting another prediction. Live follower state is recorded after every
executed row. The three RGB inputs are resized to `240x320` and mapped as
`cam_head -> face_view`, `cam_left_wrist -> left_wrist_view`, and
`cam_right_wrist -> right_wrist_view`.

The scene uses lift `0.45`, head pitch `0.1`, and head yaw `-0.1`. Per-task
reference states and gripper encoder ranges are defined in `openpi_bridge.py`.
A single unreachable Cartesian command holds that arm for one controller frame
and records the IK error in `openpi_policy.json`; it does not stop the remaining
policy stream.
