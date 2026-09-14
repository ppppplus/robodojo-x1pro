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
