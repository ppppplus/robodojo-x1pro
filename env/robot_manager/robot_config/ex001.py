"""EX001 imported user geometry; scripted-expert simulation drives."""
from pathlib import Path
import xml.etree.ElementTree as ET
import os
from env.robot_manager.ex001_variants import gripper_profile, selected_usd, finger_positions
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils

def spawn_ex001(prim_path,cfg,translation=None,orientation=None,**kwargs):
    from pxr import Usd,UsdPhysics
    prim=sim_utils.spawn_from_usd(prim_path,cfg,translation=translation,orientation=orientation,**kwargs)
    for child in list(Usd.PrimRange(prim)):
        path=str(child.GetPath())
        if ('_gripper_left_link/' in path or '_gripper_right_link/' in path) and child.IsInstance():child.SetInstanceable(False)
    updated=[]
    for child in Usd.PrimRange(prim):
        path=str(child.GetPath())
        if ('_gripper_left_link' in path or '_gripper_right_link' in path) and child.HasAPI(UsdPhysics.MeshCollisionAPI):
            UsdPhysics.MeshCollisionAPI(child).GetApproximationAttr().Set('convexDecomposition');updated.append(path)
    print('EX001_FINGER_COLLIDERS '+str(updated),flush=True)
    if len(updated)!=4:raise RuntimeError(f'Expected four EX001 finger collision meshes, got {len(updated)}')
    return prim

def get_robot_config():
    asset=Path(__file__).resolve().parents[3]/'x1pro/assets'
    profile=gripper_profile();usd_path=selected_usd(profile)
    if not usd_path.is_file():raise FileNotFoundError(usd_path)
    tree=ET.parse(profile['urdf']).getroot();acts={}
    initial={}
    for j in tree.findall('joint'):
        if j.get('type')=='fixed':continue
        name=j.get('name');lim=j.find('limit');force=float(lim.get('effort'));speed=float(lim.get('velocity'))
        if name=='lift_joint':kp,kd=20000.,1000.;q=.55
        elif '_gripper_' in name:
            kp,kd=(40.,2.) if profile['rotary'] else (1500.,20.)
            q=finger_positions(profile['motor_open'],profile)[0 if '_left_' in name else 1]
        elif name.endswith('_gripper'):kp,kd=5.,.5;q=profile['motor_open']
        elif name.startswith('head_'):kp,kd=30.,3.;q=0.
        elif 'wheel' in name:kp,kd=5.,1.;q=0.
        else:kp,kd=1000.,60.;q=.5 if name.endswith('joint2') else (-.8 if name.endswith('joint3') else 0.)
        initial[name]=q
        acts[name]=ImplicitActuatorCfg(joint_names_expr=[name],stiffness=kp,damping=kd,effort_limit_sim=force,velocity_limit_sim=speed)
    return ArticulationCfg(spawn=sim_utils.UsdFileCfg(func=spawn_ex001,usd_path=str(usd_path),rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True,max_depenetration_velocity=1.),articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False,fix_root_link=True,solver_position_iteration_count=16,solver_velocity_iteration_count=4)),init_state=ArticulationCfg.InitialStateCfg(joint_pos=initial),actuators=acts,soft_joint_pos_limit_factor=1.)
