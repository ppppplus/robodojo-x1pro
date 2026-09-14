"""Two arm views of the user's single fixed-base EX001 articulation."""
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation

from env.robot_manager.ex001_variants import gripper_profile, optical_camera_config, finger_positions, nominal_tcp

def head_camera_config(urdf):
    return optical_camera_config(urdf,'head_pitch_link','camera_head_front_color_optical_frame','cam_head')


class EX001Arm:
    def __init__(self,cfg,side):
        self.robot_type='arm';self.robot_name='ex001';self.is_coupled=True
        self.default_root_pos=list(cfg['default_root_pos']);self.default_root_rot=list(cfg['default_root_rot'])
        self.entity_origin_pose=self.default_root_pos+self.default_root_rot
        self.SceneCfg=None;self.static_camera_list=None;self.grasp_perfect_direction='top_down'
        self.robot_file=str(Path(__file__).resolve().parents[3]/'x1pro/assets')
        self.gripper_profile=gripper_profile();self.gripper_type=self.gripper_profile['name']
        self.urdf_path=str(self.gripper_profile['urdf']);self.mesh_dir=self.robot_file
        self.arm_joints_name=[f'{side}_arm_joint{i}' for i in range(1,7)]
        self.ee_joint_name=self.arm_joints_name[-1];self.ee_link_name=f'{side}_arm_link6';self.base_link='base_link'
        self.gripper_joints_name=[f'{side}_arm_gripper_left_joint',f'{side}_arm_gripper_right_joint']
        self.save_gripper_joints_name=self.arm_joints_name+self.gripper_joints_name
        self.gripper_move={'base':self.gripper_joints_name[0],'sign':1.,'mimic':[self.gripper_joints_name[1],-1.,0.]}
        self.gripper_scale=[finger_positions(0.,self.gripper_profile)[0],finger_positions(self.gripper_profile['motor_open'],self.gripper_profile)[0]]
        self.gripper_tcp_offset=nominal_tcp(self.gripper_profile);self.gripper_bias=float(self.gripper_tcp_offset[0])
        self.gripper_motor_joint_name=f'{side}_arm_gripper'
        self.delta_matrix=np.eye(3);self.inv_delta_matrix=np.eye(3);self.global_trans_matrix=np.eye(3)
        self.ee_type='gripper';self.rotate_lim=[0,0];self.grasp_camera_reference_axis=[1,0,0]
        self.camera=[optical_camera_config(self.urdf_path,self.ee_link_name,f'{side}_arm_gripper_camera_color_frame','cam_wrist',self.gripper_profile['camera_type'])]
        if side=='left':self.camera.append(head_camera_config(self.urdf_path))
    def sync_gripper_motor_target(self, articulation, finger_target, env_ids):
        indices,_=articulation.find_joints([self.gripper_motor_joint_name])
        p=self.gripper_profile
        motor=(finger_target[:,0:1]-p['offset'])/p['multiplier']
        articulation.set_joint_position_target(motor,joint_ids=indices,env_ids=env_ids)

class EX001Left(EX001Arm):
    def __init__(self,cfg):super().__init__(cfg,'left')
class EX001Right(EX001Arm):
    def __init__(self,cfg):super().__init__(cfg,'right')
