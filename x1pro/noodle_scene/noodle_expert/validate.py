"""Validate actual recorded arrays, gripper mapping and mounted camera transforms."""
import json,sys,xml.etree.ElementTree as ET
from pathlib import Path
import h5py,numpy as np
from scipy.spatial.transform import Rotation
out=Path(sys.argv[1]);f=h5py.File(out/'episode.hdf5','r');summary=json.loads((out/'summary.json').read_text());names=json.loads(f.attrs['joint_names_json']);q=f['observations/joint_position'][:];act=f['actions/joint_position'][:];steps=f['image_step'][:];dt=float(f.attrs['physics_dt']);conditions=json.loads(f.attrs['conditions_json'])
checks={}
checks['state_action_lengths']=len(q)==len(act)+1
checks['image_step_valid']=bool(steps[0]==0 and steps[-1]==len(act) and np.all(np.diff(steps)>0))
checks['timestamps']=bool(np.allclose(f['timestamp'][:],np.arange(len(q))*dt))
checks['finite_states']=bool(np.isfinite(q).all() and np.isfinite(act).all() and np.isfinite(f['observations/object_pose_wxyz'][:]).all())
checks['parallel_gripper']=f.attrs['gripper_type']=='fx001_h_evt1'
for side in ['left','right']:
 m=act[:,names.index(side+'_arm_gripper')]*.00852694
 checks[side+'_mimic_actions']=bool(np.allclose(act[:,names.index(side+'_arm_gripper_left_joint')],m,atol=1e-6) and np.allclose(act[:,names.index(side+'_arm_gripper_right_joint')],-m,atol=1e-6))
initial=f['observations/object_pose_wxyz'][0,:2,:3]
if f.attrs['task']=='pour_two_noodles_into_bowl':
 bp=f['observations/object_pose_wxyz'][0,2];br=Rotation.from_quat(bp[[4,5,6,3]]).as_matrix();rel=(initial-bp[:3])@br;checks['two_objects_initially_in_basket']=bool(np.all(np.linalg.norm(rel[:,:2],axis=1)<.078) and np.all((rel[:,2]>.002)&(rel[:,2]<.14)))
else:checks['two_objects_same_plate']=bool(np.all(np.linalg.norm(initial[:,:2]-[.195,-.245],axis=1)<.10) and np.linalg.norm(initial[0]-initial[1])>.05)
camnames=json.loads(f.attrs['camera_names_json']);camera_errors={}
if summary['expert_demonstration']:checks['three_mounted_cameras']=all(n in camnames for n in ['cam_head','cam_left_wrist','cam_right_wrist'])
root=ET.parse(Path(__file__).resolve().parents[2]/'assets/fx001_h_evt1/kinematics.urdf').getroot();bychild={j.find('child').get('link'):j for j in root.findall('joint')}
def fk(link,qs):
 chain=[]
 while link in bychild:j=bychild[link];chain.append(j);link=j.find('parent').get('link')
 T=np.eye(4);T[:3,3]=conditions['base_position'];T[:3,:3]=Rotation.from_euler('z',90,degrees=True).as_matrix();values=dict(zip(names,qs))
 for j in reversed(chain):
  O=np.eye(4);o=j.find('origin')
  if o is not None:O[:3,3]=np.fromstring(o.get('xyz','0 0 0'),sep=' ');O[:3,:3]=Rotation.from_euler('xyz',np.fromstring(o.get('rpy','0 0 0'),sep=' ')).as_matrix()
  T=T@O
  if j.get('type')=='fixed':continue
  axis=np.fromstring(j.find('axis').get('xyz'),sep=' ');v=values[j.get('name')]
  if j.get('type')=='prismatic':T[:3,3]+=T[:3,:3]@axis*v
  else:T[:3,:3]=T[:3,:3]@Rotation.from_rotvec(axis*v).as_matrix()
 return T
poses=f['observations/camera_to_world'][:]
for ci,name in enumerate(camnames):
 ds=f['observations/images/'+name];checks[name+'_image_count']=len(ds)==len(steps);checks[name+'_nonblank']=bool(all(ds[i].std()>4 for i in np.linspace(0,len(ds)-1,8,dtype=int)))
 if name=='overview':continue
 frame='camera_head_front_color_optical_frame' if name=='cam_head' else ('left' if 'left' in name else 'right')+'_arm_gripper_camera_color_frame';ep=[];er=[]
 for i in np.unique(np.linspace(0,len(steps)-1,min(60,len(steps)),dtype=int)):
  expected=fk(frame,q[steps[i]]);expected[:3,:3]=expected[:3,:3]@np.diag([1.,-1.,-1.]);actual=poses[i,ci];ep.append(np.linalg.norm(expected[:3,3]-actual[:3,3]));er.append(Rotation.from_matrix(expected[:3,:3].T@actual[:3,:3]).magnitude())
 camera_errors[name]={'max_position_error_m':float(max(ep)),'max_rotation_error_rad':float(max(er))};checks[name+'_urdf_mount_sync']=max(ep)<.003 and max(er)<.005
checks['success_flag_consistency']=bool(f.attrs['expert_demonstration'])==bool(summary['expert_demonstration'])
basket_quat=f['observations/object_pose_wxyz'][:,2,3:7];basket_r=Rotation.from_quat(basket_quat[:,[1,2,3,0]]).as_matrix();basket_tilt=np.rad2deg(np.arccos(np.clip(basket_r[:,2,2],-1,1)))
result={'max_actual_basket_tilt_degrees':float(basket_tilt.max()),'all_passed':bool(all(checks.values())),'checks':{k:bool(v) for k,v in checks.items()},'camera_errors':camera_errors,'physical_task_success':bool(summary['expert_demonstration']),'state_count':len(q),'action_count':len(act),'frames_per_camera':len(steps),'cameras':camnames}
(out/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));sys.exit(0 if result['all_passed'] else 1)
