"""Single source of truth for X1 Pro gripper assets and joint mappings."""
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

ASSET=Path(__file__).resolve().parents[2]/'x1pro/assets'
VARIANTS=('fx001_h_evt1','rm001_g1_dvt1')

def selected_gripper(value=None):
    value=value or os.environ.get('ROBODOJO_EX001_GRIPPER',VARIANTS[0])
    value={'fx001':VARIANTS[0],'parallel':VARIANTS[0],'rm001':VARIANTS[1],'rotary':VARIANTS[1]}.get(value,value)
    if value not in VARIANTS:raise ValueError(f'Unsupported EX001 gripper {value!r}; choose {VARIANTS}')
    return value

def gripper_profile(value=None):
    name=selected_gripper(value);rotary=name==VARIANTS[1]
    folder=ASSET/name
    return {'name':name,'rotary':rotary,'urdf':folder/'kinematics.urdf',
            'usd_realistic':folder/'appearance/x1pro_appearance.usd',
            'motor_open':1.7 if rotary else 5.,'multiplier':.5 if rotary else .00852694,
            'offset':-.073 if rotary else 0.,'camera_type':'d435',
            'camera_intrinsics_source':'nominal simulation D435 profile; no factory intrinsics supplied',
            'camera_extrinsics_source':'selected gripper URDF fixed chain'}

def selected_usd(profile=None,appearance=None):
    p=profile or gripper_profile();appearance=appearance or os.environ.get('ROBODOJO_EX001_APPEARANCE','realistic')
    if appearance != 'realistic':raise ValueError('Only the packaged realistic X1 Pro USD is supported')
    override=os.environ.get('ROBODOJO_EX001_USD')
    if override:
        # Reject a known asset from the other variant rather than mismatching control and visuals.
        path=Path(override).resolve()
        other=gripper_profile(VARIANTS[0] if p['rotary'] else VARIANTS[1])
        if path == other['usd_realistic'].resolve():
            raise ValueError('ROBODOJO_EX001_USD belongs to the other gripper; unset it or select a matching asset')
        return path
    return p['usd_'+appearance]

def finger_positions(motor,profile=None):
    p=profile or gripper_profile();left=motor*p['multiplier']+p['offset']
    return left,-left

def map_mimic_targets(target,names,profile=None):
    p=profile or gripper_profile();idx={n:i for i,n in enumerate(names)}
    for side in ('left','right'):
        left,right=finger_positions(target[...,idx[side+'_arm_gripper']],p)
        target[...,idx[side+'_arm_gripper_left_joint']]=left;target[...,idx[side+'_arm_gripper_right_joint']]=right
    return target

def fixed_chain_transform(urdf,parent,child):
    joints={j.find('child').get('link'):j for j in ET.parse(urdf).getroot().findall('joint')};chain=[]
    while child!=parent:
        j=joints[child]
        if j.get('type')!='fixed':raise ValueError(f'{child} is not fixed to {parent}')
        chain.append(j);child=j.find('parent').get('link')
    transform=np.eye(4)
    for j in reversed(chain):
        o=j.find('origin');local=np.eye(4)
        if o is not None:
            local[:3,3]=np.fromstring(o.get('xyz','0 0 0'),sep=' ')
            local[:3,:3]=Rotation.from_euler('xyz',np.fromstring(o.get('rpy','0 0 0'),sep=' ')).as_matrix()
        transform=transform@local
    return transform

def optical_camera_config(urdf,parent,optical_frame,name,camera_type='d435'):
    transform=fixed_chain_transform(urdf,parent,optical_frame)
    transform[:3,:3]=transform[:3,:3]@np.diag([1.,-1.,-1.])
    xyzw=Rotation.from_matrix(transform[:3,:3]).as_quat()
    return {'link':parent,'name':name,'type':camera_type,'mesh':'pinhole','pos':transform[:3,3].tolist(),'ori':xyzw[[3,0,1,2]].tolist()}

def nominal_tcp(profile=None):
    """Midpoint of both tip frames at the closed motor coordinate (simulation reference)."""
    p=profile or gripper_profile();r=ET.parse(p['urdf']).getroot();points=[]
    for side in ('left','right'):
        j=r.find(f"joint[@name='right_arm_gripper_{side}_joint']");o=j.find('origin');T=np.eye(4)
        T[:3,3]=np.fromstring(o.get('xyz'),sep=' ');T[:3,:3]=Rotation.from_euler('xyz',np.fromstring(o.get('rpy'),sep=' ')).as_matrix()
        axis=np.fromstring(j.find('axis').get('xyz'),sep=' ');q=finger_positions(0.,p)[0 if side=='left' else 1]
        if j.get('type')=='prismatic':T[:3,3]+=T[:3,:3]@axis*q
        else:T[:3,:3]=T[:3,:3]@Rotation.from_rotvec(axis*q).as_matrix()
        tip=fixed_chain_transform(p['urdf'],f'right_arm_gripper_{side}_link',f'right_arm_gripper_{side}_tip_link')
        points.append((T@tip)[:3,3])
    return np.mean(points,axis=0)
