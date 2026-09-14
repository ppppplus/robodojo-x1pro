"""URDF-derived forward/inverse kinematics for EX001 Cartesian scripted expert."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

class Kinematics:
    def __init__(self, urdf, side='right', base=(0,-.95,0),lift=.2,tcp=.163504):
        self.side=side;self.root=ET.parse(urdf).getroot();self.tcp=np.array([tcp,0.,0.]) if np.isscalar(tcp) else np.asarray(tcp,dtype=float)
        self.names=[f'{side}_arm_joint{i}' for i in range(1,7)]
        self.by_child={j.find('child').get('link'):j for j in self.root.findall('joint')}
        chain=[];link=f'{side}_arm_gripper_base_link'
        while link in self.by_child:
            j=self.by_child[link];chain.append(j);link=j.find('parent').get('link')
        self.chain=[]
        for j in reversed(chain):
            o=j.find('origin');T=np.eye(4)
            if o is not None:T[:3,3]=np.fromstring(o.get('xyz','0 0 0'),sep=' ');T[:3,:3]=Rotation.from_euler('xyz',np.fromstring(o.get('rpy','0 0 0'),sep=' ')).as_matrix()
            axis=j.find('axis');axis=np.fromstring(axis.get('xyz'),sep=' ') if axis is not None else np.array([1.,0,0])
            self.chain.append((j.get('name'),j.get('type'),T,axis))
        self.base=np.eye(4);self.base[:3,3]=base;self.base[:3,:3]=Rotation.from_euler('z',np.pi/2).as_matrix();self.lift=lift
        limits={j.get('name'):j.find('limit').attrib for j in self.root.findall('joint') if j.get('name') in self.names}
        self.lower=np.array([float(limits[n]['lower']) for n in self.names]);self.upper=np.array([float(limits[n]['upper']) for n in self.names])
    def fk(self,q,tcp=True):
        T=self.base.copy();qd=dict(zip(self.names,q));qd['lift_joint']=self.lift
        for name,typ,origin,axis in self.chain:
            T=T@origin
            if typ in ['revolute','continuous']:
                R=np.eye(4);R[:3,:3]=Rotation.from_rotvec(axis*qd.get(name,0)).as_matrix();T=T@R
            elif typ=='prismatic':T[:3,3]+=T[:3,:3]@axis*qd.get(name,0)
        if tcp:T[:3,3]+=T[:3,:3]@self.tcp
        return T
    def ik(self,pos,rot,q0,attempts=8):
        def residual(q):
            T=self.fk(q);return np.r_[T[:3,3]-pos,.15*Rotation.from_matrix(rot@T[:3,:3].T).as_rotvec()]
        best=None;rng=np.random.default_rng(0)
        seeds=[np.clip(q0,self.lower+1e-7,self.upper-1e-7)]
        seeds += [rng.uniform(self.lower,self.upper) for _ in range(attempts-1)]
        for seed in seeds:
            fit=least_squares(residual,seed,bounds=(self.lower+1e-8,self.upper-1e-8),max_nfev=120,ftol=1e-9,xtol=1e-9,gtol=1e-9)
            err=np.linalg.norm(residual(fit.x))
            if best is None or err<best[0]:best=(err,fit.x)
            if err<1e-5:return fit.x
        raise RuntimeError(f'IK failed at {np.asarray(pos).tolist()}, residual {best[0]:.6f}')
