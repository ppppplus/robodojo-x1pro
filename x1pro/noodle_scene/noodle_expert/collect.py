"""Contact-driven X1 Pro noodle transfer in the photo-reconstructed workstation."""
import os,sys,json,argparse,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'x1pro'),str(HERE)]
os.environ['ROBODOJO_EX001_GRIPPER']='fx001_h_evt1';os.environ.pop('ROBODOJO_EX001_USD',None)
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--fast',action='store_true');p.add_argument('--basket-test',action='store_true');p.add_argument('--pour-only',action='store_true');p.add_argument('--initial-episode',default=None);p.add_argument('--force-test',action='store_true');p.add_argument('--scene-only',action='store_true');p.add_argument('--replay-json',default=None,help='Replay a robot-bridge JSON trajectory using follow_* joint states');AppLauncher.add_app_launcher_args(p);args=p.parse_args()
OUT=Path(args.output);OUT.mkdir(parents=True,exist_ok=True)
if (OUT/'summary.json').exists():raise FileExistsError(OUT)
(OUT/'collector_source.py').write_text(Path(__file__).read_text());(OUT/'scene_builder_source.py').write_text((HERE/'scene_builder.py').read_text());(OUT/'seasoning.py').write_text((HERE/'seasoning.py').read_text())
app=AppLauncher(args).app;code=1;writers=[];h5=None
try:
 import numpy as np,torch,h5py
 from PIL import Image,ImageDraw,ImageFont
 from scipy.spatial.transform import Rotation,Slerp
 from scipy.optimize import least_squares
 from pxr import Usd,UsdGeom,UsdPhysics,PhysxSchema,UsdShade,Gf,Sdf
 import isaaclab.sim as su
 from isaaclab.assets import Articulation,RigidObject,RigidObjectCfg
 import omni.replicator.core as rep
 from env.robot_manager.ex001_variants import gripper_profile,map_mimic_targets,nominal_tcp
 from env.robot_manager.robot_config.ex001 import get_robot_config
 from env.robot_manager.robot_class.ex001 import EX001Left,EX001Right
 from env_cfg.camera.template import D435
 from utils.save_file import VideoStreamWriter
 from kinematics import Kinematics
 from scene_builder import build_scene
 dt=1/240;fps=5 if args.fast else 10;stride=round(1/dt/fps)
 sim=su.SimulationContext(su.SimulationCfg(dt=dt,device=args.device,render=su.RenderCfg(enable_reflections=True,enable_global_illumination=True,rendering_mode='quality')));stage=sim.stage
 sc=build_scene(stage);cube=sc['cube'];lathe=sc['lathe'];xform=sc['xform']
 physics_mat=UsdShade.Material.Define(stage,'/World/Materials/Contact');pm=UsdPhysics.MaterialAPI.Apply(physics_mat.GetPrim());pm.CreateStaticFrictionAttr(1.0);pm.CreateDynamicFrictionAttr(.8);pm.CreateRestitutionAttr(0.)
 def collision(prim,mesh=False,hidden=False):
  UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
  if mesh:UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('none')
  pa=PhysxSchema.PhysxCollisionAPI.Apply(prim);pa.CreateContactOffsetAttr(.001);pa.CreateRestOffsetAttr(0.)
  UsdShade.MaterialBindingAPI.Apply(prim).Bind(physics_mat,materialPurpose='physics')
  if hidden:UsdGeom.Imageable(prim).CreateVisibilityAttr('invisible')
  return prim
 def box(name,pos,size,hidden=True):return collision(cube(name,pos,size,'steel').GetPrim(),hidden=hidden)
 for prim in list(Usd.PrimRange(stage.GetPrimAtPath('/World/Set/WhiteBowl'))):
  if prim.HasAPI(UsdPhysics.RigidBodyAPI):UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
  if prim.HasAPI(UsdPhysics.CollisionAPI):UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
 for n in ['Floor','Worktable','BoilerFront','BoilerBack','BoilerSide0','BoilerSide1','BoilerBottom']:
  collision(stage.GetPrimAtPath('/World/Set/'+n))
 for n in ['NoodlePlate','BoilerDeck0','BoilerDeck1']:
  collision(stage.GetPrimAtPath('/World/Set/'+n),mesh=True)
 # A compound well wall avoids the folded visual lip trapping the basket flange.
 for wi,(cx,cy) in enumerate([(.495,.037),(.495,.273)]):
  for j,a in enumerate(np.arange(32)*2*np.pi/32):
   ob=cube(f'WellProxy{wi}_{j}',(cx+.095*np.cos(a),cy+.095*np.sin(a),.963),(1,1,1),'steel');xf=UsdGeom.Xformable(ob);xf.ClearXformOpOrder();xf.AddTranslateOp().Set(Gf.Vec3d(cx+.095*np.cos(a),cy+.095*np.sin(a),.963));xf.AddRotateZOp().Set(np.rad2deg(a));xf.AddScaleOp().Set(Gf.Vec3f(.004,.019,.170));collision(ob.GetPrim(),hidden=True)
  ob=sc['cyl'](f'WellProxyBottom{wi}',(cx,cy,.879),.095,.008,'steel');collision(ob.GetPrim(),hidden=True)
 lathe('BowlCollision',(-.035,-.025,.766),[(0,0),(.05,0),(.076,.015),(.098,.065),(.094,.070),(.075,.027),(.05,.009),(0,.009)],'porcelain')
 collision(stage.GetPrimAtPath('/World/Set/BowlCollision'),mesh=True,hidden=True)
 def dynamic(path,mass):
  prim=stage.GetPrimAtPath(path);UsdPhysics.RigidBodyAPI.Apply(prim).CreateRigidBodyEnabledAttr(True);UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
  rb=PhysxSchema.PhysxRigidBodyAPI.Apply(prim);rb.CreateSolverPositionIterationCountAttr(16);rb.CreateSolverVelocityIterationCountAttr(4);rb.CreateMaxDepenetrationVelocityAttr(.5)
  return RigidObject(RigidObjectCfg(prim_path=path,spawn=None))
 objects=[]
 for i in range(2):
  box(f'Noodle{i}/Collision',(0,0,0),(.052,.068,.029));objects.append(dynamic(f'/World/Set/Noodle{i}',.045))
 # Put the removable front basket's existing visual geometry into a rigid frame.
 b0=np.array([.495,.037,.928]);basket=UsdGeom.Xform.Define(stage,'/World/Set/MovingBasket');xform(basket,b0)
 edits=Sdf.BatchNamespaceEdit()
 for prim in list(stage.GetPrimAtPath('/World/Set').GetChildren()):
  n=prim.GetName()
  if n.startswith('Basket0_'):
   mesh=UsdGeom.Mesh(prim);mesh.GetPointsAttr().Set([Gf.Vec3f(*v) for v in np.asarray(mesh.GetPointsAttr().Get())-b0]);edits.Add(str(prim.GetPath()),'/World/Set/MovingBasket/'+n)
 assert stage.GetRootLayer().Apply(edits)
 for i,a in enumerate(np.arange(32)*2*np.pi/32):
  ob=cube(f'MovingBasket/Wall{i}',(.082*np.cos(a),.082*np.sin(a),.067),(.007,.017,.134),'steel');UsdGeom.Xformable(ob).AddRotateZOp().Set(np.rad2deg(a));collision(ob.GetPrim(),hidden=True)
 ob=sc['cyl']('MovingBasket/Bottom',(0,0,.0015),.082,.003,'steel');collision(ob.GetPrim(),hidden=True)
 for i,a in enumerate([0,np.pi/2,np.pi,3*np.pi/2]):box(f'MovingBasket/RimSupport{i}',(.095*np.cos(a),.095*np.sin(a),.129),(.012,.012,.007))
 # The photo handle is a flattened grip, represented with broad contact faces.
 ob=cube('MovingBasket/HandleCollision',(.135,0,.269),(1,1,1),'black_handle');xf=UsdGeom.Xformable(ob);xf.ClearXformOpOrder();xf.AddTranslateOp().Set(Gf.Vec3d(.135,0,.269));xf.AddRotateYOp().Set(7.04);xf.AddScaleOp().Set(Gf.Vec3f(.026,.016,.162));collision(ob.GetPrim(),hidden=True)

 objects.append(dynamic('/World/Set/MovingBasket',.16))
 # Estimated food/steel contact coefficients; gripper and handle retain their grip material.
 food_mat=UsdShade.Material.Define(stage,'/World/Materials/FoodContact');fm=UsdPhysics.MaterialAPI.Apply(food_mat.GetPrim());fm.CreateStaticFrictionAttr(.10);fm.CreateDynamicFrictionAttr(.06);fm.CreateRestitutionAttr(0.)
 slide_mat=UsdShade.Material.Define(stage,'/World/Materials/BasketInteriorContact');sm=UsdPhysics.MaterialAPI.Apply(slide_mat.GetPrim());sm.CreateStaticFrictionAttr(.12);sm.CreateDynamicFrictionAttr(.08);sm.CreateRestitutionAttr(0.)
 for i in range(2):UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(f'/World/Set/Noodle{i}/Collision')).Bind(food_mat,materialPurpose='physics')
 for prim in stage.GetPrimAtPath('/World/Set/MovingBasket').GetChildren():
  if prim.GetName().startswith('Wall') or prim.GetName()=='Bottom':UsdShade.MaterialBindingAPI.Apply(prim).Bind(slide_mat,materialPurpose='physics')
 PROFILE=gripper_profile();base=(.05,-.65,0);lift=.75
 cfg=get_robot_config();cfg.prim_path='/World/EX001';cfg.init_state.pos=base;cfg.init_state.rot=(.707106781,0,0,.707106781);cfg.init_state.joint_pos['lift_joint']=lift;cfg.init_state.joint_pos['head_pitch_joint']=.45
 robot=Articulation(cfg)
 # Keep the original imported finger geometry and frictional contact; no grasp joints.
 for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/EX001')):
  if '_gripper_' in str(prim.GetPath()) and prim.HasAPI(UsdPhysics.CollisionAPI):UsdShade.MaterialBindingAPI.Apply(prim).Bind(physics_mat,materialPurpose='physics')
 cameras=[];mounts={}
 if not args.fast:
  arms=[EX001Left({'default_root_pos':[0,0,0],'default_root_rot':[1,0,0,0]}),EX001Right({'default_root_pos':[0,0,0],'default_root_rot':[1,0,0,0]})]
  for side,arm in zip(('left','right'),arms):
   for c in arm.camera:
    name='cam_'+side+'_wrist' if c['name']=='cam_wrist' else c['name'];path='/World/EX001/'+c['link']+'/'+name
    cam=UsdGeom.Camera.Define(stage,path);cam.AddTranslateOp().Set(Gf.Vec3d(*c['pos']));quat=c['ori'];cam.AddOrientOp().Set(Gf.Quatf(quat[0],Gf.Vec3f(*quat[1:])))
    cam.CreateFocalLengthAttr(D435['focal_length']);cam.CreateHorizontalApertureAttr(D435['horizontal_aperture']);cam.CreateVerticalApertureAttr(D435['vertical_aperture']);cam.CreateClippingRangeAttr(Gf.Vec2f(.005,10.))
    product=rep.create.render_product(path,(640,480));ann=rep.AnnotatorRegistry.get_annotator('rgb');ann.attach(product);param=rep.AnnotatorRegistry.get_annotator('camera_params');param.attach(product);cameras.append((name,ann,param,(640,480)));mounts[name]=c
 overview=rep.create.camera(position=(1.22,-1.32,1.93),look_at=(.04,-.01,1.01),focal_length=23,clipping_range=(.01,20));prod=rep.create.render_product(overview,(960,720));ann=rep.AnnotatorRegistry.get_annotator('rgb');ann.attach(prod);param=rep.AnnotatorRegistry.get_annotator('camera_params');param.attach(prod);cameras.append(('overview',ann,param,(960,720)))
 sim.reset();robot.update(dt)
 for ob in objects:ob.update(dt)
 names=list(robot.joint_names);index={n:i for i,n in enumerate(names)};target=robot.data.default_joint_pos.clone();robot.write_joint_state_to_sim(target,torch.zeros_like(target));robot.reset()
 def state():return robot.data.joint_pos[0].cpu().numpy().copy()
 def objstate():return np.stack([o.data.root_state_w[0,:7].cpu().numpy().copy() for o in objects])
 def step_physics(q):
  map_mimic_targets(q,names,PROFILE);robot.set_joint_position_target(q);robot.set_joint_velocity_target(torch.zeros_like(q));robot.write_data_to_sim();sim.step(render=False);robot.update(dt)
  for ob in objects:ob.update(dt)
 for _ in range(240):step_physics(target)
 if args.initial_episode:
  with h5py.File(Path(args.initial_episode)/'episode.hdf5','r') as init:
   initial_q=torch.as_tensor(init['observations/joint_position'][-1],device=args.device,dtype=torch.float32).unsqueeze(0);initial_p=init['observations/object_pose_wxyz'][-1]
  target.copy_(initial_q);robot.write_joint_state_to_sim(target,torch.zeros_like(target));robot.reset()
  for i,ob in enumerate(objects):ob.write_root_pose_to_sim(torch.as_tensor(initial_p[i],device=args.device,dtype=torch.float32).unsqueeze(0));ob.write_root_velocity_to_sim(torch.zeros((1,6),device=args.device))
  for _ in range(120):step_physics(target)
 if args.force_test:
  print('FORCE_TEST_INITIAL',objstate()[2].tolist(),'MASS',objects[2].root_physx_view.get_masses().tolist(),flush=True)
  objects[2].set_external_force_and_torque(torch.tensor([[[0.,0.,20.]]],device=args.device),torch.zeros((1,1,3),device=args.device),is_global=True)
  for _ in range(240):objects[2].write_data_to_sim();step_physics(target)
  print('FORCE_TEST_FINAL',objstate()[2].tolist(),flush=True)
  raise RuntimeError('Diagnostic force test completed; not a demonstration')
 if (args.basket_test or args.pour_only) and not args.initial_episode:
  for i in range(2):
   pos=objects[i].data.root_state_w[:,:7].clone();pos[0,:3]=torch.tensor([.495,.037,.949+i*.032],device=args.device);pos[0,3:]=torch.tensor([1.,0.,0.,0.],device=args.device);objects[i].write_root_pose_to_sim(pos);objects[i].write_root_velocity_to_sim(torch.zeros((1,6),device=args.device))
  for _ in range(240):step_physics(target)
 k=Kinematics(PROFILE['urdf'],base=base,lift=lift,tcp=nominal_tcp(PROFILE));arm_ids=[index[n] for n in k.names];gm=index['right_arm_gripper']
 h5=h5py.File(OUT/'episode.hdf5','w');datasets=[]
 for name,ann,param,(w,h) in cameras:
  datasets.append(h5.create_dataset('observations/images/'+name,shape=(0,h,w,3),maxshape=(None,h,w,3),dtype='u1',chunks=(1,h,w,3),compression='lzf'));writers.append(VideoStreamWriter(str(OUT/(name+'.mp4')),h,w,3,fps))
 for _ in range(16):sim.render()
 step=0;states=[];velocities=[];actions=[];poses=[];image_steps=[];camera_poses=[];phases=[];phase_ids=[];phase_name='initial';checks=[];error=None;start_time=time.monotonic();initial_objects=objstate();replay_done=False;replay_source=None
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
 def capture(final=False):
  sim.render();sim.render();cp=[]
  for i,(name,ann,param,(w,h)) in enumerate(cameras):
   raw=np.asarray(ann.get_data())
   if raw.shape[:2] == (h,w) and raw.ndim >= 3:
    im=raw[...,:3].astype(np.uint8)
   elif args.replay_json:
    # A partial Isaac Sim asset cache can leave RTX render products empty or
    # malformed. Preserve replay state and video shape while recording a black
    # frame; joint/object validation remains fully active.
    im=np.zeros((h,w,3),dtype=np.uint8)
   else:
    raise AssertionError(f'Unexpected camera frame shape: {raw.shape}, expected {(h,w,3)}')
   ds=datasets[i];ds.resize(len(image_steps)+1,axis=0);ds[-1]=im
   if not image_steps:Image.fromarray(im).save(OUT/(name+'_first.png'))
   if final or len(image_steps)%fps==0:Image.fromarray(im).save(OUT/(name+'_last.png'))
   if name=='overview':
    pic=Image.fromarray(im);draw=ImageDraw.Draw(pic);draw.rectangle((0,0,w,58),fill=(20,27,34));draw.text((12,5),'X1 PRO FX001 | Noodle transfer | Rigid noodle bundles',font=font,fill='white');draw.text((12,31),f'{phase_name} | {step*dt:.1f} s',font=font,fill=(210,224,242));im=np.asarray(pic)
   writers[i].append(im)
   try:
    cp.append(np.linalg.inv(np.asarray(param.get_data()['cameraViewTransform']).reshape(4,4).T))
   except np.linalg.LinAlgError:
    if not args.replay_json: raise
    cp.append(np.eye(4,dtype=float))
  image_steps.append(step);camera_poses.append(cp)
 def tick():
  global step
  if step%stride==0:capture()
  map_mimic_targets(target,names,PROFILE);states.append(state());velocities.append(robot.data.joint_vel[0].cpu().numpy().copy());actions.append(target[0].cpu().numpy().copy());poses.append(objstate());phase_ids.append(len(phases)-1)
  step_physics(target);step+=1
  if not np.isfinite(state()).all() or not np.isfinite(objstate()).all():raise RuntimeError('Non-finite physical state')
 def phase(name):
  global phase_name
  phase_name=name;phases.append({'name':name,'start_step':step});print('NOODLE_PHASE',name,step,flush=True)
 def hold(sec):
  for _ in range(round(sec/dt)):tick()
 def ramp(goal,sec):
  start=target.clone()
  for a in np.linspace(0,1,max(2,round(sec/dt))):
   w=a*a*(3-2*a);target.copy_(start+(goal-start)*w);tick()
 Rz=Rotation.from_euler('z',90,degrees=True).as_matrix();down=Rz@Rotation.from_euler('y',90,degrees=True).as_matrix();potrot=Rz@Rotation.from_euler('y',60,degrees=True).as_matrix();hand=Rz@Rotation.from_euler('y',30,degrees=True).as_matrix()
 def tcp():
  k.lift=float(state()[index['lift_joint']]);return k.fk(state()[arm_ids])
 def ik7(pos,rot,seed):
  seed=np.clip(seed,np.r_[k.lower,.48]+1e-7,np.r_[k.upper,.78]-1e-7)
  def residual(x):
   k.lift=x[-1];t=k.fk(x[:6]);return np.r_[t[:3,3]-pos,.15*Rotation.from_matrix(rot@t[:3,:3].T).as_rotvec(),.00003*(x-seed)]
  fit=least_squares(residual,seed,bounds=(np.r_[k.lower,.48]+1e-8,np.r_[k.upper,.78]-1e-8),max_nfev=150,gtol=1e-9,ftol=1e-9,xtol=1e-9)
  err=np.linalg.norm(residual(fit.x)[:6])
  if err>.001:raise RuntimeError(f'Arm/lift IK failed: {err:.5f}')
  return fit.x
 def move(pos,rot,name,cartesian=True,speed=.09,adaptive=False):
  phase(name);pos=np.asarray(pos,dtype=float);q0=state()[arm_ids];start=tcp();k.lift=lift
  if not cartesian:
   q=k.ik(pos,rot,q0,attempts=10);goal=target.clone();goal[0,arm_ids]=torch.tensor(q,device=args.device,dtype=torch.float32);ramp(goal,max(2,np.max(np.abs(q-q0))/.7))
  else:
   distance=np.linalg.norm(pos-start[:3,3]);angle=Rotation.from_matrix(rot@start[:3,:3].T).magnitude();count=max(3,int(distance/.012)+1,int(angle/.06)+1);sl=Slerp([0,1],Rotation.from_matrix([start[:3,:3],rot]));last=np.r_[q0,float(state()[index['lift_joint']])] if adaptive else q0
   for w in np.linspace(0,1,count)[1:]:
    q=ik7((1-w)*start[:3,3]+w*pos,sl(w).as_matrix(),last) if adaptive else k.ik((1-w)*start[:3,3]+w*pos,sl(w).as_matrix(),last,attempts=2)
    if np.max(np.abs(q-last))>.6:raise RuntimeError('IK branch discontinuity')
    goal=target.clone();goal[0,arm_ids]=torch.tensor(q[:6],device=args.device,dtype=torch.float32)
    if adaptive:goal[0,index['lift_joint']]=q[-1]
    ramp(goal,max(.08,distance/(count-1)/speed,angle/(count-1)/.65,np.max(np.abs(q[:6]-last[:6]))/.8,abs(q[-1]-last[-1])/.06 if adaptive else 0));last=q
  hold(.25);err=float(np.linalg.norm(tcp()[:3,3]-pos));phases[-1]['tcp_error_m']=err;print('TCP',name,err,flush=True)
  if err>.018:raise RuntimeError(f'TCP tracking error {name}: {err:.4f} m')
 def grip(value,name):
  phase(name);goal=target.clone();goal[0,gm]=value;ramp(goal,.65);hold(.35)
 def check(name,value,passed):
  checks.append({'name':name,'value':value,'passed':bool(passed)});print('PHYSICAL_CHECK',json.dumps(checks[-1]),flush=True)
  if not passed:raise RuntimeError(f'Physical check failed: {name} = {value}')
 def in_basket(i):
  pp=objstate();rot=Rotation.from_quat(pp[2,[4,5,6,3]]).as_matrix();v=rot.T@(pp[i,:3]-pp[2,:3]);return bool(np.linalg.norm(v[:2])<.078 and .002<v[2]<.14),v
 if args.replay_json:
  replay_source=str(Path(args.replay_json).resolve());replay=json.loads(Path(args.replay_json).read_text());frames=replay.get('data',[])
  if not frames:raise ValueError(f'No data frames in replay JSON: {args.replay_json}')
  side_ids={side:[index[f'{side}_arm_joint{i}'] for i in range(1,7)] for side in ('left','right')}
  replay_fps=float(replay.get('fps',30.));replay_stride=max(1,round(1./dt/replay_fps));replay_done=True
  phase('real_robot_replay');print('REPLAY_SOURCE '+json.dumps({'path':replay_source,'name':replay.get('name'),'frames':len(frames),'fps':replay_fps,'physics_steps_per_frame':replay_stride}),flush=True)
  replay_gripper_values={side:[] for side in ('left','right')}
  # The bridge has two historical gripper encodings: FX001 joint units
  # (small values) and the raw master/follower encoder (roughly 0..3.5,
  # where low is closed and high is open). Calibrate the latter from the
  # trajectory so an intermediate closed value is not clipped to fully open.
  replay_gripper_cal={}
  for side in ('left','right'):
   vals=[]
   for fr in frames:
    raw=fr.get(f'follow_{side}_gripper')
    if raw is None:
     raw=np.asarray(fr.get(f'follow_{side}_joint_position',[]),dtype=float)
     raw=raw[6] if raw.size >= 7 else None
    if raw is not None: vals.append(float(np.asarray(raw).reshape(-1)[0]))
   if vals and max(vals)-min(vals)>.5:
    replay_gripper_cal[side]=(min(vals),max(vals))
  for fi,frame in enumerate(frames):
   q=target.clone()
   for side in ('left','right'):
    values=np.asarray(frame[f'follow_{side}_joint_position'],dtype=np.float32)
    if values.shape[0]<6:raise ValueError(f'{side} follow joint vector has only {values.shape[0]} values')
    q[0,side_ids[side]]=torch.as_tensor(values[:6],device=args.device,dtype=torch.float32)
    # The bridge gripper field is a physical finger coordinate; convert it
    # to this fork's scalar gripper drive and clamp to the selected profile.
    # Prefer the explicit bridge field. Some older recordings only carry
    # the seventh follower joint, which is the same physical finger signal.
    physical=frame.get(f'follow_{side}_gripper')
    if physical is None and values.shape[0] >= 7: physical=values[6]
    if physical is not None:
     physical=float(np.asarray(physical).reshape(-1)[0]);replay_gripper_values[side].append(physical)
     if side in replay_gripper_cal:
      closed,opened=replay_gripper_cal[side]
      motor=(physical-closed)/(opened-closed)*PROFILE['motor_open'] if opened>closed else 0.
     else:
      motor=(physical-PROFILE['offset'])/PROFILE['multiplier']
     q[0,index[f'{side}_arm_gripper']]=float(np.clip(motor,0.,PROFILE['motor_open']))
   if 'head_yaw' in frame:q[0,index['head_yaw_joint']]=float(np.asarray(frame['head_yaw']).reshape(-1)[0])
   if 'head_pitch' in frame:q[0,index['head_pitch_joint']]=float(np.asarray(frame['head_pitch']).reshape(-1)[0])
   target.copy_(q)
   for _ in range(replay_stride):tick()
   if fi==0 or fi==len(frames)-1:print('REPLAY_FRAME '+json.dumps({'frame':fi,'timestamp':frame.get('timestamp'),'joint_state':state().tolist()}),flush=True)
  phase('replay_final');hold(.5);args.scene_only=True
  for side,vals in replay_gripper_values.items():
   if vals: print('REPLAY_GRIPPER',json.dumps({'side':side,'min':min(vals),'max':max(vals),'unique':len(set(vals)),'calibration':replay_gripper_cal.get(side)}),flush=True)
 try:
  phase('noodles_in_basket' if (args.basket_test or args.pour_only) else 'two_noodle_bundles_on_one_plate');hold(.5)
  stage.GetRootLayer().Export(str(OUT/'scene.usda'))
  if not args.scene_only:
   for i in ([] if (args.basket_test or args.pour_only) else range(2)):
    p0=objstate()[i,:3].astype(float)
    move(p0+[0,0,.13],down,f'approach_noodle_{i}',cartesian=i!=0)
    grip(5.,f'open_for_noodle_{i}');move(p0+[0,0,-.010],down,f'descend_to_noodle_{i}')
    grip(0.,f'grasp_noodle_{i}');move(p0+[0,0,.18],down,f'lift_noodle_{i}')
    dz=float(objstate()[i,2]-p0[2]);check(f'noodle_{i}_lift_m',dz,dz>.10)
    offset=tcp()[:3,:3].T@(objstate()[i,:3]-tcp()[:3,3])
    move([.495,.037,1.16],potrot,f'transfer_noodle_{i}_above_cooker')
    slip=float(np.linalg.norm(objstate()[i,:3]-tcp()[:3,3]-tcp()[:3,:3]@offset));check(f'noodle_{i}_retention_error_m',slip,slip<.025)
    grip(5.,f'release_noodle_{i}_into_basket');hold(1.)
    inside,v=in_basket(i);check(f'noodle_{i}_in_basket',v.tolist(),inside)
    move([.48,-.08,1.20],potrot,f'retract_after_noodle_{i}')
   bp=objstate()[2,:3].astype(float);handle=bp+Rotation.from_quat(objstate()[2,[4,5,6,3]]).as_matrix()@np.array([.135,0,.269])+hand[:,0]*.022
   move(handle-hand[:,0]*.105,hand,'approach_basket_handle',cartesian=not (args.basket_test or args.pour_only))
   move(handle,hand,'align_basket_handle');grip(0.,'grasp_basket_handle')
   move(handle+[0,0,.18],hand,'extract_basket_from_cooker',speed=.055)
   dz=float(objstate()[2,2]-bp[2]);check('basket_lift_m',dz,dz>.14)
   for i in range(2):inside,v=in_basket(i);check(f'noodle_{i}_retained_in_lifted_basket',v.tolist(),inside)
   # Rotate the held basket about its mouth; gravity releases the two bundles.
   held=tcp();br=Rotation.from_quat(objstate()[2,[4,5,6,3]]).as_matrix();basket_grasp_offset=held[:3,:3].T@(objstate()[2,:3]-held[:3,3]);basket_grasp_rotation=held[:3,:3].T@br;basket_mouth=objstate()[2,:3]+br@np.array([0,0,.133]);hand_offset=held[:3,3]-basket_mouth
   move(np.array([.22,-.025,1.25])+hand_offset,hand,'clear_cooker_at_high_position',speed=.06)
   pour_center=np.array([.04,-.012,1.15]);move(pour_center+hand_offset,hand,'move_basket_above_bowl',speed=.04)
   check('basket_transfer_retention_m',float(np.linalg.norm(objstate()[2,:3]-(tcp()[:3,3]+tcp()[:3,:3]@basket_grasp_offset))),np.linalg.norm(objstate()[2,:3]-(tcp()[:3,3]+tcp()[:3,:3]@basket_grasp_offset))<.03)
   for angle in [10,20,30,40,50,60,70,80,85,90,94,98]:
    r=Rotation.from_euler('z',10,degrees=True).as_matrix()@Rotation.from_euler('y',-angle,degrees=True).as_matrix()@br.T;move(pour_center+r@hand_offset,r@held[:3,:3],f'pour_basket_{angle}_degrees',speed=.045,adaptive=True)
   hold(5.)
   actual_basket=objstate()[2];actual_tcp=tcp();basket_slip=float(np.linalg.norm(actual_basket[:3]-(actual_tcp[:3,3]+actual_tcp[:3,:3]@basket_grasp_offset)));check('basket_retained_after_pour_m',basket_slip,basket_slip<.04)
   for i in range(2):
    op=objstate()[i];v=op[:3]-np.array([-.035,-.025,.766]);rot=Rotation.from_quat(op[[4,5,6,3]]).as_matrix();corners=np.array([[a,b,c] for a in [-.026,.026] for b in [-.034,.034] for c in [-.0145,.0145]])@rot.T+v;inside=bool(np.max(np.linalg.norm(corners[:,:2],axis=1))<.102 and np.min(corners[:,2])>.004 and np.max(corners[:,2])<.105);check(f'noodle_{i}_in_bowl',v.tolist(),inside)
  phase('final');hold(.5)
 except Exception as e:
  error=str(e);traceback.print_exc();phase('stopped_after_error');hold(.5)
 states.append(state());velocities.append(robot.data.joint_vel[0].cpu().numpy().copy());poses.append(objstate());capture(final=True)
 success=bool(error is None and not replay_done and not args.scene_only and not args.basket_test and len(checks)>=(7 if args.pour_only else 10) and all(c['passed'] for c in checks))
 for key,data in [('observations/joint_position',states),('observations/joint_velocity',velocities),('observations/object_pose_wxyz',poses),('actions/joint_position',actions),('image_step',image_steps),('phase_id',phase_ids),('timestamp',np.arange(len(states))*dt),('observations/camera_to_world',camera_poses)]:h5.create_dataset(key,data=np.asarray(data))
 conditions={'fixed_base':True,'base_position':base,'initial_lift_height_m':lift,'lift_control':'fixed during food transfer, IK coordinated during pouring','robot_gravity_compensation':True,'self_collision':False,'noodle_model':'two independent rigid bundles, not deformable noodles','basket_collision':'compound shell, bottom, handle and rim supports','well_collision':'compound open wall and bottom; decorative folded lip excluded','bowl_collision':'static concave proxy matching visual bowl','friction_static':1.,'friction_dynamic':.8,'food_friction_static_dynamic':[.10,.06],'basket_interior_friction_static_dynamic':[.12,.08],'contact_parameters':'estimated, not measured on the real food or hardware','objects_gravity_and_contact':True,'object_teleports_during_episode':0,'object_attachment_joints':0,'robot_state_writes_during_episode':0,'cooking_and_water_simulated':False,'scene_scale':'estimated from photo, not calibrated'}
 h5.attrs.update(task='pour_two_noodles_into_bowl' if args.pour_only else 'photo_noodle_transfer',robot='X1 Pro DVT2/PVT1',gripper_type=PROFILE['name'],expert_demonstration=success,success=success,physics_dt=dt,image_fps=fps,joint_names_json=json.dumps(names),object_names_json=json.dumps(['noodle_0','noodle_1','front_basket']),camera_names_json=json.dumps([c[0] for c in cameras]),camera_mounts_json=json.dumps(mounts),camera_intrinsics_json=json.dumps(D435),camera_pose_convention='USD camera -Z forward +Y up; matrices from renderer at image_step',action_semantics='23 absolute joint position drive targets; radians or metres according to URDF; state[t] precedes action[t], includes terminal state',conditions_json=json.dumps(conditions),phases_json=json.dumps(phases),error=error or '')
 h5.close();h5=None
 for w in writers:w.close()
 writers=[]
 summary={'task':'real_robot_replay' if replay_done else ('pour_two_noodles_into_bowl' if args.pour_only else 'photo_noodle_transfer'),'status':'replay_complete' if replay_done and error is None else ('success' if success else ('scene_preview' if args.scene_only else 'failed_attempt')),'expert_demonstration':success,'replay_source':replay_source,'error':error,'gripper_type':PROFILE['name'],'camera_mount':{'head_source':'mounted_urdf_rgb_optical_frame'},'camera_names':[c[0] for c in cameras],'diagnostic_basket_only':args.basket_test,'simulation_seconds':step*dt,'physics_steps':step,'frames_per_camera':len(image_steps),'fps':fps,'checks':checks,'conditions':conditions,'phases':phases,'initial_object_poses':initial_objects.tolist(),'final_object_poses':objstate().tolist(),'wall_seconds':time.monotonic()-start_time,'data_file':str(OUT/'episode.hdf5'),'video':str(OUT/'overview.mp4')}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print('NOODLE_DONE',json.dumps(summary),flush=True);code=0 if success or args.scene_only else 2
except BaseException:
 (OUT/'failure.txt').write_text(traceback.format_exc());traceback.print_exc()
finally:
 if h5 is not None:h5.close()
 for w in writers:w.close()
 import threading
 timer=threading.Timer(15,lambda:os._exit(code));timer.daemon=True;timer.start();app.close();timer.cancel();sys.exit(code)
