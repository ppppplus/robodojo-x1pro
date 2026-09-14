"""Photo-workstation chili shaker demo; scripted container, free PhysX grains."""
import argparse,json,os,sys,traceback,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT),str(HERE.parent/'noodle_expert')]
os.environ['ROBODOJO_EX001_GRIPPER']='fx001_h_evt1'
os.environ.pop('ROBODOJO_EX001_USD',None)
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--output',required=True);AppLauncher.add_app_launcher_args(p);args=p.parse_args()
OUT=Path(args.output);OUT.mkdir(parents=True,exist_ok=True)
if (OUT/'summary.json').exists():raise FileExistsError(OUT)
for src in [Path(__file__),HERE.parent/'noodle_expert/scene_builder.py',HERE.parent/'noodle_expert/seasoning.py']:(OUT/src.name).write_text(src.read_text())
app=AppLauncher(args).app;code=1;writers=[]
try:
 import numpy as np,torch,carb
 from PIL import Image,ImageDraw,ImageFont
 from pxr import Usd,UsdGeom,UsdShade,UsdPhysics,PhysxSchema,Gf,Sdf,Vt
 import isaaclab.sim as su
 from isaaclab.assets import Articulation
 import omni.replicator.core as rep
 from omni.physx.scripts import particleUtils,physicsUtils
 from omni.physx import get_physx_interface
 from utils.save_file import VideoStreamWriter
 from env.robot_manager.robot_config.ex001 import get_robot_config
 from env.robot_manager.ex001_variants import gripper_profile,map_mimic_targets
 from scene_builder import build_scene
 from seasoning import grain_positions,RADIUS
 sim=su.SimulationContext(su.SimulationCfg(dt=1/240,device=args.device,use_fabric=False,render=su.RenderCfg(enable_reflections=True,enable_global_illumination=True,rendering_mode='quality')));stage=sim.stage
 sc=build_scene(stage);bind=sc['bind'];material=sc['material']
 get_physx_interface().overwrite_gpu_setting(1)
 def readback():
  for k,v in [('/physics/suppressReadback',False),('/physics/updateToUsd',True),('/physics/updateParticlesToUsd',True)]:carb.settings.get_settings().set_bool(k,v)
 readback()
 physics_scene=next(p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene))
 api=PhysxSchema.PhysxSceneAPI.Apply(physics_scene);api.CreateEnableGPUDynamicsAttr(True);api.CreateBroadphaseTypeAttr('GPU')
 def collision(prim):
  UsdPhysics.CollisionAPI.Apply(prim)
  if prim.IsA(UsdGeom.Mesh):UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('none')
  px=PhysxSchema.PhysxCollisionAPI.Apply(prim);px.CreateContactOffsetAttr(.0003);px.CreateRestOffsetAttr(0.)
 for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/Set/WhiteBowl')):
  if prim.HasAPI(UsdPhysics.RigidBodyAPI):UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
  if prim.HasAPI(UsdPhysics.CollisionAPI):UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
 for n in ['Worktable','Floor']:collision(stage.GetPrimAtPath('/World/Set/'+n))
 sc['lathe']('BowlCollision',(-.035,-.025,.766),[(0,0),(.05,0),(.076,.015),(.098,.065),(.094,.070),(.075,.027),(.05,.009),(0,.009)],'porcelain')
 bowlproxy=stage.GetPrimAtPath('/World/Set/BowlCollision');collision(bowlproxy);UsdGeom.Imageable(bowlproxy).MakeInvisible()
 # Keep the recognizable pink shaker, including its real perforated white cap.
 start=np.array([-.53,.18,.768]);root=UsdGeom.Xform.Define(stage,'/World/Set/ChiliShaker');tr=root.AddTranslateOp();tr.Set(Gf.Vec3d(*start));rot=root.AddOrientOp();rot.Set(Gf.Quatf(1,Gf.Vec3f(0)))
 body=UsdPhysics.RigidBodyAPI.Apply(root.GetPrim());body.CreateKinematicEnabledAttr(True);UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(.08)
 stage.RemovePrim('/World/Set/Shaker0_Contents')
 edits=Sdf.BatchNamespaceEdit()
 for prim in list(stage.GetPrimAtPath('/World/Set').GetChildren()):
  n=prim.GetName()
  if not n.startswith('Shaker0_'):continue
  if prim.IsA(UsdGeom.Mesh):
   mesh=UsdGeom.Mesh(prim);points=np.asarray(mesh.GetPointsAttr().Get())-start;mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
  else:
   attr=prim.GetAttribute('xformOp:translate')
   if attr:attr.Set(Gf.Vec3d(*(np.asarray(attr.Get())-start)))
  edits.Add(str(prim.GetPath()),str(root.GetPath())+'/'+n)
 assert stage.GetRootLayer().Apply(edits)
 for n in ['Shaker0_Body','Shaker0_Neck','Shaker0_ShakerTop']:collision(stage.GetPrimAtPath(str(root.GetPath())+'/'+n))
 initial=grain_positions(start);N=len(initial);system_path=Sdf.Path('/World/ChiliParticleSystem')
 system=particleUtils.add_physx_particle_system(stage,system_path,simulation_owner=physics_scene.GetPath(),contact_offset=.00135,rest_offset=RADIUS,particle_contact_offset=.00135,solid_rest_offset=RADIUS,fluid_rest_offset=.00065,enable_ccd=True,solver_position_iterations=16,max_velocity=3.,global_self_collision_enabled=True,non_particle_collision_enabled=True)
 matpath='/World/Materials/ChiliPhysics';particleUtils.add_pbd_particle_material(stage,matpath,friction=.35,particle_friction_scale=1.,damping=.05,cohesion=0.,adhesion=0.,gravity_scale=1.);physicsUtils.add_physics_material_to_prim(stage,system.GetPrim(),Sdf.Path(matpath))
 pp=Sdf.Path('/World/ChiliParticles');particleUtils.add_physx_particleset_pointinstancer(stage,pp,Vt.Vec3fArray.FromNumpy(initial),Vt.Vec3fArray.FromNumpy(np.zeros_like(initial)),system_path,True,False,0,.000003,0.,num_prototypes=3,prototype_indices=[i%3 for i in range(N)])
 inst=UsdGeom.PointInstancer.Get(stage,pp)
 for i in range(3):
  ob=UsdGeom.Sphere.Get(stage,pp.AppendChild('particlePrototype'+str(i)));ob.CreateRadiusAttr(RADIUS);UsdShade.MaterialBindingAPI.Apply(ob.GetPrim()).Bind(UsdShade.Material.Get(stage,f'/World/Materials/chili_grain{i}'))
 cfg=get_robot_config();cfg.prim_path='/World/EX001';cfg.init_state.pos=(.05,-.65,0);cfg.init_state.rot=(.707106781,0,0,.707106781);cfg.init_state.joint_pos['lift_joint']=.75;cfg.init_state.joint_pos['head_pitch_joint']=.45
 robot=Articulation(cfg);sim.reset();robot.update(sim.get_physics_dt());q=robot.data.default_joint_pos.clone();map_mimic_targets(q,list(robot.joint_names),gripper_profile());robot.write_joint_state_to_sim(q,torch.zeros_like(q));readback()
 fps=20;dt=sim.get_physics_dt();duration=16.;stride=round(1/fps/dt);steps=round(duration/dt)
 streams=[]
 for label,pos,look,size,focal in [('closeup',(.40,-.80,1.57),(-.09,-.025,1.015),(960,720),34.),('overview',(.32,-.90,1.77),(-.045,.035,.84),(960,720),24.)]:
  cam=rep.create.camera(position=pos,look_at=look,focal_length=focal,clipping_range=(.005,20));prod=rep.create.render_product(cam,size);ann=rep.AnnotatorRegistry.get_annotator('rgb');ann.attach(prod);streams.append((label,ann,size));writers.append(VideoStreamWriter(str(OUT/(label+'.mp4')),size[1],size[0],3,fps))
 for _ in range(20):sim.render()
 stage.GetRootLayer().Export(str(OUT/'scene_initial.usda'))
 samples=[];angles=[];times=[];counts=[];container_positions=[];started=time.monotonic();font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
 def particles():return np.asarray(inst.GetPositionsAttr().Get(),dtype=np.float32)
 def in_bowl(p):
  h=p[:,2]-.766;r=np.linalg.norm(p[:,:2]-[-.035,-.025],axis=1);inner=np.interp(h,[.009,.027,.070],[.05,.075,.094]);return (h>=.007)&(h<.075)&(r<inner)
 def smooth(u):return np.clip(u,0,1)**2*(3-2*np.clip(u,0,1))
 mouth=np.array([-.035,-.025,1.035]);upright=mouth-[0,0,.127];high=start+[0,0,.25]
 def pose(t):
  angle=0.
  if t<1:pos=start
  elif t<2:pos=start+(high-start)*smooth(t-1)
  elif t<4:pos=high+(upright-high)*smooth((t-2)/2)
  elif t<6.5:angle=165*smooth((t-4)/2.5);pos=None
  elif t<11:angle=165+7*np.sin((t-6.5)*2*np.pi*1.5);pos=None
  elif t<13:angle=165*(1-smooth((t-11)/2));pos=None
  elif t<15:pos=upright+(start-upright)*smooth((t-13)/2)
  else:pos=start
  if pos is None:
   a=np.deg2rad(angle);pos=mouth-np.array([.127*np.sin(a),0,.127*np.cos(a)])
  return np.asarray(pos),angle
 def capture(t):
  sim.render();sim.render();pts=particles();assert pts.shape==(N,3) and np.isfinite(pts).all()
  pos,angle=pose(t);samples.append(pts.copy());angles.append(angle);times.append(t);counts.append(int(in_bowl(pts).sum()));container_positions.append(pos)
  phase='SETTLE IN SHAKER' if t<1 else ('MOVE SHAKER' if t<4 else ('TILT / SHAKE CHILI THROUGH CAP' if t<11 else ('RETURN SHAKER' if t<15 else 'SETTLE IN BOWL')))
  for (label,ann,size),writer in zip(streams,writers):
   raw=Image.fromarray(np.asarray(ann.get_data())[...,:3].astype(np.uint8));im=raw.copy();d=ImageDraw.Draw(im);d.rectangle((0,0,size[0],56),fill=(18,25,35));d.text((12,6),f'CHILI | PhysX granular particles | {t:.2f}s | In bowl: {counts[-1]} / {N}',font=font,fill='white');d.text((12,31),phase+' | Scripted shaker motion',font=font,fill=(240,183,142));writer.append(np.asarray(im))
   if t==0:raw.save(OUT/(label+'_first.png'))
   if abs(t-8)<dt:raw.save(OUT/(label+'_pouring.png'))
   if abs(t-duration)<dt:raw.save(OUT/(label+'_last.png'))
 print('CHILI_READY',N,flush=True)
 for step in range(steps+1):
  t=step*dt
  if step%stride==0:capture(t)
  if step==steps:break
  pos,angle=pose(t+dt);a=np.deg2rad(angle)/2;tr.Set(Gf.Vec3d(*pos));rot.Set(Gf.Quatf(float(np.cos(a)),Gf.Vec3f(0,float(np.sin(a)),0)))
  robot.set_joint_position_target(q);robot.write_data_to_sim();sim.step(render=False);robot.update(dt)
  if (step+1)%240==0:print('CHILI_SECOND',(step+1)*dt,'IN_BOWL',int(in_bowl(particles()).sum()),flush=True)
 for w in writers:w.close()
 writers=[];sample=np.stack(samples);final=particles();pos,angle=pose(duration)
 local=final-pos;in_jar=(np.linalg.norm(local[:,:2],axis=1)<.030)&(local[:,2]>.004)&(local[:,2]<.134)
 on_table=(np.abs(final[:,0])<.70)&(np.abs(final[:,1])<.45)&(final[:,2]>.75)&(final[:,2]<.78)&(~in_bowl(final))&(~in_jar)
 validation={'finite_positions':bool(np.isfinite(sample).all()),'constant_particle_count':all(len(x)==N for x in samples),'initially_in_jar':bool((np.linalg.norm(initial[:,:2]-start[:2],axis=1)<.03).all()),'initial_bowl_empty':counts[0]==0,'chili_reached_bowl':counts[-1]>100,'free_particle_motion':float(np.linalg.norm(sample[-1]-sample[0],axis=1).max())>.2}
 np.savez_compressed(OUT/'particle_states.npz',positions=sample,timestamp=times,container_position=container_positions,container_angle_degrees=angles,in_bowl_count=counts)
 stage.GetRootLayer().Export(str(OUT/'scene_final.usda'))
 result={'status':'success' if all(validation.values()) else 'failed','particle_count':N,'particle_radius_m':RADIUS,'initial_fill_height_m':.039,'duration_seconds':duration,'fps':fps,'frames':len(times),'final_in_bowl':counts[-1],'final_in_jar':int(in_jar.sum()),'final_on_table':int(on_table.sum()),'final_elsewhere':int(N-counts[-1]-in_jar.sum()-on_table.sum()),'validation':validation,'conditions':{'robot':'idle; not grasping the shaker','shaker':'prescribed kinematic transport, tilt and shaking','chili':'PhysX GPU PBD solid spheres; coarse approximation, not calibrated chili powder','particle_position_writes_after_initialization':0,'cap':'seven real 6.4 mm apertures','salt':'black jar contains static white particle visuals','expert_robot_trajectory':False},'wall_seconds':time.monotonic()-started}
 (OUT/'summary.json').write_text(json.dumps(result,indent=2));print('CHILI_DONE',json.dumps(result),flush=True);code=0 if result['status']=='success' else 2
except BaseException:
 (OUT/'failure.txt').write_text(traceback.format_exc());traceback.print_exc()
finally:
 import threading
 for w in writers:
  try:w.close()
  except Exception:pass
 timer=threading.Timer(15,lambda:os._exit(code));timer.daemon=True;timer.start();app.close();timer.cancel();sys.exit(code)
