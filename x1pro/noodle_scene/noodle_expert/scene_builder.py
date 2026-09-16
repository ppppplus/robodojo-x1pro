"""Photo-based noodle workstation; geometry is in metres, scale estimated from the photo."""
from pathlib import Path
import numpy as np
from pxr import Usd,UsdGeom,UsdShade,UsdLux,Gf,Sdf,UsdPhysics
import isaaclab.sim as su
ROOT=Path(__file__).resolve().parents[3]
def build_scene(stage):
    mats={}
    def material(name,color,metal=0.,rough=.4):
     m=UsdShade.Material.Define(stage,'/World/Materials/'+name);s=UsdShade.Shader.Define(stage,str(m.GetPath())+'/Surface');s.CreateIdAttr('UsdPreviewSurface');s.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color));s.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(metal);s.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(rough);m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(),'surface');mats[name]=m;return name
    for a in [('steel',(.72,.75,.76),.65,.25),('darksteel',(.15,.18,.19),.72,.35),('glass',(.025,.035,.042),.25,.16),('ivory',(.85,.87,.82),0,.4),('wood',(.40,.24,.115),0,.65),('floor',(.24,.28,.28),0,.8),('wall',(.67,.73,.70),0,.85),('tile',(.81,.84,.80),0,.6),('noodle',(.86,.64,.29),0,.66),('water',(.12,.30,.31),.25,.13),('teal',(.06,.24,.22),0,.24),('red',(.82,.10,.035),.0,.3),('green',(.14,.35,.06),0,.65)]:material(*a)
    def bind(obj,mat):UsdShade.MaterialBindingAPI.Apply(obj.GetPrim()).Bind(mats[mat])
    def xform(obj,pos=(0,0,0),scale=None):
     xf=UsdGeom.Xformable(obj);xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
     if scale is not None:xf.AddScaleOp().Set(Gf.Vec3f(*scale))
    def cube(name,pos,size,mat):
     ob=UsdGeom.Cube.Define(stage,'/World/Set/'+name);ob.CreateSizeAttr(1);xform(ob,pos,size);bind(ob,mat);return ob
    def cyl(name,pos,radius,height,mat):
     ob=UsdGeom.Cylinder.Define(stage,'/World/Set/'+name);ob.CreateRadiusAttr(radius);ob.CreateHeightAttr(height);ob.CreateAxisAttr('Z');xform(ob,pos);bind(ob,mat);return ob
    def tube(name,pts,radius,mat):
     pts=np.asarray(pts,dtype=float);vertices=[];normals=[];n=8
     for i,p in enumerate(pts):
      t=pts[min(i+1,len(pts)-1)]-pts[max(i-1,0)];t/=np.linalg.norm(t);a=np.cross(t,[0,0,1])
      if np.linalg.norm(a)<.01:a=np.cross(t,[0,1,0])
      a/=np.linalg.norm(a);b=np.cross(t,a)
      for j in range(n):
       norm=np.cos(j*2*np.pi/n)*a+np.sin(j*2*np.pi/n)*b;vertices.append(p+radius*norm);normals.append(norm)
     faces=[]
     for i in range(len(pts)-1):
      for j in range(n):faces.extend([i*n+j,i*n+(j+1)%n,(i+1)*n+(j+1)%n,(i+1)*n+j])
     ob=UsdGeom.Mesh.Define(stage,'/World/Set/'+name);ob.CreatePointsAttr([Gf.Vec3f(*p) for p in vertices]);ob.CreateFaceVertexCountsAttr([4]*((len(pts)-1)*n));ob.CreateFaceVertexIndicesAttr(faces);ob.CreateNormalsAttr([Gf.Vec3f(*v) for v in normals]);ob.SetNormalsInterpolation('vertex');ob.CreateSubdivisionSchemeAttr('none');bind(ob,mat);return ob
    def ring(name,center,radius,t,mat):
     ang=np.linspace(0,2*np.pi,97);tube(name,np.array(center)+np.c_[radius*np.cos(ang),radius*np.sin(ang),np.zeros_like(ang)],t,mat)
    def lathe(name,center,profile,mat):
     n=96;pts=[]
     for r,z in profile:
      pts.extend([(center[0]+r*np.cos(a),center[1]+r*np.sin(a),center[2]+z) for a in np.arange(n)*2*np.pi/n])
     faces=[]
     for i in range(len(profile)-1):
      for j in range(n):faces.extend([i*n+j,i*n+(j+1)%n,(i+1)*n+(j+1)%n,(i+1)*n+j])
     ob=UsdGeom.Mesh.Define(stage,'/World/Set/'+name);ob.CreatePointsAttr(pts);ob.CreateFaceVertexCountsAttr([4]*((len(profile)-1)*n));ob.CreateFaceVertexIndicesAttr(faces);ob.CreateSubdivisionSchemeAttr('catmullClark');ob.CreateDoubleSidedAttr(True);bind(ob,mat)
    # User-measured workstation: white 1.20 m x 0.60 m table, 0.75 m top height.
    material('porcelain',(.95,.945,.92),0,.17);material('table_white',(.94,.94,.92),0,.32);material('pink_powder',(.66,.24,.31),0,.48);material('bottle_gray',(.29,.41,.40),0,.40);material('yellow',(.80,.77,.025),0,.35);material('black_handle',(.012,.018,.017),0,.64);material('frame',(.60,.63,.64),.45,.36);material('meshwire',(.22,.25,.23),.90,.28);material('raw_noodle',(.81,.77,.51),0,.55)
    # Wood surface reuses the installed RoboDojo texture, with UVs along the table.
    m=UsdShade.Material.Define(stage,'/World/Materials/TableWood');sh=UsdShade.Shader.Define(stage,str(m.GetPath())+'/Surface');sh.CreateIdAttr('UsdPreviewSurface');sh.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.50)
    tex=UsdShade.Shader.Define(stage,str(m.GetPath())+'/Texture');tex.CreateIdAttr('UsdUVTexture');tex.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(str(ROOT/'Assets/Material/material_0114/Bamboo_Planks_BaseColor.png'));tex.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('sRGB');tex.CreateInput('scale',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(1,1,1,1));tex.CreateInput('wrapS',Sdf.ValueTypeNames.Token).Set('repeat');tex.CreateInput('wrapT',Sdf.ValueTypeNames.Token).Set('repeat');tex.CreateOutput('rgb',Sdf.ValueTypeNames.Float3)
    uv=UsdShade.Shader.Define(stage,str(m.GetPath())+'/UV');uv.CreateIdAttr('UsdPrimvarReader_float2');uv.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st');tex.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(),'result');sh.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(),'rgb');m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(),'surface');mats['TableWood']=m
    cube('Floor',(0,0,-.03),(6,6,.06),'wall');cube('RearPanel',(0,.57,1.20),(2.6,.032,2.4),'ivory');cube('LeftPanel',(-.86,-.2,1.20),(.032,1.5,2.4),'ivory')
    for i,x in enumerate([-.84,.19,.86]):cube(f'Extrusion{i}',(x,.547,1.18),(.027,.032,2.36),'frame')
    cube('LeftFrame',(-.84,-.35,1.18),(.028,.03,2.36),'frame')
    cube('Worktable',(0,0,.73),(1.20,.60,.04),'table_white')
    top=UsdGeom.Mesh.Define(stage,'/World/Set/WhiteTableSurface');top.CreatePointsAttr([(-.60,-.30,.75),(.60,-.30,.75),(.60,.30,.75),(-.60,.30,.75)]);top.CreateFaceVertexCountsAttr([4]);top.CreateFaceVertexIndicesAttr([0,1,2,3]);top.CreateSubdivisionSchemeAttr('none');bind(top,'table_white')
    for i,x in enumerate([-.56,.56]):
     for j,y in enumerate([-.26,.26]):cube(f'TableLeg{i}_{j}',(x,y,.355),(.045,.045,.71),'table_white')
    # Import a real RoboDojo bowl and override its glaze to match the photo.
    asset=ROOT/'Assets/Object/RoboDojo/Rigid/bowl/00014/object.usdz'
    bowl=UsdGeom.Xform.Define(stage,'/World/Set/WhiteBowl');xform(bowl,(-.035,-.025,.788),(.20/.16,)*3)
    model=stage.DefinePrim('/World/Set/WhiteBowl/Model','Xform');model.GetReferences().AddReference(str(asset));UsdShade.MaterialBindingAPI.Apply(bowl.GetPrim()).Bind(mats['porcelain'],bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    # White plate and a nest of long uncooked noodles, not the former instant noodle cake.
    # Front-right plate: its 24 cm diameter now has a 1.5 cm front margin on the measured table.
    pc=(.16,-.165,.751)
    lathe('NoodlePlate',pc,[(0,0),(.073,0),(.105,.009),(.118,.015),(.12,.018),(.116,.020),(.090,.008),(0,.006)],'porcelain')
    # Two independently movable noodle bundles share one plate.
    for block in range(2):
     center=np.array([pc[0]+(block-.5)*.09,pc[1],.781])
     root=UsdGeom.Xform.Define(stage,f'/World/Set/Noodle{block}');xform(root,center)
     # A coherent bundle of long strands, approximated as one rigid food item.
     for layer in range(7):
      for strand in range(8):
       t=np.linspace(0,1,100);xx=(strand-3.5)*.0068+.0013*np.sin(t*8*np.pi+layer);yy=(t-.5)*.066;zz=np.full_like(t,(layer-3)*.0042)+.0006*np.sin(t*6*np.pi+strand)
       tube(f'Noodle{block}/Strand{layer}_{strand}',np.c_[xx,yy,zz],.0017,'raw_noodle')
    # Hollow flip-top shakers with actual apertures and static granular contents.
    # The chili_pour demo activates PhysX particle dynamics for the chili contents.
    from seasoning import add_contents,perforated_top
    material('salt_black',(.012,.012,.012),0,.48)
    bottle_info=[]
    # Match the head-camera layout: pink, green squeeze bottle, then black salt toward the bowl.
    for i,(x,mat) in enumerate([(-.50,'pink_powder'),(-.15,'salt_black')]):
     y=.18;z=.752;r=.032
     lathe(f'Shaker{i}_Body',(x,y,z),[(0,0),(.025,0),(.031,.006),(.032,.090),(.027,.111),(.024,.117),(.021,.117),(.024,.108),(.029,.088),(.028,.007),(0,.007)],mat)
     lathe(f'Shaker{i}_Neck',(x,y,z+.108),[(.021,0),(.025,0),(.025,.015),(.021,.015),(.021,0)],'porcelain')
     perforated_top(stage,f'/World/Set/Shaker{i}_ShakerTop',(x,y,z+.127),mats['porcelain'])
     lid=cyl(f'Shaker{i}_OpenLid',(x-.049,y+.008,z+.163),.029,.004,'porcelain');UsdGeom.Xformable(lid).AddRotateYOp().Set(65.)
     tube(f'Shaker{i}_Hinge',[(x-.024,y,z+.128),(x-.035,y,z+.139),(x-.046,y,z+.145)],.004,'porcelain')
     kind='chili' if i==0 else 'salt'
     contents=add_contents(stage,f'/World/Set/Shaker{i}_Contents',(x,y,z),kind)
     bottle_info.append({'type':'flip-top shaker','position':[x,y,z],'body_color':mat,'contents':kind,'particle_count':len(contents.GetPositionsAttr().Get()),'contents_physics':'static preview; enabled in chili_pour demo'})
    # Rounded rectangular squeeze bottle with yellow spout and tethered plug.
    bx,by,bz=-.33,.18,.752;verts=[];faces=[];cross=[]
    for cx,cy,start in [(.022,.022,0),(-.022,.022,90),(-.022,-.022,180),(.022,-.022,270)]:
     for angle in np.linspace(start,start+90,9,endpoint=False):cross.append((cx+.013*np.cos(np.deg2rad(angle)),cy+.013*np.sin(np.deg2rad(angle))))
    for scale,z in [(.82,0),(1,.007),(1,.105),(.76,.126),(.54,.131)]:verts.extend([(bx+x*scale,by+y*scale,bz+z) for x,y in cross])
    n=len(cross)
    for k in range(4):
     for j in range(n):faces.extend([k*n+j,k*n+(j+1)%n,(k+1)*n+(j+1)%n,(k+1)*n+j])
    ob=UsdGeom.Mesh.Define(stage,'/World/Set/SqueezeBottleBody');ob.CreatePointsAttr(verts);ob.CreateFaceVertexCountsAttr([4]*(4*n));ob.CreateFaceVertexIndicesAttr(faces);ob.CreateSubdivisionSchemeAttr('catmullClark');bind(ob,'bottle_gray')
    cyl('SqueezeYellowCap',(bx,by,bz+.132),.029,.021,'yellow')
    lathe('SqueezeSpout',(bx,by,bz+.141),[(.010,0),(.009,.006),(.005,.035),(.003,.038),(0,.038)],'yellow')
    tube('SqueezeTether',[(bx-.02,by,bz+.134),(bx-.053,by,bz+.151),(bx-.063,by,bz+.19)],.002,'yellow');cyl('SqueezePlug',(bx-.063,by,bz+.196),.007,.016,'yellow')
    # Stainless two-basket noodle boiler at the right rear, 30 cm wide by 50 cm deep.
    fx,fy=.42,.05;base=.752;W=.30;D=.50;H=.285;topz=base+H
    cube('BoilerFront',(fx,fy-D/2,base+H/2),(W,.009,H),'steel');cube('BoilerBack',(fx,fy+D/2,base+H/2),(W,.009,H),'steel')
    for i,x in enumerate([fx-W/2,fx+W/2]):cube('BoilerSide'+str(i),(x,fy,base+H/2),(.008,D,H),'steel')
    cube('BoilerBottom',(fx,fy,base+.009),(W,D,.018),'steel')
    # Top deck is a real mesh with round holes; no solid slab closes off the baskets.
    holes=[(fx,fy-.118),(fx,fy+.118)];holeR=.095
    for k,(cx,cy) in enumerate(holes):
     points=[];faces=[];n=96
     for j in range(n):
      a=j*2*np.pi/n;c,s=np.cos(a),np.sin(a);extent=min((W/2)/max(abs(c),1e-7),(D/4)/max(abs(s),1e-7));points.extend([(cx+holeR*c,cy+holeR*s,topz),(cx+extent*c,cy+extent*s,topz)])
     for j in range(n):faces.extend([2*j,2*j+1,2*((j+1)%n)+1,2*((j+1)%n)])
     ob=UsdGeom.Mesh.Define(stage,f'/World/Set/BoilerDeck{k}');ob.CreatePointsAttr(points);ob.CreateFaceVertexCountsAttr([4]*n);ob.CreateFaceVertexIndicesAttr(faces);ob.CreateSubdivisionSchemeAttr('none');ob.CreateDoubleSidedAttr(True);bind(ob,'steel')
     # Smooth circular well and independently modeled wire basket.
     lathe(f'Well{k}',(cx,cy,topz-.18),[(0,0),(.091,0),(.096,.01),(.096,.179),(.094,.181),(.091,.177),(.087,.01),(0,.006)],'darksteel')
     ring(f'WellRim{k}',(cx,cy,topz+.002),.095,.004,'steel')
     bbottom=topz-.125;br=.082
     for j,z in enumerate(np.arange(bbottom,topz+.01,.0035)):ring(f'Basket{k}_WireRing{j}',(cx,cy,z),br,.00065,'meshwire')
     for j,a in enumerate(np.arange(96)*2*np.pi/96):tube(f'Basket{k}_Upright{j}',[(cx+br*np.cos(a),cy+br*np.sin(a),bbottom),(cx+br*np.cos(a),cy+br*np.sin(a),topz+.009)],.00065,'meshwire')
     for j,v in enumerate(np.arange(-.075,.076,.006)):
      l=np.sqrt(br*br-v*v)
      tube(f'Basket{k}_BaseX{j}',[(cx-l,cy+v,bbottom),(cx+l,cy+v,bbottom)],.00065,'meshwire');tube(f'Basket{k}_BaseY{j}',[(cx+v,cy-l,bbottom),(cx+v,cy+l,bbottom)],.00065,'meshwire')
     ring(f'Basket{k}_TopRim',(cx,cy,topz+.008),br,.0028,'steel')
     # Handle mount is on the right of each basket, rising to black grips.
     hx=cx+.12
     for s in [-1,1]:tube(f'Basket{k}_HandleSupport{int(s+1)}',[(cx+.077,cy+s*.022,topz+.002),(hx,cy+s*.017,topz+.035),(hx+.005,cy+s*.012,topz+.073)],.0032,'steel')
     grip=tube(f'Basket{k}_BlackGrip',[(hx+.005,cy,topz+.063),(hx+.025,cy,topz+.225)],.013,'black_handle');gp=np.asarray(grip.GetPointsAttr().Get()).copy();gp[:,1]=cy+(gp[:,1]-cy)*.62;grip.GetPointsAttr().Set([Gf.Vec3f(*map(float,v)) for v in gp])
     tube(f'Basket{k}_Hook',[(hx+.025,cy,topz+.215),(hx+.032,cy,topz+.25),(hx+.055,cy,topz+.266),(hx+.088,cy,topz+.266)],.0026,'steel')
    for i,x in enumerate([fx-.14,fx+.14]):cube(f'TopRaisedEdge{i}',(x,fy,topz+.005),(.01,D,.012),'steel')
    cube('BoilerControlPanel',(fx,fy-D/2-.006,base+.085),(.23,.002,.115),'black_handle')
    cube('ControlScreen',(fx-.047,fy-D/2-.008,base+.104),(.064,.001,.028),'glass')
    for i in range(2):
     knob=cyl(f'BoilerKnob{i}',(fx+.026+i*.057,fy-D/2-.016,base+.081),.017,.016,'black_handle');UsdGeom.Xformable(knob).AddRotateXOp().Set(90.)
    # Lighting and a view from the robot side, similar to the supplied photograph.
    dome=su.DomeLightCfg(intensity=900,color=(.98,.98,1.),texture_file=str(ROOT/'Assets/Background/brown_photostudio_02_4k.hdr'));dome.func('/World/Dome',dome)
    for name,pos,power,size in [('Key',(-.6,-.2,2.7),1300,(2.,1.2)),('Fill',(1.0,-1.3,2.2),700,(1.4,1.4))]:
     ob=UsdLux.RectLight.Define(stage,'/World/'+name);ob.CreateIntensityAttr(power);ob.CreateWidthAttr(size[0]);ob.CreateHeightAttr(size[1]);M=Gf.Matrix4d().SetLookAt(Gf.Vec3d(*pos),Gf.Vec3d(0,.1,.8),Gf.Vec3d(0,0,1)).GetInverse();UsdGeom.Xformable(ob).AddTransformOp().Set(M)
    return locals()
