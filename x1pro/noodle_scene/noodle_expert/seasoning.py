"""Estimated granular contents and genuinely perforated shaker caps (metres)."""
import numpy as np
from pxr import UsdGeom, UsdShade, Gf, Sdf, Vt
from scipy.spatial import Delaunay
RADIUS=.0011
SPACING=.0027
HOLE_RADIUS=.0032

def grain_positions(origin):
    pts=[]
    for z in np.arange(.008,.039,SPACING):
        for y in np.arange(-.023,.024,SPACING):
            for x in np.arange(-.023,.024,SPACING):
                if x*x+y*y<.023**2:pts.append(np.asarray(origin)+[x,y,z])
    return np.asarray(pts,dtype=np.float32)

def add_contents(stage,path,origin,kind):
    pts=grain_positions(origin)
    obj=UsdGeom.PointInstancer.Define(stage,path)
    obj.CreatePositionsAttr(Vt.Vec3fArray.FromNumpy(pts))
    obj.CreateProtoIndicesAttr([i%3 for i in range(len(pts))])
    colors=[(.48,.035,.008),(.75,.09,.014),(.92,.32,.055)] if kind=='chili' else [(.92,.92,.87),(.99,.99,.97),(.80,.82,.80)]
    targets=[]
    for i,color in enumerate(colors):
        proto=UsdGeom.Sphere.Define(stage,path+'/grain'+str(i));proto.CreateRadiusAttr(RADIUS)
        mat=UsdShade.Material.Define(stage,f'/World/Materials/{kind}_grain{i}')
        sh=UsdShade.Shader.Define(stage,str(mat.GetPath())+'/Surface');sh.CreateIdAttr('UsdPreviewSurface')
        sh.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color));sh.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.75)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(),'surface');UsdShade.MaterialBindingAPI.Apply(proto.GetPrim()).Bind(mat)
        targets.append(proto.GetPath())
    obj.CreatePrototypesRel().SetTargets(targets)
    obj.GetPrim().SetCustomDataByKey('contents',kind)
    obj.GetPrim().SetCustomDataByKey('static_preview',True)
    return obj

def perforated_top(stage,path,origin,material):
    holes=np.array([[.014*np.cos(a),.014*np.sin(a)] for a in np.arange(7)*2*np.pi/7])
    angles=np.arange(64)*2*np.pi/64
    points=[np.c_[.028*np.cos(angles),.028*np.sin(angles)]]
    for h in holes:points.append(h+np.c_[HOLE_RADIUS*np.cos(angles),HOLE_RADIUS*np.sin(angles)])
    points=np.concatenate(points)
    triangles=Delaunay(points).simplices
    mid=points[triangles].mean(axis=1)
    keep=(np.linalg.norm(mid,axis=1)<.028)&(np.linalg.norm(mid[:,None,:]-holes[None,:,:],axis=2).min(axis=1)>HOLE_RADIUS)
    triangles=triangles[keep];n=len(points)
    xyz=np.concatenate([np.c_[points,np.full(n,-.004)],np.c_[points,np.full(n,.004)]])+np.asarray(origin)
    faces=[];edges={}
    for tri in triangles:
        faces.extend(tri[::-1].tolist());faces.extend((tri+n).tolist())
        for a,b in zip(tri,np.roll(tri,-1)):
            k=tuple(sorted((int(a),int(b))));edges[k]=edges.get(k,0)+1
    for (a,b),count in edges.items():
        if count==1:faces.extend([a,b,b+n,a,b+n,a+n])
    obj=UsdGeom.Mesh.Define(stage,path);obj.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(xyz.astype(np.float32)))
    obj.CreateFaceVertexCountsAttr([3]*(len(faces)//3));obj.CreateFaceVertexIndicesAttr(faces);obj.CreateSubdivisionSchemeAttr('none');obj.CreateDoubleSidedAttr(True)
    UsdShade.MaterialBindingAPI.Apply(obj.GetPrim()).Bind(material)
    return obj
