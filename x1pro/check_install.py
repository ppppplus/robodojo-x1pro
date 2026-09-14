"""Read-only X1 Pro fork asset and kinematic-chain check."""
from pathlib import Path
import xml.etree.ElementTree as ET

repo = Path(__file__).resolve().parents[1]
required = [
    repo / 'Assets/Object/RoboDojo/Rigid/bowl/00014/object.usdz',
    repo / 'Assets/Material/material_0114/Bamboo_Planks_BaseColor.png',
    repo / 'Assets/Background/brown_photostudio_02_4k.hdr',
]
for variant in ('fx001_h_evt1', 'rm001_g1_dvt1'):
    base = repo / 'x1pro/assets' / variant
    required.extend((base / 'kinematics.urdf', base / 'appearance/x1pro_appearance.usd', base / 'appearance/textures/chest_logo_basecolor.png', base / 'usd_control/ex001.usd', base / 'usd_control/configuration/ex001_base.usd', base / 'usd_control/configuration/ex001_robot.usd', base / 'usd_control/configuration/ex001_physics.usd', base / 'usd_control/configuration/ex001_sensor.usd'))
missing = [str(path.relative_to(repo)) for path in required if not path.is_file()]
for usd in (repo / 'x1pro/assets').glob('*/usd_control/configuration/ex001_base.usd'):
    with usd.open('rb') as stream:
        header = stream.read(8)
    if header != b'PXR-USDC':
        raise SystemExit(f'{usd}: binary USD not hydrated; install the model bundle with x1pro/init_assets.sh')
if missing:
    raise SystemExit('Missing X1 Pro dependencies:\n' + '\n'.join(missing))
for variant in ('fx001_h_evt1', 'rm001_g1_dvt1'):
    urdf = ET.parse(repo / 'x1pro/assets' / variant / 'kinematics.urdf').getroot()
    joint_names = {joint.get('name') for joint in urdf.findall('joint')}
    expected = {'lift_joint', 'head_yaw_joint', 'head_pitch_joint'} | {f'{side}_arm_joint{i}' for side in ('left', 'right') for i in range(1, 7)}
    if not expected.issubset(joint_names):
        raise SystemExit(f'{variant}: missing joints {sorted(expected - joint_names)}')
    if any(root.get('filename', '').startswith('/') for root in urdf.iter('mesh')):
        raise SystemExit(f'{variant}: absolute mesh path in portable kinematics URDF')
    print(f'{variant}: {len(joint_names)} joints, assets found')
print('X1 Pro fork dependencies: OK')
