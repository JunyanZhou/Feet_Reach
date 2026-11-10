#!/usr/bin/env python3
"""Bake scripted leaning G1 and the dynamically pushed box into visual USD."""
from __future__ import annotations
import argparse, json, re, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import mujoco
import trimesh
from pxr import Usd, UsdGeom, UsdShade, UsdLux, Gf, Sdf, Vt
from make_demo import OUT, MODEL


def nums(s, default):
    return np.array([float(v) for v in s.split()]) if s else np.array(default, dtype=float)


def gfquat(q):
    return Gf.Quatf(float(q[0]), Gf.Vec3f(*map(float, q[1:])))


def export(out):
    motion = dict(np.load(out / 'motion.npz'))
    sim = dict(np.load(out / 'simulation.npz'))
    xml = ET.parse(out / 'scene.xml').getroot()
    model = mujoco.MjModel.from_xml_path(str(out / 'scene.xml'))
    data = mujoco.MjData(model)
    nframes = len(motion['time'])
    path = out / 'push_box_support.usdc'
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    stage.SetTimeCodesPerSecond(float(motion['fps']))
    stage.SetFramesPerSecond(float(motion['fps']))
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(nframes - 1)
    world = UsdGeom.Xform.Define(stage, '/World')
    stage.SetDefaultPrim(world.GetPrim())
    world.GetPrim().SetCustomDataByKey('demo_mode',
        'scripted leaning G1; MuJoCo free box, ground friction, palm contacts; visual replay')
    mesh_defs = {m.get('name'): m for m in xml.findall('./asset/mesh')}
    cache, materials = {}, {}

    def material(rgba):
        key = tuple(round(float(v), 5) for v in rgba)
        if key in materials:
            return materials[key]
        path_m = '/World/Materials/m%03d' % len(materials)
        mat = UsdShade.Material.Define(stage, path_m)
        shader = UsdShade.Shader.Define(stage, path_m + '/PreviewSurface')
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*key[:3]))
        shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.48)
        shader.CreateInput('opacity', Sdf.ValueTypeNames.Float).Set(key[3])
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
        materials[key] = mat
        return mat

    def geom(parent, element, index):
        rgba = nums(element.get('rgba'), [.6, .6, .6, 1])
        if rgba[3] < .01 or element.get('group') == '3':
            return
        name = re.sub('[^a-zA-Z0-9_]', '_', element.get('name', f'geom_{index}'))
        gp = parent + '/' + name
        xf = UsdGeom.Xform.Define(stage, gp)
        xf.AddTranslateOp().Set(Gf.Vec3d(*nums(element.get('pos'), [0, 0, 0])))
        xf.AddOrientOp().Set(gfquat(nums(element.get('quat'), [1, 0, 0, 0])))
        size = nums(element.get('size'), [.01])
        kind = element.get('type', 'sphere')
        shape = gp + '/Shape'
        if kind == 'mesh':
            mesh_name = element.get('mesh')
            if mesh_name not in cache:
                entry = mesh_defs[mesh_name]
                source = MODEL.parent / 'meshes' / entry.get('file')
                loaded = trimesh.load(source, process=False)
                scale = nums(entry.get('scale'), [1, 1, 1])
                cache[mesh_name] = (np.asarray(loaded.vertices, dtype=np.float32) * scale,
                                    np.asarray(loaded.faces, dtype=np.int32))
            points, faces = cache[mesh_name]
            g = UsdGeom.Mesh.Define(stage, shape)
            g.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
            g.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(
                np.full(len(faces), 3, dtype=np.int32)))
            g.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.flatten()))
            g.CreateSubdivisionSchemeAttr('none')
            g.CreateDoubleSidedAttr(True)
        elif kind in ('box', 'plane'):
            g = UsdGeom.Cube.Define(stage, shape)
            g.CreateSizeAttr(2)
            if kind == 'plane':
                size = [15, 15, .001]
                xf.GetOrderedXformOps()[0].Set(Gf.Vec3d(0, 0, -.002))
                rgba = np.array([.21, .26, .31, 1])
            UsdGeom.Xformable(g).AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
        elif kind == 'sphere':
            g = UsdGeom.Sphere.Define(stage, shape)
            g.CreateRadiusAttr(float(size[0]))
        else:
            raise ValueError('Unsupported geom ' + kind)
        UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(material(rgba))

    wb = xml.find('worldbody')
    for j, g in enumerate(wb.findall('geom')):
        geom('/World/Environment', g, j)
    for body_name in motion['body_names']:
        body_name = str(body_name)
        source = wb.find(f'./body[@name="{body_name}"]')
        xf = UsdGeom.Xform.Define(stage, '/World/G1/' + body_name)
        p, q = xf.AddTranslateOp(), xf.AddOrientOp()
        j = list(motion['body_names']).index(body_name)
        for i in range(nframes):
            p.Set(Gf.Vec3d(*map(float, motion['body_pos'][i, j])), i)
            q.Set(gfquat(motion['body_quat_wxyz'][i, j]), i)
        for k, element in enumerate(source.findall('geom')):
            geom(str(xf.GetPath()), element, k)
    box_source = wb.find('./body[@name="box"]')
    box_id = model.body('box').id
    box_xf = UsdGeom.Xform.Define(stage, '/World/Box')
    box_p, box_q = box_xf.AddTranslateOp(), box_xf.AddOrientOp()
    box_positions, box_quats = [], []
    for i, t in enumerate(motion['time']):
        data.qpos[:7] = sim['box_qpos'][int(round(t/model.opt.timestep))]
        mujoco.mj_forward(model, data)
        pos, quat = data.xpos[box_id].copy(), data.xquat[box_id].copy()
        box_positions.append(pos)
        box_quats.append(quat)
        box_p.Set(Gf.Vec3d(*map(float, pos)), i)
        box_q.Set(gfquat(quat), i)
    for k, element in enumerate(box_source.findall('geom')):
        geom(str(box_xf.GetPath()), element, k)
    dome = UsdLux.DomeLight.Define(stage, '/World/Lighting/Dome')
    dome.CreateIntensityAttr(550)
    dome.CreateColorAttr(Gf.Vec3f(.80, .88, 1.0))
    key = UsdLux.DistantLight.Define(stage, '/World/Lighting/Key')
    key.CreateIntensityAttr(2100)
    key.CreateColorAttr(Gf.Vec3f(1, .93, .83))
    key.AddRotateXYZOp().Set(Gf.Vec3f(25, -35, -25))
    for name, azimuth, distance in [('Overview', 45, 3.10), ('Side', 90, 3.05)]:
        el = np.deg2rad(-18 if name == 'Overview' else -12)
        az = np.deg2rad(azimuth)
        target = np.array([.65, 0, .52])
        eye = target + distance*np.array([-np.cos(el)*np.cos(az),
                                           -np.cos(el)*np.sin(az), -np.sin(el)])
        cam = UsdGeom.Camera.Define(stage, '/World/Cameras/' + name)
        cam.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse())
        cam.CreateHorizontalApertureAttr(36)
        cam.CreateVerticalApertureAttr(20.25)
        cam.CreateFocalLengthAttr(24.44)
        cam.CreateClippingRangeAttr(Gf.Vec2f(.01, 100))
    stage.GetRootLayer().Save()
    np.savez_compressed(out / 'trajectory_portable.npz',
        time_s=motion['time'], fps=motion['fps'],
        joint_names=motion['joint_names'], joint_position_rad=motion['qpos'][:, 7:],
        root_position_world_m=motion['qpos'][:, :3],
        root_quaternion_wxyz=motion['qpos'][:, 3:7],
        body_names=motion['body_names'], body_position_world_m=motion['body_pos'],
        body_quaternion_wxyz=motion['body_quat_wxyz'],
        box_position_world_m=np.array(box_positions),
        box_quaternion_wxyz=np.array(box_quats),
        box_free_joint_qpos=np.array([sim['box_qpos'][int(round(t/model.opt.timestep))]
                                     for t in motion['time']]),
        palm_box_normal_force_N=np.interp(motion['time'],sim['time'],sim['push_force_N']))
    manifest = {'duration_s': float(motion['time'][-1]),
        'fps': float(motion['fps']), 'frame_count': nframes,
        'units': 'meters, radians, seconds', 'up_axis': 'Z',
        'quaternion_order': 'wxyz', 'robot': 'G1 29-DoF scripted kinematics',
        'box_physics': 'MuJoCo free body, floor friction, two palm contact spheres',
        'robot_motion': 'scripted forward lean, palm targets, and stepping feet',
        'usd': 'visual replay only; no PhysX articulation or force control',
        'isaac_runtime_tested': False,
        'keyframes': [{'time_s': k[0], 'phase': k[6]} for k in __import__('make_demo').KEYS]}
    (out / 'scene_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Exported {path} ({path.stat().st_size/1e6:.1f} MB)')
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT)
    export(parser.parse_args().output)
