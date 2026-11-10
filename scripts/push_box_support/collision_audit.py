#!/usr/bin/env python3
"""Audit G1 proxy/box and proxy/floor distances at 30 Hz."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from make_demo import OUT, set_robot


def audit(out=OUT):
    motion=dict(np.load(out/'motion.npz'))
    sim=dict(np.load(out/'simulation.npz'))
    model=mujoco.MjModel.from_xml_path(str(out/'scene.xml'))
    data=mujoco.MjData(model)
    robot=[i for i in range(model.ngeom) if '_proxy_' in model.geom(i).name
           or model.geom(i).name.endswith('_foot_collision')
           or model.geom(i).name.endswith('_push_pad')]
    box=model.geom('box_collision').id
    floor=model.geom('floor').id
    worst_box=dict(penetration_m=0.,time_s=None,geoms=[])
    worst_floor=dict(penetration_m=0.,time_s=None,geoms=[])
    contacts=[];hand_clearance=[];line=np.zeros(6)
    hands=[model.geom(n).id for n in ('left_wrist_yaw_link_proxy_2',
                                      'right_wrist_yaw_link_proxy_2')]
    for k in range(0,len(motion['time']),2):
        t=float(motion['time'][k])
        set_robot(model,data,motion,t)
        data.qpos[:7]=sim['box_qpos'][int(round(t/model.opt.timestep))]
        mujoco.mj_forward(model,data)
        for i in robot:
            for j,worst in ((box,worst_box),(floor,worst_floor)):
                distance=mujoco.mj_geomDistance(model,data,i,j,.1,line)
                if distance < -worst['penetration_m']:
                    worst.update(penetration_m=-float(distance),time_s=t,
                                 geoms=[model.geom(i).name,model.geom(j).name])
                if distance < -.001:
                    contacts.append(dict(time_s=t,penetration_m=-float(distance),
                        geoms=[model.geom(i).name,model.geom(j).name]))
        if 1.6<=t<=8.5:
            for i in hands:
                hand_clearance.append(float(mujoco.mj_geomDistance(
                    model,data,i,box,.1,line)))
    result=dict(sample_hz=30,robot_box_worst=worst_box,
                robot_floor_worst=worst_floor,
                max_hand_mesh_box_clearance_m=max(hand_clearance),
                min_hand_mesh_box_distance_m=min(hand_clearance),
                contacts_over_1mm=sorted(contacts,key=lambda x:-x['penetration_m'])[:12])
    (out/'collision_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT)
    audit(p.parse_args().output)
