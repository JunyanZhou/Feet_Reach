#!/usr/bin/env python3
"""Audit all robot/furniture and cargo/furniture geometry at 30 Hz."""
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
           or model.geom(i).name.endswith('_foot_collision')]
    furniture=[model.geom(n).id for n in ('door_panel','door_rail','door_push_plate',
        'hinge_post','latch_post','lintel','cargo_box')]
    cargo=model.geom('cargo_box').id
    fixed=[model.geom(n).id for n in ('door_panel','door_rail','door_push_plate',
        'hinge_post','latch_post','lintel')]
    worst=dict(penetration_m=0.,time_s=None,geoms=[])
    worst_cargo=dict(penetration_m=0.,time_s=None,geoms=[])
    contacts=[]
    line=np.zeros(6)
    for i in range(0,len(motion['time']),2):
        t=float(motion['time'][i])
        set_robot(model,data,motion,t)
        data.qpos[model.joint('door_hinge').qposadr[0]]=(
            np.interp(t,sim['time'],sim['door_angle']))
        mujoco.mj_forward(model,data)
        for a in robot:
            for b in furniture:
                distance=mujoco.mj_geomDistance(model,data,a,b,.05,line)
                if distance < -worst['penetration_m']:
                    worst=dict(penetration_m=-float(distance),time_s=t,
                               geoms=[model.geom(a).name,model.geom(b).name])
                if distance < -.001:
                    contacts.append(dict(time_s=t,penetration_m=-float(distance),
                        geoms=[model.geom(a).name,model.geom(b).name]))
        for b in fixed:
            distance=mujoco.mj_geomDistance(model,data,cargo,b,.05,line)
            if distance < -worst_cargo['penetration_m']:
                worst_cargo=dict(penetration_m=-float(distance),time_s=t,
                                 geoms=['cargo_box',model.geom(b).name])
    result=dict(sample_hz=30,robot_furniture_worst=worst,
                cargo_furniture_worst=worst_cargo,
                contacts_over_1mm=sorted(contacts,key=lambda x:-x['penetration_m'])[:12])
    (out/'collision_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=OUT)
    audit(p.parse_args().output)
