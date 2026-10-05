"""Physical small-box pour into a converted Moonlake portafilter.

This is a fixture-driven asset trial, NOT a robot execution. Welded mocap
fixtures move the box and portafilter; particles remain free dynamic bodies.
The portafilter is released from the source extraction rail so it can be
withdrawn fully, loaded, and returned. No particle pose is reset after setup.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation, Slerp

from cross_episode_sim.tasks.coffee.grounds import (
    PARTICLE_RADIUS, add_grounds, make_dosing_box,
)


def numbers(values):
    return ' '.join(f'{float(v):.10g}' for v in values)

PORTAFILTER = 'portafilter_extract_pivot_001'
CENTER = np.array([-.0173, -.0193])
BOTTOM, RIM, INNER_RADIUS = .1762, .2048, .029


def pose(position, rotation=None):
    result = np.eye(4)
    result[:3, 3] = position
    if rotation is not None:
        result[:3, :3] = rotation
    return result


def mix(a, b, u):
    u = np.clip(u, 0, 1)
    u = u*u*(3-2*u)
    result = pose((1-u)*a[:3, 3]+u*b[:3, 3])
    result[:3, :3] = Slerp([0, 1], Rotation.from_matrix([a[:3, :3], b[:3, :3]]))([u]).as_matrix()[0]
    return result


def pour_pose(spec, angle):
    """Tip over a box corner to narrow the stream for the 58 mm basket."""
    cavity, target = spec['dosing_cavity'], spec['hopper']
    x, y = cavity['half_width_xy']
    corner = np.array([x, y, cavity['rim_z']])
    yaw = np.arctan2(x, y)
    rotation = Rotation.from_euler('x', -angle, degrees=True).as_matrix() @ Rotation.from_euler('z', yaw).as_matrix()
    # The broad side walls sweep lower than the pouring corner near 75 degrees.
    # Keep clearance during that sweep, then lower only at the final pour angle.
    height = .12-(.12-.025)*np.clip((angle-35)/40, 0, 1)
    height -= (.025-spec['pour_lip_clearance_m'])*np.clip((angle-90)/15, 0, 1)
    lip = np.array([*target['center_xy'], target['top_z']+height])
    # Leave room for the stream's forward motion instead of aiming at the far rim.
    lip[1] -= .006
    return pose(lip-rotation@corner, rotation)


def fixture(world, equality, body, name, initial):
    body.set('pos', numbers(initial[:3, 3]))
    body.set('quat', numbers(Rotation.from_matrix(initial[:3, :3]).as_quat(scalar_first=True)))
    ET.SubElement(body, 'freejoint', name=name+'_free')
    ET.SubElement(world, 'body', name=name+'_fixture', mocap='true',
                  pos=body.get('pos'), quat=body.get('quat'))
    ET.SubElement(equality, 'weld', name=name+'_fixture_weld', body1=body.get('name'),
                  body2=name+'_fixture', solref='.004 1', solimp='.99 .999 .001')


def prepare(asset, output):
    report = json.loads((asset/'conversion.json').read_text())
    root = ET.parse(asset/'machine.xml').getroot()
    world = root.find('worldbody')
    # Other source articulations remain fixed for this loading-only trial.
    for parent in root.iter():
        for child in list(parent):
            if child.tag == 'joint': parent.remove(child)
    basket = root.find(f".//body[@name='{PORTAFILTER}']")
    parent = next(p for p in root.iter() if basket in list(p))
    parent.remove(basket)
    world.append(basket)
    seated = np.array(report['body_world_poses'][PORTAFILTER])
    initial_box = pose([.18, -.30, .040])
    box = ET.SubElement(world, 'body', name='coffee_dosing_box')
    cavity = make_dosing_box(box)
    equality = ET.SubElement(root, 'equality')
    fixture(world, equality, basket, 'portafilter', seated)
    fixture(world, equality, box, 'box', initial_box)
    ET.SubElement(world, 'geom', name='counter', type='box', pos='0 -.10 -.025',
                  size='.5 .6 .025', rgba='.36 .27 .20 1', friction='.8 .005 .0001')
    ET.SubElement(world, 'light', pos='-.5 -.5 1.4', dir='.3 .3 -1', diffuse='.8 .8 .8')
    ET.SubElement(world, 'light', pos='.5 .3 1.0', dir='-.3 -.2 -1', diffuse='.5 .5 .5')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='960', offheight='720')
    ET.SubElement(visual, 'headlight', ambient='.35 .35 .35', diffuse='.6 .6 .6')
    names = add_grounds(world, initial_box[:3, 3], cavity)
    withdrawn = seated.copy()
    turn = Rotation.from_euler('z', -50, degrees=True).as_matrix()
    withdrawn[:3, :3] = turn @ seated[:3, :3]
    lowered = withdrawn.copy(); lowered[2, 3] -= .08
    loading = lowered.copy(); loading[1, 3] -= .24
    offset = loading[:3, 3]-seated[:3, 3]
    spec = dict(dosing_body='coffee_dosing_box', dosing_cavity=cavity, grains=names,
                hopper=dict(center_xy=(CENTER+offset[:2]).tolist(), top_z=RIM+offset[2]),
                pour_lip_clearance_m=.010)
    scene = output/'trial.xml'
    ET.indent(root); ET.ElementTree(root).write(scene)
    return scene, spec, seated, withdrawn, lowered, loading, initial_box


def render(output):
    m = mujoco.MjModel.from_xml_path(str((output/'trial.xml').resolve()))
    d = mujoco.MjData(m)
    trace = np.load(output/'trace.npz')
    inventory = json.loads((output/'inventory.json').read_text())
    times = np.array([sample['time'] for sample in inventory])
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 19)
    camera = mujoco.MjvCamera(); camera.lookat[:] = [0, -.12, .15]
    camera.distance = .92; camera.azimuth = 115; camera.elevation = -28
    option = mujoco.MjvOption(); option.geomgroup[0] = 0; option.sitegroup[:] = 0
    m.geom_group[m.geom('counter').id] = 2
    with mujoco.Renderer(m, width=960, height=720) as renderer, imageio.get_writer(
        output/'small_box_portafilter.mp4', fps=25, codec='libx264',
        pixelformat='yuv420p', output_params=['-movflags', '+faststart']) as writer:
        for t, q, pos, quat in zip(trace['time'], trace['qpos'], trace['mocap_pos'], trace['mocap_quat']):
            d.qpos[:] = q; d.mocap_pos[:] = pos; d.mocap_quat[:] = quat
            mujoco.mj_forward(m, d); renderer.update_scene(d, camera=camera, scene_option=option)
            frame = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, 960, 39), fill=(24, 29, 35))
            draw.text((15, 8), 'Moonlake asset physics trial | motion fixtures, no robot', font=font, fill='white')
            phase = ('Withdraw portafilter' if t < 6 else 'Position small box' if t < 9
                     else 'Pour grounds into basket' if t < 15 else 'Return small box' if t < 20
                     else 'Reinstall loaded portafilter' if t < 25.5 else 'Check retention')
            counts = inventory[np.argmin(abs(times-t))]
            draw.rectangle((0, 677, 960, 720), fill=(24, 29, 35))
            draw.text((15, 688), f"{t:4.1f}s  {phase}    Basket: {counts['basket']}/32    Outside: {counts['spilled']}", font=font, fill='white')
            frame = np.array(frame); writer.append_data(frame)
            for name, at in [('withdrawn', 6.5), ('pouring', 12), ('loaded', 15), ('reinstalled', 27.9)]:
                if abs(t-at) < .021: imageio.imwrite(output/f'{name}.png', frame)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--no-render', action='store_true')
    p.add_argument('--replay', action='store_true', help='Render an existing trace without rerunning physics')
    args = p.parse_args()
    if args.replay:
        render(args.output)
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    scene, spec, seated, turned, lowered, loading, parked = prepare(args.asset, args.output)
    m = mujoco.MjModel.from_xml_path(str(scene.resolve()))
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    grain_ids = np.array([m.body(n).id for n in spec['grains']])
    pfid, boxid = m.body(PORTAFILTER).id, m.body('coffee_dosing_box').id
    mocap_pf = m.body_mocapid[m.body('portafilter_fixture').id]
    mocap_box = m.body_mocapid[m.body('box_fixture').id]
    hover = parked.copy(); hover[2, 3] = .32
    pour_start = pour_pose(spec, 0)
    lifted_pour_end = pour_pose(spec, 105); lifted_pour_end[2, 3] += .06
    lifted_pour_start = pour_start.copy(); lifted_pour_start[2, 3] += .06
    frames, samples, milestones = [], [], {}
    next_frame, next_sample = 0., 0.
    worst = dict(box_machine=0., box_portafilter=0., portafilter_machine=0.,
                 grain_portafilter=0., box_fixture_position_error=0., portafilter_fixture_position_error=0.)
    worst_contacts = {}
    grain_set = set(grain_ids.tolist())
    machine_bodies = {b for b in range(1, m.nbody) if b not in grain_set and b not in (pfid, boxid)
                      and not m.body_mocapid[b] >= 0}

    def inventory():
        points = d.xpos[grain_ids]
        b = d.body(PORTAFILTER)
        original = (points-b.xpos) @ b.xmat.reshape(3, 3) @ seated[:3, :3].T + seated[:3, 3]
        inside = (np.linalg.norm(original[:, :2]-CENTER, axis=1) <= INNER_RADIUS-PARTICLE_RADIUS+.001)
        inside &= (original[:, 2]-PARTICLE_RADIUS >= BOTTOM-.002)
        inside &= (original[:, 2]+PARTICLE_RADIUS <= RIM+.001)
        box_body = d.body('coffee_dosing_box')
        local = (points-box_body.xpos) @ box_body.xmat.reshape(3, 3)
        in_box = np.all(np.abs(local[:, :2]) <= np.array([.036, .031])+PARTICLE_RADIUS, axis=1)
        in_box &= (local[:, 2] >= -.0335-.005) & (local[:, 2] <= .0375+PARTICLE_RADIUS)
        return dict(basket=int(inside.sum()), box=int((in_box & ~inside).sum()),
                    spilled=int((~inside & ~in_box).sum()))

    while d.time < 28:
        t = d.time
        if t < 1: pf = seated
        elif t < 2.5: pf = mix(seated, turned, (t-1)/1.5)
        elif t < 4: pf = mix(turned, lowered, (t-2.5)/1.5)
        elif t < 6: pf = mix(lowered, loading, (t-4)/2)
        elif t < 20: pf = loading
        elif t < 22: pf = mix(loading, lowered, (t-20)/2)
        elif t < 24: pf = mix(lowered, turned, (t-22)/2)
        elif t < 25.5: pf = mix(turned, seated, (t-24)/1.5)
        else: pf = seated
        if t < 7: box = parked
        elif t < 8: box = mix(parked, hover, t-7)
        elif t < 9: box = mix(hover, pour_start, t-8)
        elif t < 13: box = pour_pose(spec, 105*(t-9)/4)
        elif t < 15: box = pour_pose(spec, 105)
        elif t < 15.8: box = mix(pour_pose(spec, 105), lifted_pour_end, (t-15)/.8)
        elif t < 18:
            box = pour_pose(spec, 105*(18-t)/2.2); box[2, 3] += .06
        elif t < 19: box = mix(lifted_pour_start, hover, t-18)
        elif t < 20: box = mix(hover, parked, t-19)
        else: box = parked
        for idx, target in ((mocap_pf, pf), (mocap_box, box)):
            d.mocap_pos[idx] = target[:3, 3]
            d.mocap_quat[idx] = Rotation.from_matrix(target[:3, :3]).as_quat(scalar_first=True)
        mujoco.mj_step(m, d)
        worst['box_fixture_position_error'] = max(worst['box_fixture_position_error'], float(np.linalg.norm(d.body('coffee_dosing_box').xpos-box[:3, 3])))
        worst['portafilter_fixture_position_error'] = max(worst['portafilter_fixture_position_error'], float(np.linalg.norm(d.body(PORTAFILTER).xpos-pf[:3, 3])))
        for contact in d.contact:
            b1, b2 = int(m.geom_bodyid[contact.geom1]), int(m.geom_bodyid[contact.geom2])
            pair = {b1, b2}
            depth = max(0., -float(contact.dist))
            key = None
            if pair == {boxid, pfid}: key = 'box_portafilter'
            elif boxid in pair and pair & machine_bodies: key = 'box_machine'
            elif pfid in pair and pair & machine_bodies: key = 'portafilter_machine'
            elif pfid in pair and pair & grain_set: key = 'grain_portafilter'
            if key and depth > worst[key]:
                worst[key] = depth
                worst_contacts[key] = dict(time=float(d.time), geoms=[m.geom(contact.geom1).name, m.geom(contact.geom2).name])
        if d.time >= next_frame:
            frames.append((d.time, d.qpos.copy(), d.mocap_pos.copy(), d.mocap_quat.copy()))
            next_frame += 1/25
        if d.time >= next_sample:
            counts = inventory(); samples.append(dict(time=float(d.time), **counts))
            next_sample += .1
        for name, at in [('initial_box', .8), ('withdrawn', 6.5), ('poured', 15), ('before_reinstall', 19.9), ('reinstalled', 27.9)]:
            if name not in milestones and d.time >= at:
                milestones[name] = dict(time=float(d.time), **inventory())
                print(name, milestones[name], flush=True)
    counts = inventory()
    success = (counts == dict(basket=32, box=0, spilled=0)
               and milestones['initial_box']['box'] == 32
               and max(worst[k] for k in ('box_machine', 'box_portafilter', 'portafilter_machine')) < .001
               and worst['grain_portafilter'] < .0015
               and max(worst[k] for k in ('box_fixture_position_error', 'portafilter_fixture_position_error')) < .005)
    result = dict(success=bool(success), scope=__doc__, counts=counts, milestones=milestones,
                  max_penetration_or_error_m=worst, worst_contacts=worst_contacts,
                  particle_pose_resets_after_initialization=0,
                  particle_radius_m=PARTICLE_RADIUS, particle_count=32,
                  pour=dict(corner=True, angle_degrees=105, lip_clearance_m=spec['pour_lip_clearance_m'], lip_offset_y_m=-.006),
                  source='Moonlake espresso_machine', scene_xml=str(scene.resolve()),
                  adaptation='Source extraction rail replaced by fixture-held free body for withdrawal and reinstallation; other joints fixed.',
                  asset_conversion=json.loads((args.asset/'conversion.json').read_text()),
                  robot_execution=False, brewing_simulated=False)
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    (args.output/'inventory.json').write_text(json.dumps(samples, indent=2)+'\n')
    np.savez_compressed(args.output/'trace.npz', time=[x[0] for x in frames],
                        qpos=[x[1] for x in frames], mocap_pos=[x[2] for x in frames],
                        mocap_quat=[x[3] for x in frames])
    print(json.dumps({k: v for k, v in result.items() if k != 'asset_conversion'}), flush=True)
    if not args.no_render:
        render(args.output)
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
