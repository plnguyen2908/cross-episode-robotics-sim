"""Connected extra-room wings attached to the existing three-room task core."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def settings(params):
    count=params.get('room_count',3)
    layout=params.get('room_arrangement','chain')
    width=params.get('extra_room_width',5.)
    depth=params.get('extra_room_depth',5.)
    if type(count) is not int or not 3<=count<=8:
        raise ValueError('room_count must be an integer from 3 to 8')
    if layout not in ('chain','branch'):
        raise ValueError('room_arrangement must be chain or branch')
    if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not np.isfinite(x) or not 3<=x<=6 for x in (width,depth)):
        raise ValueError('Extra-room width/depth must be finite and between 3 and 6 metres')
    return count,layout,float(width),float(depth)


def extend_room_graph(scene,params):
    """Author static geometry before robot insertion; save exact nodes and edges.

    Chain grows east. Branch grows an east spine with alternating north leaves.
    Each new room has one parent; existing core retains its three-edge loop.
    """
    count,arrangement,width,depth=settings(params)
    scene=Path(scene);tree=ET.parse(scene);root=tree.getroot();world=root.find('worldbody')
    nodes=[dict(name=n,floor_body=f) for n,f in [('kitchen','floor_room_main'),('dining','floor_dining_room_main'),('living','floor_living_room_main')]]
    edges=[dict(door=f'room_door_{i}',rooms=list(pair)) for i,pair in enumerate([('kitchen','dining'),('dining','living'),('kitchen','living')])]
    if count==3:
        graph=dict(rooms=nodes,connections=edges,arrangement=arrangement)
        scene.with_name('room_graph.json').write_text(json.dumps(graph,indent=2));return graph
    if world.find("body[@name='floor_extra_1_main']") is not None:
        raise ValueError('Room extension already installed; use a fresh scene')
    import mujoco
    model=mujoco.MjModel.from_xml_path(str(scene));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    wall=world.find("body[@name='east_living_room_main']")
    if wall is None:raise ValueError('Living room has no supported east wall')
    bid=model.body('east_living_room_main').id
    bounds=[]
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid]!=bid:continue
        r=data.geom_xmat[gid].reshape(3,3);a=model.geom_aabb[gid]
        center=data.geom_xpos[gid]+r@a[:3];half=np.abs(r)@a[3:];bounds.append((center-half,center+half))
    lo=np.min([a for a,b in bounds],axis=0);hi=np.max([b for a,b in bounds],axis=0)
    east=(lo[0]+hi[0])/2;cy=(lo[1]+hi[1])/2
    if hi[1]-lo[1] < depth:raise ValueError('Extra room depth exceeds shared living-room wall')
    walls={('living','east'):(wall,np.array([east,cy]),(hi[1]-lo[1])/2)}
    from cross_episode_sim.generation.room_furnishing import RoomFurnisher
    furnisher=RoomFurnisher(root)
    boxes={};height=3.;thickness=.06;opening=1.7
    def box(name,center,half,rgba):
        b=ET.SubElement(world,'body',name=name,pos=' '.join(map(str,center)))
        ET.SubElement(b,'geom',name=name+'_geom',type='box',size=' '.join(map(str,half)),rgba=rgba,group='1',contype='1',conaffinity='1')
        return b
    def wall_box(room,side,xy,half_length):
        half=[thickness,half_length,height/2] if side in ('east','west') else [half_length,thickness,height/2]
        b=box(side+'_'+room+'_main',[*xy,height/2],half,'.78 .78 .74 1')
        walls[(room,side)]=(b,np.array(xy),half_length)
    for i in range(count-3):
        name=f'extra_{i+1}'
        if i==0:
            parent='living';side='east';center=np.array([east+width/2,cy])
        elif arrangement=='branch' and i%2==1:
            parent=f'extra_{i}';side='north';center=boxes[parent]+[0,depth]
        else:
            parent=f'extra_{i if arrangement=="chain" else i-1}'
            side='east';center=boxes[parent]+[width,0]
        # Only non-overlapping grid cells are allowed.
        if any(np.all(np.abs(center-other)<np.array([width,depth])-1e-6) for other in boxes.values()):
            raise ValueError('Overlapping extra rooms')
        boxes[name]=center
        floor=box('floor_'+name+'_main',[*center,-.025],[width/2,depth/2,.025],'.48 .45 .40 1')
        for edge,offset,span in [('east',[width/2,0],depth/2),('west',[-width/2,0],depth/2),('north',[0,depth/2],width/2),('south',[0,-depth/2],width/2)]:
            wall_box(name,edge,center+offset,span)
        opposite='west' if side=='east' else 'south'
        # Replace both shared wall faces with matching openings, retaining
        # physical wall pieces and a lintel above the 2.35 m clear height.
        connection=walls[(parent,side)][1].copy()
        for key in ((parent,side),(name,opposite)):
            original,xy,span=walls.pop(key);world.remove(original)
            axis=1 if side=='east' else 0
            for suffix,a,b,z,h in [('a',-span,-opening/2,height/2,height/2),('b',opening/2,span,height/2,height/2),('lintel',-opening/2,opening/2,(height+2.35)/2,(height-2.35)/2)]:
                xy2=xy.copy();xy2[axis]+=(a+b)/2
                half=[thickness,thickness,h];half[axis]=(b-a)/2
                box(original.get('name')+'_'+suffix,[*xy2,z],half,'.78 .78 .74 1')
        number=i+3
        # Ordering makes initially open leaves swing into the new room.
        direction=np.array([0.,-1.]) if side=='east' else np.array([1.,0.])
        for label,sign in [('left',-1),('right',1)]:
            xy=connection+sign*(opening/2-.05)*direction
            half=[.05,.05,1.175]
            box(f'doorframe_{number}_{label}_{name}_main',[*xy,1.175],half,'.32 .23 .15 1')
        furniture=[]
        nodes.append(dict(name=name,room_type='reading_room' if i%2==0 else 'sitting_room',floor_body=floor.get('name'),center=center.tolist(),size=[width,depth],furniture=furniture))
        edges.append(dict(door=f'room_door_{number}',rooms=[parent,name],center=connection.tolist(),clear_width_m=opening-.2))
    for i,node in enumerate(nodes[3:]):
        node["furniture"]=furnisher.furnish(node["name"],node["center"],width,depth,i,leaf=(i==count-4 or arrangement=="branch" and i%2==1))
    tree.write(scene)
    graph=dict(rooms=nodes,connections=edges,arrangement=arrangement)
    scene.with_name('room_graph.json').write_text(json.dumps(graph,indent=2))
    return graph
