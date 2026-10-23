"""Seeded dining/living extensions of native RoboCasa kitchen layouts.

Geometry generation has no simulator dependency. Dimensions are metres.
The north table edge is reserved for manipulation; chairs occupy other sides.
"""
import copy

import numpy as np
import yaml


def build_dining_layout(rng=None, base_layout=1):
    rng = np.random.default_rng() if rng is None else rng
    if base_layout not in range(1,11):
        raise ValueError('Native kitchen layout must be in 1..10')
    from cross_episode_sim.paths import robocasa_dir
    path = robocasa_dir()/f'models/assets/scenes/kitchen_layouts/test/layout{base_layout:03d}.yaml'
    layout = yaml.safe_load(path.read_text())
    layout['room']['walls'] = [w for w in layout['room']['walls']
                               if not w['name'].startswith('wall_front')]
    # Reserve a circulation apron between native appliances and the added
    # room-door leaves. Do not relocate individual appliances or shrink doors.
    # Native fixtures retain their coordinates; only the room shell expands.
    entry_clearance = 1.8
    for item in layout['room']['floor']:
        item['pos'][1] -= entry_clearance/2
        item['size'][1] += entry_clearance/2
    for item in layout['room']['walls']:
        if item.get('wall_side') in ('left', 'right'):
            item['pos'][1] -= entry_clearance/2
            item['size'][0] += entry_clearance/2
    floor=layout['room']['floor'][0]
    cx,cy,_=floor['pos'];hx,hy,_=floor['size']
    xmin,xmax,front,back=cx-hx,cx+hx,cy-hy,cy+hy
    span=xmax-xmin
    depth = float(rng.uniform(4.8, 5.5))
    doorway_x = float(rng.uniform(cx-.35, cx+.35))
    doorway_width = float(rng.uniform(1.65, 1.95))
    left, right = doorway_x-doorway_width/2, doorway_x+doorway_width/2
    width, length, height = float(rng.uniform(1.4,1.9)), float(rng.uniform(.8,1.0)), float(rng.uniform(.76,.84))
    tx, ty = float(rng.uniform(cx-.35,cx+.35)), front-depth*.60
    furniture = [dict(name='dining_table', type='counter', size=[width,length,height],
                      pos=[tx,ty,height/2], hollow=[True,True],top_thickness=.04)]
    for i,(dx,dy) in enumerate(((-1,-1),(-1,1),(1,-1),(1,1))):
        furniture.append(dict(name=f'table_leg_{i}',type='box',
            pos=[tx+dx*(width/2-.08),ty+dy*(length/2-.08),(height-.04)/2],
            size=[.09,.09,height-.04]))
    # Two side chairs, plus zero, one or two along the far edge.
    chair_positions = [(tx-width/2-.42,ty,'west'),(tx+width/2+.42,ty,'east')]
    extra = int(rng.integers(0,3))
    if extra == 1:
        chair_positions.append((tx,ty-length/2-.42,'south'))
    elif extra == 2:
        chair_positions.extend((tx+dx,ty-length/2-.42,'south') for dx in (-.38,.38))
    chair_bounds = []
    for i,(x,y,side) in enumerate(chair_positions):
        size = float(rng.uniform(.40,.46))
        seat_z = float(rng.uniform(.43,.47))
        furniture.append(dict(name=f'chair_{i}_seat',type='box',
            pos=[x,y,seat_z],size=[size,size,.08]))
        bx,by=x,y
        if side=='west': bx-=size/2-.035
        elif side=='east': bx+=size/2-.035
        else: by-=size/2-.035
        furniture.append(dict(name=f'chair_{i}_back',type='box',pos=[bx,by,.70],
            size=[.07,size,.52] if side!='south' else [size,.07,.52]))
        for j,(dx,dy) in enumerate(((-1,-1),(-1,1),(1,-1),(1,1))):
            furniture.append(dict(name=f'chair_{i}_leg_{j}',type='box',
                pos=[x+dx*(size/2-.05),y+dy*(size/2-.05),(seat_z-.04)/2],
                size=[.055,.055,seat_z-.04]))
        chair_bounds.append([x-size/2,y-size/2,x+size/2,y+size/2])
    def wall(name,pos,size,side):
        return dict(name=name,type='wall',pos=pos,size=size,wall_side=side)
    walls=[wall('partition_left',[(xmin+left)/2,front,1.5],[(left-xmin)/2,1.5,.02],'front'),
           wall('partition_right',[(right+xmax)/2,front,1.5],[(xmax-right)/2,1.5,.02],'front'),
           wall('left',[xmin,front-depth/2,1.5],[depth/2,1.5,.02],'left'),
           wall('right',[xmax,front-depth/2,1.5],[depth/2,1.5,.02],'right'),
           dict(wall('front',[cx,front-depth,1.5],[span/2,1.5,.02],'front'),enclosing_wall=True)]
    layout['dining_room']=dict(walls=walls,
        floor=[dict(name='floor',type='floor',size=[span/2,depth/2,.02],pos=[cx,front-depth/2,0])],
        furniture=furniture)
    kitchen_counters=[f['name']+'_'+group_name for group_name,group in layout.items()
        if group_name != 'dining_room' for fixtures in group.values() if isinstance(fixtures,list)
        for f in fixtures if isinstance(f,dict) and f.get('type')=='counter']
    metadata=dict(layout_id=1002,base_layout_id=base_layout,kitchen_entry_clearance_m=entry_clearance,room_bounds={'kitchen':[xmin,front,xmax,back],
        'dining_room':[xmin,front-depth,xmax,front]},
        doorway=dict(center=[doorway_x,front],nominal_width_m=doorway_width,
                     clear_width_m=doorway_width-.08),
        table=dict(center=[tx,ty],size=[width,length,height],
                   fixture='dining_table_dining_room'),
        chairs=chair_bounds,chair_count=len(chair_bounds),
        manipulation_stance=[tx,ty+length/2+.72,-np.pi/2],
        robot_radius_m=.6,kitchen_receptacle=('counter_right_main_group' if 'counter_right_main_group' in kitchen_counters else kitchen_counters[0]),
        simulation_tested=False)
    validate_dining_layout(metadata)
    return copy.deepcopy(layout),metadata


def validate_dining_layout(meta):
    """Conservative planar clearance, not an IK or full-robot physics test."""
    r=meta['robot_radius_m']; bounds=meta['room_bounds']['dining_room']
    tx,ty=meta['table']['center'];w,d,_=meta['table']['size']
    obstacles=[[tx-w/2,ty-d/2,tx+w/2,ty+d/2]]+meta['chairs']
    assert meta['doorway']['clear_width_m']>2*r
    for box in obstacles:
        assert bounds[0]<box[0]<box[2]<bounds[2]
        assert bounds[1]<box[1]<box[3]<bounds[3]
    for i,a in enumerate(obstacles):
        for b in obstacles[i+1:]:
            assert a[2]<=b[0] or b[2]<=a[0] or a[3]<=b[1] or b[3]<=a[1]
    # A reserved route through the doorway to the clear north side of the table.
    start=np.array(meta['doorway']['center'])
    goal=np.array(meta['manipulation_stance'][:2])
    for p in np.linspace(start,goal,101):
        assert bounds[0]+r<p[0]<bounds[2]-r and p[1]>bounds[1]+r
        for box in obstacles:
            closest=np.clip(p,box[:2],box[2:])
            assert np.linalg.norm(p-closest)>=r
    return True


def build_three_room_layout(rng=None, base_layout=1):
    """Kitchen -> dining -> living, with seeded furniture and open portals."""
    rng = np.random.default_rng() if rng is None else rng
    layout, meta = build_dining_layout(rng, base_layout)
    bottom = meta['room_bounds']['dining_room'][1]
    edge=meta['room_bounds']['dining_room'][2]
    front=meta['room_bounds']['dining_room'][3]
    east = edge + float(rng.uniform(4.5, 5.1))
    portal_y = front-1.05
    portal_width = float(rng.uniform(1.65, 1.85))
    low, high = portal_y-portal_width/2, portal_y+portal_width/2
    walls = layout['dining_room']['walls']
    walls[:] = [w for w in walls if w['name'] != 'right']
    for name, a, b in [('east_south', bottom, low), ('east_north', high, front)]:
        walls.append(dict(name=name,type='wall',pos=[edge,(a+b)/2,1.5],
                          size=[(b-a)/2,1.5,.02],wall_side='right'))
    cx = (edge+east)/2
    sofa_width = float(rng.uniform(2.1,2.7))
    sofa_y = bottom+.65
    coffee_y = bottom+1.9
    furniture = [
        dict(name='sofa_seat',type='box',pos=[cx,sofa_y,.30],size=[sofa_width,.85,.60]),
        dict(name='sofa_back',type='box',pos=[cx,sofa_y-.35,.75],size=[sofa_width,.15,.60]),
        dict(name='coffee_table',type='counter',pos=[cx,coffee_y,.23],
             size=[float(rng.uniform(1.,1.35)),.65,.46],hollow=[True,True],top_thickness=.04),
        dict(name='side_table',type='box',pos=[east-.55,front-1.1,.39],size=[.65,.65,.78]),
    ]
    for sign in (-1,1):
        furniture.append(dict(name='sofa_arm_'+str(sign),type='box',
            pos=[cx+sign*(sofa_width/2-.09),sofa_y,.65],size=[.18,.85,.30]))
    w = furniture[2]['size'][0]
    for i,(dx,dy) in enumerate(((-1,-1),(-1,1),(1,-1),(1,1))):
        furniture.append(dict(name=f'coffee_leg_{i}',type='box',
            pos=[cx+dx*(w/2-.07),coffee_y+dy*.255,.21],size=[.07,.07,.42]))
    layout['living_room'] = dict(
        floor=[dict(name='floor',type='floor',pos=[cx,(bottom+front)/2,0],
                    size=[(east-edge)/2,(front-bottom)/2,.02])],
        walls=[dict(name='north',type='wall',pos=[cx,front,1.5],size=[(east-edge)/2,1.5,.02],wall_side='back'),
               dict(name='south',type='wall',pos=[cx,bottom,1.5],size=[(east-edge)/2,1.5,.02],wall_side='front'),
               dict(name='east',type='wall',pos=[east,(bottom+front)/2,1.5],size=[(front-bottom)/2,1.5,.02],wall_side='right')],
        furniture=furniture)
    meta.update(layout_id=1003 if base_layout == 1 else 1100+base_layout)
    meta['room_bounds']['living_room']=[edge,bottom,east,front]
    meta['portals']=[dict(meta['doorway'],rooms=['kitchen','dining_room']),
        dict(center=[edge,portal_y],nominal_width_m=portal_width,
             clear_width_m=portal_width-.08,rooms=['dining_room','living_room'])]
    meta['living_receptacles']=['coffee_table_living_room','side_table_living_room']
    meta['living_stance']=[cx,portal_y,-np.pi/2]
    meta['connecting_route']=[meta['doorway']['center'],
        [meta['doorway']['center'][0],portal_y],[cx,portal_y]]
    add_door_frames(layout,meta)
    validate_three_room_layout(layout,meta)
    return layout,meta



def add_door_frames(layout,meta):
    """Static collision-bearing jambs and headers; passages remain open."""
    frames=[]
    thickness,depth,height,header=.10,.18,2.35,.14
    for index,portal in enumerate(meta['portals']):
        x,y=portal['center'];width=portal['nominal_width_m']
        along_x=index==0
        # Native walls sit 2 cm outside the room boundary.
        if along_x:y-=.02
        else:x+=.02
        for side,sign in [('left',-1),('right',1)]:
            offset=sign*(width/2-thickness/2)
            pos=[x+offset if along_x else x,y if along_x else y+offset,height/2]
            size=[thickness,depth,height] if along_x else [depth,thickness,height]
            frames.append(dict(name=f'doorframe_{index}_{side}',type='box',pos=pos,size=size))
        for name,z,h in [('header',height+header/2,header),
                         ('infill',(height+header+3)/2,3-height-header)]:
            size=[width,depth,h] if along_x else [depth,width,h]
            frames.append(dict(name=f'doorframe_{index}_{name}',type='box',pos=[x,y,z],size=size))
        portal.update(clear_width_m=width-2*thickness,clear_height_m=height,
                      frame_depth_m=depth,door_leaf=False)
    layout['dining_room']['door_frames']=frames
    meta['doorway'].update({k:v for k,v in meta['portals'][0].items() if k!='rooms'})


def validate_three_room_layout(layout,meta):
    """Check a radius-.6 disc along the connecting route against furniture/jambs."""
    validate_dining_layout(meta)
    r=meta['robot_radius_m']
    obstacles=[]
    for group in ('dining_room','living_room'):
        for f in layout[group]['furniture']:
            p=np.array(f['pos'][:2]);half=np.array(f['size'][:2])/2
            obstacles.append(np.r_[p-half,p+half])
    for f in layout['dining_room'].get('door_frames',[]):
        if f['name'].endswith(('left','right')):
            p=np.array(f['pos'][:2]);half=np.array(f['size'][:2])/2
            obstacles.append(np.r_[p-half,p+half])
    for w in layout['dining_room']['walls']:
        if w['name'].startswith('east_'):
            x,y,_=w['pos'];half=w['size'][0]+.04
            obstacles.append(np.array([x-.04,y-half,x+.04,y+half]))
    for portal in meta['portals']:
        assert portal['clear_width_m']>2*r
    boxes=np.asarray(obstacles)
    for a,b in zip(meta['connecting_route'],meta['connecting_route'][1:]):
        points=np.linspace(a,b,101)[:,None,:]
        nearest=np.clip(points,boxes[None,:,:2],boxes[None,:,2:])
        assert np.all(np.linalg.norm(points-nearest,axis=-1)>=r)
    bounds=meta['room_bounds']['living_room']
    for f in layout['living_room']['furniture']:
        p=np.array(f['pos'][:2]);half=np.array(f['size'][:2])/2
        assert np.all(p-half>np.array(bounds[:2]))
        assert np.all(p+half<np.array(bounds[2:]))
    return True
