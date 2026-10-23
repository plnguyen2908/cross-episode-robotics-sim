import json
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
import pytest
from cross_episode_sim.generation.room_graph import extend_room_graph,settings
from cross_episode_sim.generation.room_routes import room_door_states
from cross_episode_sim.generation.room_door_bindings import room_door_bindings
from cross_episode_sim.tasks.room_doors import install_room_doors


def scene(tmp_path):
 p=tmp_path/'scene.xml';p.write_text('''<mujoco><worldbody>
 <body name="east_living_room_main" pos="5 0 1.5"><geom type="box" size=".06 4 1.5"/></body>
 <body name="floor_living_room_main" pos="3 0 -.025"><geom type="box" size="2 4 .025"/></body>
 </worldbody></mujoco>''');return p


@pytest.mark.parametrize('count,arrangement',[(3,'chain'),(4,'chain'),(8,'chain'),(8,'branch')])
def test_connected_geometry_and_dynamic_doors(tmp_path,count,arrangement):
 p=scene(tmp_path);g=extend_room_graph(p,dict(room_count=count,room_arrangement=arrangement))
 assert len(g['rooms'])==count and len(g['connections'])==count
 assert json.loads(p.with_name('room_graph.json').read_text())==g
 spec=mujoco.MjSpec.from_file(str(p));m=spec.compile();d=mujoco.MjData(m);mujoco.mj_forward(m,d)
 records=install_room_doors(spec,{f'room_door_{i}':'open' for i in range(3,count)},model=m,data=d)
 m=spec.compile();assert len(records)==count-3
 assert len(room_door_bindings(m))==2*(count-3)
 # Probe each doorway centre below its lintel: no static wall blocks it.
 for edge in g['connections'][3:]:
  pnt=np.array([*edge['center'],1.7]);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
  for gid in range(m.ngeom):
   if 'doorframe' in m.geom(gid).name or 'room_door' in m.geom(gid).name:continue
   local=(pnt-d.geom_xpos[gid])@d.geom_xmat[gid].reshape(3,3)
   assert not np.all(np.abs(local)<m.geom_size[gid])


def test_states_scale_and_validate():
 assert len(room_door_states(dict(room_count=8)))==8
 assert room_door_states(dict(room_count=5,room_doors={'room_door_4':'closed'}))['room_door_4']=='closed'
 for params in [dict(room_count=2),dict(room_count=9),dict(room_arrangement='bad'),dict(extra_room_width=float('nan'))]:
  with pytest.raises(ValueError):settings(params)


def test_extension_inherits_existing_house_finishes(tmp_path):
 p=scene(tmp_path);tree=ET.parse(p);root=tree.getroot()
 assets=ET.SubElement(root,'asset')
 ET.SubElement(assets,'material',name='existing_floor',rgba='.4 .4 .4 1')
 ET.SubElement(assets,'material',name='existing_wall',rgba='.6 .3 .2 1')
 root.find("worldbody/body[@name='floor_living_room_main']/geom").set('material','existing_floor')
 wall=ET.SubElement(root.find('worldbody'),'body',name='north_living_room_main',pos='3 4 1.5')
 ET.SubElement(wall,'geom',type='box',size='2 .02 1.5',material='existing_wall')
 tree.write(p)
 extend_room_graph(p,dict(room_count=4))
 result=ET.parse(p).getroot()
 assert result.find("worldbody/body[@name='floor_extra_1_main']/geom").get('material')=='existing_floor'
 for body in result.find('worldbody').findall('body'):
  if '_extra_' in body.get('name','') and body.get('name','').startswith(('east_','west_','north_','south_')):
   assert all(g.get('material')=='existing_wall' for g in body.findall('geom'))
