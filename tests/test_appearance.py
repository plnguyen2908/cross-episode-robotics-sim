import xml.etree.ElementTree as ET
from cross_episode_sim.generation.appearance import apply_appearance


def test_recolor_preserves_physics_and_semantic_basket(tmp_path):
 p=tmp_path/'scene.xml';p.write_text('''<mujoco><worldbody>
 <body name="north_living_room_main"><geom name="wall" size="1 1 .1"/></body>
 <body name="cup"><geom name="cup" material="old" mass=".2"/></body>
 <body name="basket"><geom name="basket" rgba="1 0 0 1"/></body>
 </worldbody></mujoco>''')
 apply_appearance(p,dict(wall_rgb=[.1,.2,.3],object_rgb=[.3,.2,.1]),[
 dict(body='cup',task_object=True,category='cup'),dict(body='basket',task_object=True,category='basket')])
 r=ET.parse(p).getroot();cup=r.find(".//geom[@name='cup']")
 assert cup.get('mass')=='.2' and cup.get('material') is None
 assert r.find(".//geom[@name='basket']").get('rgba')=='1 0 0 1'


def test_theme_preserves_physics_and_task_colors(tmp_path):
 import xml.etree.ElementTree as ET
 p=tmp_path/'scene.xml'
 p.write_text('''<mujoco><asset><material name="shared" rgba="1 1 1 1"/></asset><worldbody>
 <body name="0_living_room_main_0"><geom name="wall" type="box" size="1 .1 2" contype="1"/></body>
 <body name="floor_living_room_main"><geom name="floor" size="2 2 .1" material="shared"/></body>
 <body name="furn_living_sofa_1_1"><geom name="seat" size="1 .5 .5" material="shared"/></body>
 <body name="target"><geom name="target_geom" size=".1 .1 .1" rgba="1 0 0 1"/></body>
 </worldbody></mujoco>''')
 before=ET.parse(p)
 result=apply_appearance(p,{'preset':'cool_daylight'},[{'body':'target','task_object':True,'color':'red'}])
 after=ET.parse(p)
 assert after.find(".//geom[@name='target_geom']").get('rgba')=='1 0 0 1'
 assert result['object_rgb']['geoms']==[]
 for old,new in zip(before.findall('.//geom'),after.findall('.//geom')):
  for key in ['name','type','size','contype']:assert old.get(key)==new.get(key)
 assert after.find(".//material[@name='shared']").get('rgba')=='1 1 1 1'
 assert after.find('.//light') is not None


def test_hidden_bbox_stays_hidden(tmp_path):
 import xml.etree.ElementTree as ET
 p=tmp_path/'scene.xml';p.write_text('<mujoco><worldbody><body name="cup"><geom name="mesh" rgba="1 1 1 1"/><geom name="bbox" rgba="0 0 0 0" group="1"/></body></worldbody></mujoco>')
 apply_appearance(p,{'object_rgb':[.2,.4,.6]},[{'body':'cup','task_object':True}])
 assert ET.parse(p).find(".//geom[@name='bbox']").get('rgba')=='0 0 0 0'
