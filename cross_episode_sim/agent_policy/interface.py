"""Shared tool contract; no object truth or oracle planning API."""
import json
import math

TOOLS = {
    'move_joints': ('arm: seven absolute joint angles in radians; gripper: fraction closed; seconds: duration',
        {'arm': {'type':'array','items':{'type':'number'},'minItems':7,'maxItems':7},
         'gripper':{'type':'number','minimum':0,'maximum':1},'seconds':{'type':'number','minimum':0.02,'maximum':10}}),
    'move_base': ('forward, left in meters and yaw in radians RELATIVE to current base; arm posture retained; seconds: duration. No automatic navigation or obstacle avoidance.',
        {**{k:{'type':'number'} for k in ('forward','left','yaw')},'seconds':{'type':'number','minimum':0.02,'maximum':10}}),
    'move_ee': ('Absolute TCP position in planar robot base frame (meters), quaternion wxyz; gripper fraction closed; duration. Native inverse kinematics, no oracle grasp selection.',
        {'position':{'type':'array','items':{'type':'number'},'minItems':3,'maxItems':3},
         'quaternion_wxyz':{'type':'array','items':{'type':'number'},'minItems':4,'maxItems':4},
         'gripper':{'type':'number','minimum':0,'maximum':1},'seconds':{'type':'number','minimum':0.02,'maximum':10}}),
    'wait': ('Hold current commands and advance physical simulation; useful for heating.',
        {'seconds':{'type':'number','minimum':0.02,'maximum':30}}),
    'observe': ('Read fresh navigation/wrist RGB and measured robot state; no simulation steps.', {}),
    'history_frame': ('Inspect an allowed historical video at selected timestamps; IDs come from catalog. Frames may show a different scene, not the current object positions.',
        {'video_id':{'type':'string'},'seconds':{'type':'array','items':{'type':'number','minimum':0},'minItems':1,'maxItems':12}}),
    'finish': ('Declare completion; only environment determines success.', {})
}

def schema(memory):
    variants=[]
    for name,(description,props) in TOOLS.items():
        if name=='history_frame' and memory=='none': continue
        variants.append({'type':'object','properties':{'name':{'type':'string','enum':[name]},
            'arguments':{'type':'object','description':description,'properties':props,'required':list(props),'additionalProperties':False}},
            'required':['name','arguments'],'additionalProperties':False})
    return {'type':'object','properties':{'decision':{'anyOf':variants}},'required':['decision'],'additionalProperties':False}

def validate(decision,memory):
    import jsonschema
    jsonschema.validate({'decision':decision},schema(memory))
    def finite(v):
        if isinstance(v,float) and not math.isfinite(v):raise ValueError('Non-finite argument')
        if isinstance(v,dict):
            for x in v.values():finite(x)
        if isinstance(v,list):
            for x in v:finite(x)
    finite(decision)
    return decision

INSTRUCTIONS='''You control a mobile Franka in a physical simulator. Solve the task from the two onboard camera views, measured robot state, and allowed history only. Choose one tool per decision. Observe results and correct failures. There are no object IDs, object poses, oracle grasps or navigation plans. Joint angles are radians. Base motion is relative: forward along current heading, left perpendicular, positive yaw counterclockwise. TCP pose uses the planar base frame: origin at base world x/y and z=0, z up. Quaternion order w,x,y,z. Gripper 0=open, 1=closed. Commands interpolate at 50Hz through the MolmoSpaces controller, with collisions active. Move in small increments near contact; no automatic collision planning. Retain grasp while carrying. A failed tool may have moved partially; inspect feedback before retrying. Finish only when the task appears complete. Historical videos are references, never current scene truth. Return the requested JSON schema. Do not use external tools or access files outside supplied observations.'''
