"""Closed, versioned atomic vocabulary. Task recipes cannot extend this catalog."""
from copy import deepcopy


def schema(required, **properties):
    return dict(type='object', additionalProperties=False, required=list(required), properties={
        key: dict(type='string', **value) for key, value in properties.items()})


def target_schema(**extra):
    return schema(['target', *extra], target=dict(description='Object or fixture type/name exposed by this scene.'), **extra)


CATALOG = {
    'navigate': ('Approach target when destination is omitted. With destination, carry the already-held target there. Never grasp or release implicitly.',
                 schema(['target'], target={}, destination={}, dock=dict(default='0', enum=[str(i) for i in range(14)]))),
    'pick': ('Grasp and lift a visible target from the current base pose with an empty hand. No previous navigation is required. Use a top-down grasp; no hidden base navigation.', target_schema()),
    'place': ('Place the held object on or inside receptacle, release and withdraw. region optionally selects a supported area. No base navigation.',
              schema(['receptacle'], receptacle={}, region=dict(default='default'))),
    'open_door': ('Open the passive door to near its physical opening limit, release and withdraw. May reach from the released handle directly to the panel and push with coordinated local base/arm motion. If that contact path is blocked, may reposition locally and regrasp the same door. Requires a free hand and reachable handle. A partial opening is not success; a blocked or unreachable full stroke returns failure.', target_schema()),
    'close_door': ('Close the passive door from its current angle, release and withdraw. May release and reposition the empty robot locally to regrasp the same door. Requires clearance and a free hand; travel to other objects or rooms remains a navigate call.', target_schema()),
    'open_drawer': ('Grasp the drawer handle from above, pull along its slide axis to near its physical opening limit, release and withdraw. May translate the base locally along the slide axis while retaining the handle grasp. A partial opening is not success. Do not rotate the drawer or base.', target_schema()),
    'close_drawer': ('Grasp the drawer handle from above, push along its slide axis, release and withdraw. May translate the base locally along that axis while retaining the grasp; no base rotation or travel to another target.', target_schema()),
    'open_lid': ('Open a hinged lid or lift off a removable lid. A removable lid remains held; call place to free the hand before another manipulation, then pick that lid before close_lid. A hinged lid is released. Scene target metadata specifies the type.', target_schema()),
    'close_lid': ('Close a hinged lid or reseat the already-held matching removable lid, then release and withdraw. For a removable lid, call pick first if it was placed down. Requires physical alignment and clearance; does not implicitly retrieve the lid.', target_schema()),
    'push_button': ('Press the selected button through physical contact and withdraw. Press success does not imply completion of the appliance task.', target_schema()),
    'turn_lever': ('Grasp and turn a rotary lever to the requested state, then release. Appliance-specific state meanings are supplied by the scene.', target_schema(state=dict(enum=['on', 'off']))),
    'push_lever': ('Push the selected sliding lever to activation, then withdraw. Verify contact and measured travel.', target_schema()),
    'rotate_knob': ('Grasp and rotate the selected knob to the requested state, then release. No implicit navigation or object placement.', target_schema(state=dict(enum=['on', 'off']))),
    'insert': ('Align the already-held object with receptacle, insert, release and withdraw. No pickup or base navigation.', schema(['receptacle'], receptacle={})),
    'wait': ('Advance simulation by duration_seconds while preserving the current robot commands. Existing environment processes such as heating may progress; this does not start an appliance or guarantee readiness. Holding an object is allowed.',
             dict(type='object', additionalProperties=False, required=['duration_seconds'], properties=dict(duration_seconds=dict(type='number', exclusiveMinimum=0, maximum=600)))),
    'slide_oven_rack': ('Grasp the oven rack, slide it out or in, release and withdraw. The oven door must physically clear its path.', target_schema(state=dict(enum=['out', 'in']))),
}
SKILL_NAMES = tuple(CATALOG)
CATALOG_VERSION = 'fixed-atomic-v1'


def list_skills():
    return [dict(name=name, description=description, parameters=deepcopy(parameters))
            for name, (description, parameters) in CATALOG.items()]


def require_fixed_catalog(entries):
    """Never expose a legacy task-specific skill catalog to a new planner."""
    unexpected = {entry['name'] for entry in entries} - set(SKILL_NAMES)
    if unexpected:
        raise ValueError('Legacy task-specific skill interface is retired from the fixed-atomic benchmark: '
                         + ', '.join(sorted(unexpected)))
    return entries
