# Explicit appliance control-panel conventions

User decision, November 10, 2026: provide a documented appliance convention
for the open-source baseline rather than changing existing scene textures.
This is supplied appliance-manual knowledge, not visual recognition of an
unseen burner-to-knob association. It must be disclosed in evaluation results.

## Front-control seven-knob stove: `front_control_seven_v1`

Looking at the appliance from the front, count the complete row left to right:

| Knob | Function |
|---|---|
| 1 | Oven temperature — not a cooktop burner |
| 2 | Oven timer — not a cooktop burner |
| 3 | Front-left burner |
| 4 | Rear-left burner |
| 5 | Center burner |
| 6 | Rear-right burner |
| 7 | Front-right burner |

This convention documents the authored RoboCasa Stove002/front-control panel
used in the current baseline. It is not universal across stoves. The mapping
was checked against that appliance's authored control definitions; that
offline documentation step does not supply live joint values or coordinates.

Execution must explicitly opt into this convention for a compatible appliance.
Molmo proposes all seven knob centers from the onboard RGB image. The adapter
orders that observed row and selects the documented ordinal. Onboard depth
then locates the control in the robot frame, and the existing geometry,
M2T2 grasp and physical rotation routines execute the command. ON/OFF angle
feedback still comes from observed knob geometry.

A partial/occluded row does not justify counting from the first visible knob.
The adapter returns no point for an incomplete or inconsistent row; the caller
can inspect a different view. It never falls back to a simulator object ID,
joint index, target pose or burner state. Other appliance controls, including
microwave buttons, retain their existing visual pointing behavior.

```python
from cross_episode_sim.perception.stove_control_convention import DocumentedStovePointer

# Explicit application configuration, after choosing the compatible manual.
backend.pointer = DocumentedStovePointer(backend.pointer)
```

Status: documented selection adapter implemented; physical qualification and
full stove-task integration are pending. Do not report a stove-task pass from
correct ordinal selection alone.
