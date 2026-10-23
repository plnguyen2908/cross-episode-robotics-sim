# Cabinet motion after release — 2026-10-14

The unexpected swing is caused by door-panel contact with the cabinet housing
near closed, rather than simply residual momentum. This was reproduced without
robot contact and with zero initial joint velocities. No production geometry,
collision settings, damping, or friction were changed during this inspection.

## Evidence

The latest failed trace has door-panel contacts against the cabinet top, bottom,
and hinge-side wall while the robot withdraws and navigates away. Penetration is
approximately 2.5–2.7 mm at 15–30 degrees. From about 84 to 93 simulated seconds,
the released door accelerates from roughly 0.6 to 1.1 degrees/second while these
contacts persist. Saved trace velocities were estimated from adjacent poses;
contact forces were reconstructed, not recorded live.

To establish causality, a separate scratch simulation initialized the door at
15, 30, or 60 degrees, zeroed velocities, retained the initial robot pose away
from the door, and advanced normal physics for 10 seconds. A diagnostic-only
control disabled the housing's collision geoms. This control is not a proposed
production fix and was never used to qualify a task.

| Initial angle | Wider cabinet, normal collisions | Diagnostic without housing collisions |
| --- | --- | --- |
| 15 degrees | approximately 4.65 degrees | 15 degrees |
| 30 degrees | approximately 39 degrees | 30 degrees |
| 60 degrees | 60 degrees | 60 degrees |

The previous narrower composite cabinet reproduces the same pattern at 15 and
30 degrees. The original successful atomic demo released near 59.99 degrees,
then near 89.08 degrees. This explains why its partial opening stayed still:
the 60-degree pose has no door/housing contact in the test. The newer 15-degree
strokes introduce releases in the problematic angular region.

The authored scene XML, before the robot is added, also produces the same
door/housing contacts at 30 degrees. The joint remains passive, with damping 2,
zero stiffness, and zero friction loss. The saved failure state has no applied
hinge force or external wrench on the door subtree. These observations identify
the geometry/contact issue in this generated cabinet; they do not establish
that every RoboCasa cabinet or layout behaves this way.

## Implication

The previous explanation blaming free swinging and the subsequent suggestion
that withdrawal was necessarily pushing the door were not established by the
logs. The controlled test identifies housing contact as sufficient to drive the
observed motion. Fix and validate the hinge/panel/housing clearance in scene
setup, retaining real collisions, before retesting one cup and one bowl. Merely
increasing damping or retrying the grasp does not address that geometry defect.

Evidence: [normal and diagnostic simulations](../runs/cabinet_drift_inspection_20261014/door_counterfactual_results.json),
[60-degree controls](../runs/cabinet_drift_inspection_20261014/door_counterfactual_60_results.json),
[trace inspection](../runs/cabinet_drift_inspection_20261014/door_drift_analysis.jsonl).

## Clearance correction

Scene construction now checks each active native single-door cabinet over 181
angles across its joint range. If its swing intersects surrounding geometry, it
tries a small outward mounting adjustment of the complete door, hinge, and handle
assembly, failing setup if no clear placement is found within 10 mm. The affected
cabinet needs 4 mm. No collision shapes, masks, damping, friction, or joint limits
are changed. Corrections are recorded in the scene-builder metadata.

The compiled corrected cabinet stayed at 15, 30, 60, and 89 degrees for ten
simulated seconds from rest with normal collisions enabled:
[physical checks](../runs/qualify_cabinet_storage_2_928zdl/passive_hinge_checks.json).
Tests cover unchanged collision/dynamics parameters, repeat application, passive
stability, and existing door/controller behavior (41 tests passed).
The robot's opening passed in `runs/qualify_cabinet_storage_2_928zdl`:
released intermediate angles were 18.18, 29.65, 44.57, 59.48, and 74.39 degrees,
followed by a released full opening of 89.30 degrees. Execution advanced to
call 3/12, navigation to the bowl. The two-object storage/closing sequence is
still in progress and is not yet qualified.
