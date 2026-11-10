"""Room/portal exploration decisions from an observed topology provider.

No simulator access. Room masks must include estimated occluded navigable area,
not merely observed free cells. Boundary validity and portal associations come
from perception; missing evidence never means complete coverage.
"""
from dataclasses import dataclass
from typing import FrozenSet

Cell = tuple[int, int]


@dataclass(frozen=True)
class RoomEvidence:
    room: str
    estimated_navigable: FrozenSet[Cell]
    observed_navigable: FrozenSet[Cell]
    boundary_supported: bool = False

    @property
    def coverage(self):
        if not self.boundary_supported or not self.estimated_navigable:
            return None
        return len(self.estimated_navigable & self.observed_navigable)/len(self.estimated_navigable)


@dataclass(frozen=True)
class PortalEvidence:
    portal: str
    source: str
    destination: str | None
    closed: bool
    # True only after observation-based association, never geometric guessing.
    association_confirmed: bool = False
    reachable: bool = True


@dataclass(frozen=True)
class ExplorationDecision:
    action: str
    target: str | None
    reason: str


class RoomExplorationPolicy:
    def __init__(self, coverage_target=.9):
        if not 0 < coverage_target <= 1:
            raise ValueError('Coverage target must be in (0, 1]')
        self.coverage_target=coverage_target

    def complete(self, room):
        value=room.coverage
        return value is not None and value >= self.coverage_target

    def decide(self, current_room, rooms, portals, *, reachable_frontier=True, open_route_available=False):
        """Decide only; navigate/open skills retain their physical checks.

        Call again after every map update or physical door operation. A failed
        opening must not change portal state or room coverage.
        """
        rooms={r.room:r for r in rooms}
        room=rooms.get(current_room)
        if room is None:
            return ExplorationDecision('observe',None,'current_room_unassociated')
        if not self.complete(room) and reachable_frontier:
            return ExplorationDecision('explore_room',current_room,'coverage_below_target_or_unknown')
        pending=[];remote_pending=[]
        for portal in portals:
            destination=rooms.get(portal.destination) if portal.association_confirmed else None
            if destination is not None and self.complete(destination):
                continue
            if portal.source != current_room:
                remote_pending.append(portal)
            else:
                pending.append(portal)
        reachable=[p for p in pending if p.reachable]
        opened=[p for p in reachable if not p.closed]
        if opened:
            portal=min(opened,key=lambda p:p.portal)
            return ExplorationDecision('navigate_portal',portal.portal,'open_route_to_unexplored_destination')
        if open_route_available:
            return ExplorationDecision('explore_room',current_room,'open_frontier_before_closed_door')
        # Prefer an already-open route over manipulation; stable ordering.
        if reachable:
            portal=min(reachable,key=lambda p:(p.closed,p.portal))
            return ExplorationDecision('open_door' if portal.closed else 'navigate_portal',portal.portal,
                'destination_unknown' if not portal.association_confirmed else 'destination_below_target')
        if pending:
            return ExplorationDecision('blocked',None,'unexplored_portals_unreachable')
        if remote_pending:
            portal=min(remote_pending,key=lambda p:(not p.reachable,p.portal))
            if not portal.reachable:
                return ExplorationDecision('blocked',None,'unexplored_portals_unreachable')
            return ExplorationDecision('navigate_room',portal.source,'pending_portal_in_other_room')
        incomplete=[r.room for r in rooms.values() if not self.complete(r)]
        if incomplete:
            other=[r for r in incomplete if r!=current_room]
            if other:
                return ExplorationDecision('navigate_room',sorted(other)[0],'known_room_below_target')
            return ExplorationDecision('blocked',current_room,'coverage_incomplete_without_reachable_frontier')
        return ExplorationDecision('observed_topology_complete',None,'all_estimated_rooms_at_target_no_pending_portal')
