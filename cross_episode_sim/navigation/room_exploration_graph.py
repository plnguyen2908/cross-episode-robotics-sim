"""Observed room/portal exploration state, separate from successful motion routes.

No simulator geometry, room count or object identities are accepted here.
Unknown destinations remain unresolved until observation associates them.
"""
from collections import deque
from dataclasses import asdict, replace
from .room_exploration_policy import RoomExplorationPolicy, ExplorationDecision


class RoomExplorationGraph:
    def __init__(self, coverage_target=.9):
        self.policy=RoomExplorationPolicy(coverage_target)
        self.rooms={};self.portals={};self.visits={};self.crossings={};self.failures={}

    def observe(self, rooms, portals, current=None):
        self.rooms.update((r.room,r) for r in rooms)
        for p in portals:
            old=self.portals.get(p.portal)
            # Physical opening/crossing is not undone by a stale handle detector.
            if old is not None and self.crossings.get(p.portal):
                p=replace(p,source=old.source,destination=old.destination,
                          association_confirmed=True,closed=False)
            if p.portal in self.failures:p=replace(p,reachable=False)
            self.portals[p.portal]=p
        if current is not None:self.visits[current]=self.visits.get(current,0)+1

    def crossed(self, portal, source, destination):
        if source==destination:raise ValueError('A crossing needs distinct observed rooms')
        p=self.portals[portal]
        self.portals[portal]=replace(p,source=source,destination=destination,
                                    closed=False,association_confirmed=True,reachable=True)
        self.crossings[portal]=self.crossings.get(portal,0)+1
        self.failures.pop(portal,None)

    def failed(self, portal, reason):
        self.failures[portal]=reason
        self.portals[portal]=replace(self.portals[portal],reachable=False)

    def retire_duplicate(self,portal):
        self.portals.pop(portal,None)
        self.failures.pop(portal,None)

    def _routes(self, current):
        """BFS only over observed, open, reachable room connections."""
        routes={current:[]};queue=deque([current])
        while queue:
            here=queue.popleft()
            for p in sorted(self.portals.values(),key=lambda p:p.portal):
                if p.closed or not p.reachable or not p.association_confirmed:continue
                there=p.destination if p.source==here else p.source if p.destination==here else None
                if there is None or there in routes:continue
                routes[there]=routes[here]+[p.portal];queue.append(there)
        return routes

    def decide(self,current,*,reachable_frontier=True,open_route_available=False):
        # Finish reachable frontiers here before returning to another known
        # room. Otherwise two incomplete rooms can preempt one another forever.
        if current in self.rooms and not self.policy.complete(self.rooms[current]) and reachable_frontier:
            return ExplorationDecision('explore_room',current,'coverage_below_target_or_unknown')
        routes=self._routes(current)
        # Open routes to uncovered rooms outrank closed doors, even if the
        # closed door is nearer or was detected earlier.
        targets=[(len(path),name,path) for name,path in routes.items() if path and
                 name in self.rooms and not self.policy.complete(self.rooms[name])]
        if targets:
            _,name,path=min(targets)
            return ExplorationDecision('navigate_portal',path[0],f'open_route_to_unexplored_room:{name}')
        # View each undirected edge from the current room.
        local=[]
        for p in self.portals.values():
            if p.association_confirmed and p.destination==current:
                p=replace(p,source=current,destination=p.source)
            local.append(p)
        decision=self.policy.decide(current,list(self.rooms.values()),local,
            reachable_frontier=reachable_frontier,open_route_available=open_route_available)
        if decision.action=='navigate_room':
            path=routes.get(decision.target)
            if path:return ExplorationDecision('navigate_portal',path[0],decision.reason)
            return ExplorationDecision('blocked',decision.target,'no_observed_open_room_route')
        return decision

    def snapshot(self):
        return dict(schema_version=1,kind='observed_room_exploration_graph',
            coverage_target=self.policy.coverage_target,
            nodes=[dict(room=r.room,coverage=r.coverage,boundary_supported=r.boundary_supported,
                        status='covered' if self.policy.complete(r) else 'unexplored',
                        observations=self.visits.get(r.room,0)) for r in self.rooms.values()],
            edges=[dict(**asdict(p),crossings=self.crossings.get(p.portal,0),
                        last_failure=self.failures.get(p.portal)) for p in self.portals.values()])
