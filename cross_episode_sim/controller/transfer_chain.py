"""Small persistent two-receptacle reorder protocol, independent of the simulator.

The executor supplies physical work, observed assignments, and harness-only
interventions. A failed operation aborts the chain; expected state is never used
as proof of a successful transfer.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Transfer:
    object_name: str
    source: str
    destination: str


class TwoReceptacleChain:
    def __init__(self, receptacles, objects, executor):
        self.receptacles = tuple(receptacles)
        self.objects = tuple(objects)
        if len(set(self.receptacles)) != 2 or len(self.receptacles) != 2:
            raise ValueError("Exactly two distinct receptacles are required")
        if not self.objects or len(set(self.objects)) != len(self.objects):
            raise ValueError("Tracked objects must be nonempty and unique")
        self.executor = executor
        self.snapshots = []
        self.events = []
        self.manipulated = set()

    def validate_assignment(self, assignment):
        if set(assignment) != set(self.objects):
            raise ValueError("Observation must account for every tracked object")
        if any(value not in self.receptacles for value in assignment.values()):
            raise ValueError("Object has missing, ambiguous, or out-of-pair support")
        return dict(assignment)

    def explore(self, label=None):
        assignment = self.validate_assignment(self.executor.observe(self.receptacles))
        snapshot = {"episode": len(self.events), "assignment": assignment, "label": label}
        self.snapshots.append(snapshot)
        self.events.append({"kind": "explore", **snapshot})
        return len(self.snapshots) - 1

    def work(self, object_name, destination):
        current = self.validate_assignment(self.executor.assignment())
        if object_name not in self.objects or destination not in self.receptacles:
            raise ValueError("Transfer references an object or receptacle outside the pair")
        source = current[object_name]
        if source == destination:
            raise ValueError("Work must require a physical transfer")
        move = Transfer(object_name, source, destination)
        self.executor.transfer(move)
        after = self.validate_assignment(self.executor.assignment())
        expected = current | {object_name: destination}
        if after != expected:
            raise RuntimeError("Physical transfer did not produce the expected supported assignment")
        # Only a completed physical transfer can admit an object to changes.
        remember = getattr(self.executor, 'remember_successful_transfer', None)
        if remember is not None:
            remember(move)
        self.manipulated.add(object_name)
        self.events.append({"kind": "work", **vars(move), "assignment": after})

    def work_batch(self, moves):
        """Execute consecutive transfers within one physical access cycle."""
        moves = tuple(moves)
        if not moves:
            raise ValueError('A work batch needs at least one transfer')
        with self.executor.transfer_group(len(moves)):
            for obj, destination in moves:
                self.work(obj, destination)

    def intervene(self, moves):
        current = self.validate_assignment(self.executor.assignment())
        moves = dict(moves)
        if not moves or any(o not in self.objects or r not in self.receptacles for o, r in moves.items()):
            raise ValueError("Intervention must move tracked objects within the pair")
        if any(current[o] == r for o, r in moves.items()):
            raise ValueError("Every intervention move must change receptacle")
        if any(obj not in self.manipulated for obj in moves):
            raise ValueError("Dynamic changes require a successful physical transfer of each object")
        self.executor.intervene(moves)
        after = self.validate_assignment(self.executor.assignment())
        if after != current | moves:
            raise RuntimeError("Intervention placement did not settle on the requested receptacles")
        self.events.append({"kind": "intervention", "observed": False, "moves": [
            {"object_name": o, "source": current[o], "destination": r} for o, r in moves.items()
        ]})

    def restore(self, snapshot_index):
        snapshot = self.snapshots[snapshot_index]
        current = self.validate_assignment(self.executor.assignment())
        target = snapshot["assignment"]
        displaced = [o for o in self.objects if current[o] != target[o]]
        if not displaced:
            raise ValueError("Restoration is already solved; refusing a trivial success")
        # Empty the fridge before inserting another object into its limited slots.
        displaced.sort(key=lambda obj: target[obj] != self.receptacles[0])
        self.work_batch((obj, target[obj]) for obj in displaced)
        self.events.append({"kind": "reorder", "source_episode": snapshot["episode"],
                            "target_assignment": dict(target), "n_restored": len(displaced), "source_snapshot": snapshot_index,
                            "source_label": snapshot.get("label"),
                            "episodes_back": len(self.events) - snapshot["episode"]})
        return dict(target)

    def run(self):
        """Bounded first chain: two work moves, change both, revisit, restore."""
        self.explore()
        for obj in self.objects:
            current = self.executor.assignment()[obj]
            self.work(obj, next(r for r in self.receptacles if r != current))
        target = self.explore()
        current = self.executor.assignment()
        self.intervene({o: next(r for r in self.receptacles if r != current[o]) for o in self.objects})
        self.explore()
        return self.restore(target)

    def history_work_pair(self, cycle):
        """Introduce unused objects while retaining ordinary work in both directions."""
        current = self.validate_assignment(self.executor.assignment())
        counts = {obj: sum(e['kind'] == 'work' and e['object_name'] == obj
                           for e in self.events) for obj in self.objects}
        unused = [obj for obj in self.objects if obj not in self.manipulated]
        if len(unused) >= 2:
            first, second = unused[:2]
        else:
            stored = [obj for obj in self.objects if current[obj] == self.receptacles[1]]
            first = min(stored if cycle and stored else self.objects, key=counts.get)
            second = min((obj for obj in self.objects if obj != first), key=counts.get)
        return tuple((obj, next(r for r in self.receptacles if r != current[obj]))
                     for obj in (first, second))

    def run_history(self, change_cycles=2):
        """Two work transfers per change, using a larger pool as history grows."""
        if change_cycles < 2 or len(self.objects) < 2:
            raise ValueError('History test needs at least two objects and two changes')
        if len(self.objects) > 2 * change_cycles:
            raise ValueError('Two transfers per cycle cannot introduce every tracked object')
        initial = self.validate_assignment(self.executor.assignment())
        target_index = None
        for cycle in range(change_cycles):
            self.executor.history_cycle = cycle + 1
            pair = self.history_work_pair(cycle)
            previously_manipulated = set(self.manipulated)
            self.work_batch(pair)
            current = self.executor.assignment()
            changed = {obj: initial[obj] for obj, _ in pair if current[obj] != initial[obj]}
            if cycle == 0:
                # Leave the first object stored for an ordinary retrieval next cycle.
                changed = {pair[1][0]: initial[pair[1][0]]}
            elif any(obj not in previously_manipulated for obj, _ in pair):
                # Exchange the newcomer with a previously handled object, using
                # its own demonstrated destination rather than an arbitrary slot.
                spare = next((obj for obj in self.objects
                              if obj in previously_manipulated
                              and obj not in dict(pair) and current[obj] == initial[obj]), None)
                if spare is not None:
                    exchange = changed | {spare: next(r for r in self.receptacles if r != current[spare])}
                    if current | exchange != self.snapshots[target_index]['assignment']:
                        changed = exchange
            if not changed:
                raise RuntimeError('No nontrivial demonstrated change is available')
            self.intervene(changed)
            index = self.explore(label=f'after change {cycle + 1}')
            if target_index is None:
                target_index = index
        if self.manipulated != set(self.objects):
            raise RuntimeError('The history did not physically manipulate every tracked object')
        self.executor.restoring_history = True
        self.events.append({'kind': 'restore_request', 'source_snapshot': target_index,
                            'source_episode': self.snapshots[target_index]['episode'],
                            'source_label': self.snapshots[target_index]['label'],
                            'target_assignment': dict(self.snapshots[target_index]['assignment'])})
        return self.restore(target_index)
