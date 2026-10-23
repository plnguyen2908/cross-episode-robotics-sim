"""Strict qualification wrapper for the shared direct panel-push implementation."""

def install():
    from cross_episode_sim.fixtures import door_push
    from cross_episode_sim.fixtures.cabinet_door import CabinetDoorTest
    original = CabinetDoorTest.push_open_door
    def direct(c, target):
        try:
            return original(c, target)
        except door_push.PushApproachError as exc:
            if abs(target) > abs(c.angle()):
                raise RuntimeError('Direct push without re-navigation rejected: '+str(exc)) from exc
            raise
    CabinetDoorTest.push_open_door = direct
