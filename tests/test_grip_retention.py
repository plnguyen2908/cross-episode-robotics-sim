from cross_episode_sim.control.grip_retention import GripRetention,GRIP_LOSS_TICKS

HELD={'left_driver_joint':.738,'right_driver_joint':.738}          # food container, fix144 call 19


def test_held_object_with_small_finger_motion_is_kept():
    grip=GripRetention(HELD)
    assert not any(grip.lost({'left_driver_joint':.730,'right_driver_joint':.722}) for _ in range(500))


def test_fingers_running_to_their_stop_is_a_loss_after_a_short_time():
    grip=GripRetention(HELD);closed={'left_driver_joint':.824,'right_driver_joint':.824}
    seen=[grip.lost(closed) for _ in range(GRIP_LOSS_TICKS)]
    assert seen[:-1]==[False]*(GRIP_LOSS_TICKS-1) and seen[-1]


def test_a_brief_knock_is_not_a_loss():
    grip=GripRetention(HELD);knock={'left_driver_joint':.638,'right_driver_joint':.637}
    for _ in range(GRIP_LOSS_TICKS-1):assert not grip.lost(knock)
    assert not grip.lost(HELD) and grip.count==0


def test_wide_object_lost_is_seen():
    grip=GripRetention({'left_driver_joint':.247});assert all(not grip.lost({'left_driver_joint':.47}) for _ in range(GRIP_LOSS_TICKS-1))
    assert grip.lost({'left_driver_joint':.824})
