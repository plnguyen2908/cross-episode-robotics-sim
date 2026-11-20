"""Notice, from the finger joint encoders alone, that a held object is no longer in the hand."""

# A finger joint this far (rad) from where it stood when the carry began ...
GRIP_LOSS_JOINT_CHANGE=.05
# ... for this many consecutive 50 Hz ticks means the object has left the hand.
GRIP_LOSS_TICKS=10


class GripRetention:
    """Compare the finger joints with their values at the start of a carry.

    The gripper keeps squeezing while it carries. When the object is knocked
    out the fingers run on to their closed stop: +0.09 rad on every finger
    joint for the food container and +0.58 rad for the drink bottle, within
    0.2 s (fix144 calls 19 and 27; both carries then drove on to the dining
    table and the next 'place' reported success with an empty hand). Before a
    loss the joints moved at most 0.016 rad.

    Only closing counts. A held pot that slides in the grip pushes the fingers
    apart instead: fix167 call 25, -0.05 rad over 12 s with the pot still
    between both pads (private), and the carry was stopped as 'lost'.
    """
    def __init__(self,reference,change=GRIP_LOSS_JOINT_CHANGE,ticks=GRIP_LOSS_TICKS):
        self.reference=dict(reference);self.change=float(change);self.ticks=int(ticks);self.count=0

    def lost(self,current):
        """Feed the current joint positions (name -> rad); True once the loss has lasted long enough."""
        moved=any(float(current[name])-value>self.change for name,value in self.reference.items())
        self.count=self.count+1 if moved else 0
        return self.count>=self.ticks
