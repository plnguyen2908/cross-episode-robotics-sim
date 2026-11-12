"""Main refrigerator door contact for the visible vertical-pull fixture family.

The lower horizontal freezer pull belongs to a drawer, not the hinged door.
No fixture IDs or coordinates are supplied to the visual pointer. This profile
does not claim support for refrigerators with different handle arrangements.
"""


class FridgeDoorHandlePointer:
    def __init__(self,pointer):self.pointer=pointer

    def point(self,image,description):
        return self.pointer.point(image,'vertical refrigerator door handle')
