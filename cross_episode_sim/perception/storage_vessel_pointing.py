"""Disambiguate bowls and mugs for the added clear-and-store recipe.

The qualified set-table supervisor is unchanged. These descriptions use visible
shape, not asset IDs, colors, counts or hidden category annotations.
"""
from .molmo_pointing import PointingSupervisor


class StorageVesselPointer:
    descriptions = {
        'bowl': 'wide shallow eating bowls without handles, excluding cups and mugs',
        'cup': 'cups or mugs with handles, excluding shallow eating bowls',
    }

    def __init__(self, pointer, label):
        self.pointer = pointer
        self.description = self.descriptions[label]

    def point(self, image, target):
        return self.pointer.point(image, self.description)


def storage_vessel_supervisor(pointer, segmenter, label):
    return PointingSupervisor(StorageVesselPointer(pointer, label), segmenter, label)
