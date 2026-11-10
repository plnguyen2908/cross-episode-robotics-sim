"""Additional generic-object navigation; baseline cup/bowl dispatch is unchanged."""
from scripts.demo_semantic_navigation import main
from cross_episode_sim.navigation.household_object_docking import HouseholdObjectDocking

from cross_episode_sim.perception.portable_object_pointing import household_pointing_supervisor

if __name__=='__main__':main(docking_factory=HouseholdObjectDocking,pointing_factory=household_pointing_supervisor)
