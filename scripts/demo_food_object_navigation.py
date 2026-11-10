"""Food-search extension; restore the baseline navigator after this stage."""
from unittest.mock import patch
from cross_episode_sim.navigation.food_search_view import CounterFoodNavigator
from scripts.demo_semantic_navigation import main
from cross_episode_sim.navigation.household_object_docking import HouseholdObjectDocking
from cross_episode_sim.perception.portable_object_pointing import household_pointing_supervisor
if __name__=='__main__':
    with patch('cross_episode_sim.navigation.hybrid_navigation.HybridNavigator',CounterFoodNavigator):
        main(docking_factory=HouseholdObjectDocking,pointing_factory=household_pointing_supervisor)
