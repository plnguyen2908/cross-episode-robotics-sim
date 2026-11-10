"""Use the common navigation pipeline with vertical-fixture docking only."""
from scripts.demo_semantic_navigation import main
from cross_episode_sim.navigation.fixture_docking import FixtureDocking

from cross_episode_sim.perception.fixture_pointing import fixture_pointing_supervisor

if __name__=='__main__':main(docking_factory=FixtureDocking,pointing_factory=fixture_pointing_supervisor)
