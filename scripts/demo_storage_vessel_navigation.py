"""Clear/store's vessel descriptions with the existing vessel docking policy."""
from scripts.demo_semantic_navigation import main
from cross_episode_sim.perception.storage_vessel_pointing import storage_vessel_supervisor


if __name__ == '__main__':
    main(pointing_factory=storage_vessel_supervisor)
