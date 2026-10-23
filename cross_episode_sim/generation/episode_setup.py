"""Datagen scene lifecycle shared by atomic training and composite episodes."""
from copy import deepcopy


def initialize_environment(config, specification, output, factory=None):
    """Configure exactly once, before giving an interface to any policy."""
    if factory is None:
        from cross_episode_sim.generation.environment_factory import create_environment
        factory = create_environment
    env = factory(config, output)
    try:
        env.configure(deepcopy(config['scene']), deepcopy(specification))
        evidence = env.verify_setup(config)
        if not evidence or not all(v is True for v in evidence.values()):
            raise RuntimeError('Scene setup verification failed: ' + str(evidence))
        env.save_initial_state(output / 'initial_state.npz')
        if not (output / 'initial_state.npz').is_file():
            raise RuntimeError('Initial simulator state missing')
    except BaseException:
        env.close()
        raise
    return env, evidence


def prepare_composite_episode(config, output, factory=None):
    from cross_episode_sim.evaluation.composite_goals import CompositeEvaluator
    from cross_episode_sim.generation.composite_scenes import scene_specification
    from cross_episode_sim.tasks.composites import create_task
    env, evidence = initialize_environment(config, scene_specification(config), output, factory)
    try:
        evaluator = CompositeEvaluator(config, env.snapshot())
        resolved_config=deepcopy(config)
        route=getattr(env,'manifest',{}).get('resolved_route')
        if route is not None:resolved_config['resolved_route']=route
        policy = create_task(resolved_config, output, interface=env.interface)
        return env, policy, evaluator, evidence
    except BaseException:
        env.close()
        raise


def execute_atomic_example(config, specification, output, action, factory=None):
    """One datagen-authored training example; the atomic never builds the scene.

    The caller varies the scene spec/seed/objects using the same factory as a
    composite. Success is the atomic's measured postcondition, not a composite
    goal or a heating deadline. The caller owns rendering/closing the returned env.
    """
    from cross_episode_sim.tasks.composites.execution import execute_plan
    env, setup = initialize_environment(config, specification, output, factory)
    try:
        result = execute_plan(env.interface, [action], label='atomic training example')
        return env, dict(result, setup_evidence=setup)
    except BaseException:
        env.close()
        raise
