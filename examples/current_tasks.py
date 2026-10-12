"""Example current task plugin: practice coffee preparation from one history.

Load with --task-module examples.current_tasks. Uses the existing coffee worker;
this is a different goal from serving paired breakfast settings.
"""
from cross_episode_sim.generation import CurrentPlan, CurrentTaskDefinition, register_current_task
from cross_episode_sim.generation.library import _task_options, _validate, _validate_count, run_episode
from cross_episode_sim.tasks.full_breakfast.__main__ import VIDEO, sample


def validate(config):
    if set(config.parameters) - {'cups'}:
        raise ValueError('coffee_practice parameters: cups (1–4)')
    _validate(config.seed, config.parameters.get('cups', 1), config.layout_profile, config.scene)
    _task_options('coffee', config.scene)
    if len(config.histories) != 1:
        raise ValueError('coffee_practice requires one coffee history')


def compose(config, histories):
    if len(histories) != 1 or histories[0]['task'] != 'coffee':
        raise ValueError('coffee_practice requires a coffee history')
    cups = config.parameters.get('cups', 1)
    settings = sample(config.seed, config.layout_profile, cups)[1]['coffee']
    settings.update(config.scene)
    prompt = (f'Prepare {cups} cup(s) of coffee using the procedure demonstrated in the history. '
              'Return each brewed cup to its kitchen home and finish with your gripper empty.\n')
    return CurrentPlan('coffee', settings, prompt, VIDEO['coffee'])


def execute(plan, output, gpu):
    run_episode(plan.worker_task, plan.settings, output, 'current', gpu)


def verify(plan, run):
    _validate_count(run, plan.settings['people'])


register_current_task('coffee_practice', CurrentTaskDefinition(validate, compose, execute, verify))
