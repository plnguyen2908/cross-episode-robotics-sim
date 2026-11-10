"""Attempt the semantic recipe; only the private environment scores the task."""


def execute_recipe(interface, plan, sim_time, *, continue_on_failure=True):
    for index, call in enumerate(plan):
        print(f'PUBLIC CALL {index+1}/{len(plan)}: {call.skill} {call.arguments}', flush=True)
        start = sim_time()
        result = interface.execute_skill(call.skill, call.arguments)
        yield index, call, result, sim_time() - start
        if not result['success']:
            if not continue_on_failure:
                raise RuntimeError(f'{call.skill}: {result["reason"]}')
            print(f'CALL FAILED; CONTINUING RECIPE: {call.skill}: {result["reason"]}', flush=True)
