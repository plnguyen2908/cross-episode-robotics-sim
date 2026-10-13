"""History-grounded multiple-choice QA paired with breakfast action examples."""
import hashlib
import json
import random
from pathlib import Path

from .dataset import write_json

ROOMS = {'kitchen': 'Kitchen', 'living': 'Living room', 'dining': 'Dining room'}


def build_eqa(record, manifest, coffee_scene):
    """Return public questions and a separate private answer key."""
    histories = [e for e in record['episodes'] if e['role'] == 'history']
    by_task = {e['task']: e for e in histories}
    if len(histories) != 2 or set(by_task) != {'coffee', 'breakfast'}:
        raise ValueError('EQA v1 requires one coffee and one breakfast history')
    if record['current']['task'] not in ('full_breakfast', 'breakfast_with_coffee'):
        raise ValueError('EQA v1 supports breakfast with coffee')
    breakfast, coffee = by_task['breakfast'], by_task['coffee']
    settings = manifest['episodes']
    people = settings[breakfast['name']]['settings']['people']
    if type(people) is not int or people not in (1, 2, 3):
        raise ValueError('Expected 1–3 paired place settings')
    if settings['current']['settings']['people'] != people:
        raise ValueError('Breakfast history count does not match current')
    room = coffee_scene['coffee']['machine_room']
    if room not in ROOMS or room != settings[coffee['name']]['settings']['machine_room']:
        raise ValueError('Coffee history room metadata is missing or inconsistent')
    identity = json.dumps([record['seed'], [(e['name'], e.get('history_id')) for e in histories]])
    rng = random.Random(int(hashlib.sha256(identity.encode()).hexdigest(), 16))
    public, answers = [], []
    definitions = [
        ('bowls_needed', 'counting',
         f"Based on the breakfast demonstration ({breakfast['name']}), how many breakfast bowls should be served in the current task?",
         [1, 2, 3, 4], people, breakfast, 'settings.people'),
        ('cups_needed', 'counting',
         f"Based on the breakfast demonstration ({breakfast['name']}), how many cups of coffee should be served in the current task, with one cup per person?",
         [1, 2, 3, 4], people, breakfast, 'settings.people'),
        ('past_coffee_machine_room', 'location',
         f"In the past coffee-making demonstration ({coffee['name']}), which room contained the coffee machine?",
         list(ROOMS), room, coffee, 'task_manifest.coffee.machine_room'),
    ]
    for identifier, category, question, values, correct, history, field in definitions:
        rng.shuffle(values)
        choices = [dict(id=chr(65+i), text=ROOMS[v] if isinstance(v, str) else str(v))
                   for i, v in enumerate(values)]
        answer = choices[values.index(correct)]
        public.append(dict(id=identifier, category=category, question=question,
                           choices=choices, history_videos=[history['video']]))
        answers.append(dict(question_id=identifier, correct_choice=answer['id'],
                            answer=answer['text'], evidence=dict(episode=history['name'],
                            history_id=history.get('history_id'), field=field, value=correct)))
    return (dict(version=1, task='history_eqa', instruction='Watch the referenced history videos and select one choice per question.',
                 questions=public),
            dict(version=1, private=True, answers=answers))


def write_eqa(example):
    root = Path(example)
    record = json.loads((root / 'episode.json').read_text())
    manifest = json.loads((root / 'run_manifest.json').read_text())
    coffee = next(e for e in record['episodes'] if e['role'] == 'history' and e['task'] == 'coffee')
    scene_path = root / 'initial_scenes' / coffee['name'] / 'task_manifest.json'
    if not scene_path.exists():
        scene_path = root / coffee['run'] / 'task_manifest.json'
    questions, answers = build_eqa(record, manifest, json.loads(scene_path.read_text()))
    write_json(root / 'eqa_questions.json', questions)
    write_json(root / 'eqa_answers.json', answers)
    return len(questions['questions'])
