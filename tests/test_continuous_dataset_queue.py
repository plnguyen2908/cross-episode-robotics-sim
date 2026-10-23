import threading
from cross_episode_sim.generation.continuous_queue import run_continuous_queue


def test_refills_while_other_episode_is_still_running():
    pending=iter(range(4));replacement_started=threading.Event();results=[]
    def run(job):
        if job==0:
            assert replacement_started.wait(3), 'Pool waited for slow episode before refilling'
        if job==2:replacement_started.set()
        return {'success':job!=1}
    run_continuous_queue(lambda:next(pending,None),run,lambda job,result:results.append((job,result)),2)
    assert len(results)==4 and replacement_started.is_set()


def test_same_task_can_fill_all_eight_slots():
    pending=iter([{'task':'set_table','index':i} for i in range(8)])
    barrier=threading.Barrier(8);results=[]
    def run(job):
        barrier.wait(timeout=3)
        return True
    run_continuous_queue(lambda:next(pending,None),run,lambda job,result:results.append(job),8)
    assert len(results)==8
