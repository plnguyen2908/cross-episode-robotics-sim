"""Target-focused bowl proposals; preserve the original cup proposal path."""
import sys,json
from cross_episode_sim.runtime import artifacts
from .sensor_household import SensorHouseholdPipeline


class TargetConditionedHouseholdPipeline(SensorHouseholdPipeline):
    def _pick_object(self,number,object_dock,pick):
        observed=json.loads(artifacts.read_text(object_dock/'report.json')).get('selected_object',{})
        if self.config.get('task')!='set_table' or observed.get('label')!='bowl':
            return super()._pick_object(number,object_dock,pick)
        return self.stage(f'{number}_pick',
            [sys.executable,'scripts/execute_target_pick.py','--source',str(object_dock),'--output',str(pick)],
            pick/'physics/report.json','execution_success')
