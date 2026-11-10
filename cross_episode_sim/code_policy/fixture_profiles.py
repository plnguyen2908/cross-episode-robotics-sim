"""Qwen component profiles using already-qualified sensor contact implementations."""
from ..runtime.fixture_controls import LiveFixtureControls


class HouseholdFixtureControls(LiveFixtureControls):
    def execute(self,skill,*,target,progress):
        self.current_target=target
        return super().execute(skill,target=target,progress=progress)

    def run_script(self,command,root):
        command=list(command)
        if (self.current_target=='fridge' and len(command)>1
                and command[1]=='scripts/test_sensor_hinged_cycle.py'):
            # The qualified fridge profile uses finite-difference robot Jacobian
            # IK. This is the existing sensor backend, not oracle contact poses.
            index=command.index('--contact-ik')
            command[index+1]='finite'
        return super().run_script(command,root)
