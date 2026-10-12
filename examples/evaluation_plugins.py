"""Minimal adapter examples; these are interface smoke tests, not trained models."""


class HoldPlanner:
    def __init__(self, decisions=2):
        self.remaining = decisions

    def reset(self, context):
        self.context = context

    def plan(self, observation, feedback):
        if self.remaining <= 0:
            return {'finish': True}
        self.remaining -= 1
        return {'actions': [observation['state'].tolist()]}


class InspectAgent:
    def run(self, context, tools):
        observation = tools.observe()
        # A coding-agent adapter can write programs that call these same tools.
        # Its model may choose oracle/VLA skill calls from context['skill_catalog'].
        assert 'images' in observation
        tools.finish()


class HoldVLA:
    def __init__(self, steps=5):
        self.steps = steps

    def start_skill(self, name, arguments, context):
        self.remaining = self.steps

    def act(self, observation):
        if self.remaining <= 0:
            return {'done': True}
        self.remaining -= 1
        return {'actions': [observation['state'].tolist()]}


class OneSkillPlanner:
    """Request one named skill, then finish; useful for testing adapter wiring."""

    def __init__(self, skill, arguments):
        self.skill, self.arguments = skill, arguments

    def reset(self, context):
        self.context = context

    def plan(self, observation, feedback):
        return ({'finish': True} if feedback is not None else
                {'skill': self.skill, 'arguments': self.arguments})


class RobotToolsAgent:
    """Exercise AGP motion tools at the current pose; does not solve the task."""

    def run(self, context, tools):
        import numpy as np
        from scipy.spatial.transform import Rotation
        observation = tools.observe()
        kinematics = tools.kinematics()
        pose = np.asarray(kinematics['tcp_pose_world'])
        quat = Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]].tolist()
        assert tools.move_ee(pose[:3, 3].tolist(), quat, duration=.08)['status'] == 'executed'
        assert tools.move_joints(observation['state'][3:10].tolist(), duration=.08)['status'] == 'executed'
        assert tools.gripper(float(observation['state'][10]), duration=.08)['status'] == 'executed'
        tools.finish()


def agent_written_position_ik(tools, target_world, iterations=20):
    """Example agent-side numerical IK using only public robot queries.

    This solves position only, leaving orientation unconstrained. It neither
    selects a task target nor moves the robot; the caller chooses whether to
    execute the returned joints with move_joints.
    """
    import numpy as np
    state = tools.kinematics()
    joints = np.asarray(state['joints_rad'])
    limits = np.asarray(state['joint_limits_rad'])
    for _ in range(iterations):
        fk = tools.kinematics(joints=joints.tolist())
        position = np.asarray(fk['tcp_pose_world'])[:3, 3]
        error = np.asarray(target_world)-position
        if np.linalg.norm(error) < .001:
            return joints.tolist()
        jacobian = np.asarray(fk['jacobian_world'])[:3]
        delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + .001*np.eye(3), error)
        joints = np.clip(joints+np.clip(delta, -.1, .1), limits[:, 0], limits[:, 1])
    raise RuntimeError('Agent-side position IK did not converge')


class ProgrammedIKAgent:
    """Exercise agent-written IK for a 5 mm TCP motion, without move_ee."""

    def run(self, context, tools):
        import numpy as np
        tools.observe()
        start = np.asarray(tools.kinematics()['tcp_pose_world'])[:3, 3]
        target = start + [0., 0., .005]
        joints = agent_written_position_ik(tools, target)
        response = tools.move_joints(joints, duration=1.)
        assert response['status'] == 'executed'
        measured = np.asarray(tools.kinematics()['tcp_pose_world'])[:3, 3]
        assert np.linalg.norm(measured-target) < .002, (measured, target)
        tools.observe()
        tools.finish()
