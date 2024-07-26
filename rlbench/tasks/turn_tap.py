from typing import List
from pyrep.objects.dummy import Dummy
from pyrep.objects.joint import Joint
from rlbench.backend.task import Task
from rlbench.backend.conditions import JointCondition
from rlbench.backend.observation import Observation
import numpy as np

OPTIONS = ['left', 'right']
KEYPOINT_NUM = 2

class TurnTap(Task):

    def init_task(self) -> None:
        self.left_start = Dummy('waypoint0')
        self.left_end = Dummy('waypoint1')
        self.right_start = Dummy('waypoint5')
        self.right_end = Dummy('waypoint6')
        self.left_joint = Joint('left_joint')
        self.right_joint = Joint('right_joint')
        self._keypoints = [Dummy('keypoint%s' % i) 
                           for i in range(KEYPOINT_NUM)]

    def init_episode(self, index: int) -> List[str]:
        option = OPTIONS[index]
        if option == 'right':
            self.left_start.set_position(self.right_start.get_position())
            self.left_start.set_orientation(self.right_start.get_orientation())
            self.left_end.set_position(self.right_end.get_position())
            self.left_end.set_orientation(self.right_end.get_orientation())
            self.register_success_conditions(
                [JointCondition(self.right_joint, 1.57)])
        else:
            self.register_success_conditions(
                [JointCondition(self.left_joint, 1.57)])

        return ['turn %s tap' % option,
                'rotate the %s tap' % option,
                'grasp the %s tap and turn it' % option]

    def variation_count(self) -> int:
        return 2
    
    def decorate_observation(self, observation: Observation) -> Observation:
        pcd = []
        poses = []
        for keypoint in self._keypoints:
            H = keypoint.get_matrix()
            rot, pos = H[:3, :3], H[:3, 3]
            pcd.append(pos)
            poses.append(H)
            for ax in rot:
                pcd.append(pos + 0.05 * ax)
                pcd.append(pos - 0.05 * ax)
        pcd = np.array(pcd)
        poses = np.array(poses)
        observation.misc['low_dim_pcd'] = pcd
        observation.misc['low_dim_poses'] = poses
        return observation
