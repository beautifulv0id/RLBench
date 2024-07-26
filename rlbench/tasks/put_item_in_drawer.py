from typing import List, Tuple

import numpy as np
from pyrep.objects.dummy import Dummy
from pyrep.objects.joint import Joint
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import DetectedCondition
from rlbench.backend.task import Task
from rlbench.backend.observation import Observation

KEYPOINT_NUM = 4

class PutItemInDrawer(Task):

    def init_task(self) -> None:
        self._options = ['bottom', 'middle', 'top']
        self._anchors = [Dummy('waypoint_anchor_%s' % opt)
                         for opt in self._options]
        self._joints = [Joint('drawer_joint_%s' % opt)
                        for opt in self._options]
        self._waypoint1 = Dummy('waypoint2')
        self._item = Shape('item')
        self.register_graspable_objects([self._item])
        self._keypoints = [Dummy('keypoint%s' % i) 
                           for i in range(KEYPOINT_NUM)]

    def init_episode(self, index) -> List[str]:
        option = self._options[index]
        anchor = self._anchors[index]
        self._waypoint1.set_position(anchor.get_position())
        success_sensor = ProximitySensor('success_' + option)
        self.register_success_conditions(
            [DetectedCondition(self._item, success_sensor)])
        return ['put the item in the %s drawer' % option,
                'put the block away in the %s drawer' % option,
                'open the %s drawer and place the block inside of it' % option,
                'leave the block in the %s drawer' % option]

    def variation_count(self) -> int:
        return 3

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, - np.pi / 8], [0, 0, np.pi / 8]

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
