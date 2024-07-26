from typing import List
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from pyrep.objects.object import Object
from pyrep.objects.dummy import Dummy
from rlbench.backend.conditions import DetectedCondition
from rlbench.backend.observation import Observation
import numpy as np

DIRT_NUM = 5
KEYPOINT_NUM = 9

class SweepToDustpanOfSize(Task):

    def init_task(self) -> None:
        self._dustpan_sizes = ['tall', 'short']

        broom = Shape('broom')
        self.register_graspable_objects([broom])

        self._waypoint_paths = {
            0: [Dummy('point1a'),
                Dummy('point1b'),
                Dummy('point1c')],

            1: [Dummy('point2a'),
                Dummy('point2b'),
                Dummy('point2c')]
        }

        self._keypoints = [Dummy('keypoint%s' % i) 
                           for i in range(KEYPOINT_NUM)]


    def init_episode(self, index: int) -> List[str]:
        self._variation_index = index
        dustpan_size = self._dustpan_sizes[self._variation_index]

        success_sensor = ProximitySensor(f'success_{dustpan_size}')
        dirts = [Shape('dirt' + str(i)) for i in range(DIRT_NUM)]
        conditions = [DetectedCondition(dirt, success_sensor) for dirt in dirts]
        self.register_success_conditions(conditions)

        target_waypoints = self._waypoint_paths[self._variation_index]
        self._waypoints = [Dummy('waypoint%d'%(i))
                           for i in range(2, 5)]

        for i in range(len(target_waypoints)):
            self._waypoints[i].set_pose(target_waypoints[i].get_pose())
        self.register_stop_at_waypoint(2+i+1)

        return ['sweep dirt to the %s dustpan' % (dustpan_size),
                'sweep the dirt up into the %s dustpan' % (dustpan_size),
                'use the broom to brush the dirt into the %s dustpan' % (dustpan_size),
                'clean up the dirt with the %s pan' % (dustpan_size)]

    def variation_count(self) -> int:
        return 2

    # def boundary_root(self) -> Object:
    #     return Shape('boundary_root')

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
