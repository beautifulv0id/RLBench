from typing import List, Tuple
import numpy as np
from pyrep.objects.dummy import Dummy
from pyrep.objects.joint import Joint
from rlbench.backend.conditions import JointCondition
from rlbench.backend.task import Task
from rlbench.backend.observation import Observation
from scipy.spatial.transform import Rotation as R
import numpy as np

class OpenDrawerKeypoint(Task):

    def init_task(self) -> None:
        self._options = ['bottom', 'middle', 'top']
        self._anchors = [Dummy('waypoint_anchor_%s' % opt)
                         for opt in self._options]
        self._joints = [Joint('drawer_joint_%s' % opt)
                        for opt in self._options]
        self._waypoint1 = Dummy('waypoint1')
        self._keypoints = [Dummy('keypoint%s' % i) 
                           for i in range(5)]

    def init_episode(self, index: int) -> List[str]:
        option = self._options[index]
        self._waypoint1.set_position(self._anchors[index].get_position())
        self.register_success_conditions(
            [JointCondition(self._joints[index], 0.15)])
        return ['open the %s drawer' % option,
                'grip the %s handle and pull the %s drawer open' % (
                    option, option),
                'slide the %s drawer open' % option]
    
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
    
    def variation_count(self) -> int:
        return 3

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, - np.pi / 8], [0, 0, np.pi / 8]
