import math

from gibson2.objects.object_base import Object
import pybullet as p


class Cube(Object):
    """
    Cube shape primitive
    """

    def __init__(self, pos=[1, 2, 3], dim=[1, 2, 3], visual_only=False, mass=1000, color=[1, 1, 1, 1]):
        """
        CS331B: We have modified cube.py to be the dangerous object for DangerInteractiveNavRandomTask.
        The agent learning that task should learn to avoid cubes.
        """
        super(Cube, self).__init__()
        self.basePos = pos
        self.dimension = dim
        self.visual_only = visual_only
        self.mass = mass

        self.collision_danger = np.random.rand() #0.75
        self.color = [0, 0, self.collision_danger, 1]
        # TODO: consider changing the mass so that the cube can suffer consequence
        # you should likely be able to move it
        # 1000 kg is definitely too high

    def _load(self):
        """
        Load the object into pybullet
        """
        baseOrientation = [0, 0, 0, 1]
        colBoxId = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=self.dimension)
        visualShapeId = p.createVisualShape(
            p.GEOM_BOX, halfExtents=self.dimension, rgbaColor=self.color)
        if self.visual_only:
            body_id = p.createMultiBody(baseCollisionShapeIndex=-1,
                                        baseVisualShapeIndex=visualShapeId)
        else:
            body_id = p.createMultiBody(baseMass=self.mass,
                                        baseCollisionShapeIndex=colBoxId,
                                        baseVisualShapeIndex=visualShapeId)

        p.resetBasePositionAndOrientation(
            body_id, self.basePos, baseOrientation)

        return body_id
