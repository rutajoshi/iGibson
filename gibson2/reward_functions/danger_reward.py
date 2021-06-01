from gibson2.reward_functions.reward_function_base import BaseRewardFunction


class DangerReward(BaseRewardFunction):
    """
    Danger reward
    Penalize robot collision. Typically collision_reward_weight is negative.
    """

    def __init__(self, config):
        super(DangerReward, self).__init__(config)
        # TODO: Change config to include danger_reward_weight
        # This is the config from examples/configs
        # TODO: make a config file
        self.danger_reward_weight = self.config.get(
            'danger_reward_weight', -0.1
        )

    def get_reward(self, task, env):
        """
        Reward is danger_reward_weight * object_collision_danger
        danger_reward_weight is normally negative
            - (collision with dangerous objects is penalized)
        object_collision_danger is normally positive
            - (more positive = more dangerous)

        :param task: task instance
        :param env: environment instance
        :return: reward
        """
        # TODO: keep dict from body_id to danger score in the task
        # Go through collision links to see if any objects you collide with are in danger_dict

        # 1) find out if there was a collision
        has_collision = float(len(env.collision_links) > 0)

        # 2) If collision, for each collision link, find out collision danger of each object
        object_collision_danger = []
        collided_objs_danger = task.get_obj_collision_danger(env)
        # print(env.collision_links)
        collision_objects = set([col[2] for col in env.collision_links])
        # print(collision_objects)
        # print(collided_objs_danger)

        for obj_id in collision_objects:
            # TODO: find out how to get object collision danger from collision_links
            # Object id can be used to find object in the environment
            if obj_id == 1 or obj_id == 0:
                continue
            object_collision_danger.append(collided_objs_danger[obj_id])
            
        # 3) Compute danger reward
        reward = 0
        for obj_danger in object_collision_danger:
            reward += (self.danger_reward_weight * obj_danger)

        return reward
