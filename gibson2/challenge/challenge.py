from gibson2.utils.utils import parse_config
import numpy as np
import json
import os
from gibson2.envs.igibson_env import iGibsonEnv
import cv2
import glob

import logging
logging.getLogger().setLevel(logging.WARNING)

import csv
import pybullet as p

class Challenge:
    def __init__(self):
        self.config_file = os.environ['CONFIG_FILE']
        self.split = os.environ['SPLIT']
        self.episode_dir = os.environ['EPISODE_DIR']
        self.eval_episodes_per_scene = os.environ.get(
            'EVAL_EPISODES_PER_SCENE', 50)

    def submit(self, agent):
        env_config = parse_config(self.config_file)

        task = env_config['task']
        if task == 'interactive_nav_random':
            metrics = {key: 0.0 for key in [
                'success', 'spl', 'effort_efficiency', 'ins', 'episode_return']}
        elif task == 'social_nav_random':
            metrics = {key: 0.0 for key in [
                'success', 'stl', 'psc', 'episode_return']}
        elif task == 'danger_interactive_nav_random':
            metrics = {key: 0.0 for key in [
                'success', 'spl', 'effort_efficiency', 'ins', 'episode_return', 'danger_metric']} # TO DO: change this to incorporate danger metric!
        else:
            assert False, 'unknown task: {}'.format(task)

        num_episodes_per_scene = self.eval_episodes_per_scene
        split_dir = os.path.join(self.episode_dir, self.split)
        assert os.path.isdir(split_dir)
        num_scenes = len(os.listdir(split_dir))
        assert num_scenes > 0
        total_num_episodes = num_scenes * num_episodes_per_scene

        idx = 0
        for json_file in os.listdir(split_dir):
            scene_id = json_file.split('.')[0]
            print("new scene: ", scene_id)
            json_file = os.path.join(split_dir, json_file)

            env_config['scene_id'] = scene_id
            env_config['load_scene_episode_config'] = True
            env_config['scene_episode_config_name'] = json_file
            env = iGibsonEnv(config_file=env_config,
                             mode='headless',
                             action_timestep=1.0 / 10.0,
                             physics_timestep=1.0 / 40.0)

            for _ in range(num_episodes_per_scene):
                idx += 1
                print('Episode: {}/{}'.format(idx, total_num_episodes))
                try:
                    agent.reset()
                except:
                    pass
                state = env.reset()
                episode_return = 0.0

                # For each episode, log the start position, goal position, and positions of all objects
                with open('episode_' + str(idx) + '_statics.csv', 'w', newline='') as csvfile:
                    spamwriter = csv.writer(csvfile, delimiter=',')
                    robot_id = env.robots[0].robot_ids[0]
                    spamwriter.writerow(['Initial position', robot_id, env.task.initial_pos[0], env.task.initial_pos[1]])
                    spamwriter.writerow(['Target position', robot_id, env.task.target_pos[0], env.task.target_pos[1]])
                    print("Initial position: " + str(env.task.initial_pos))
                    print("Target position: " + str(env.task.target_pos))

                    for int_obj in env.task.interactive_objects:
                        pos, _ = p.getBasePositionAndOrientation(int_obj.body_id)
                        spamwriter.writerow(['Interactive Obj Pos', str(int_obj.body_id), pos[0], pos[1], pos[2]])
                        print("Interactive object " + str(int_obj.body_id) + " position = " + str(pos))
                    for dan_obj in env.task.dangerous_objects:
                        pos, _ = p.getBasePositionAndOrientation(dan_obj.body_id)
                        spamwriter.writerow(['Danger Obj Pos', str(dan_obj.body_id), pos[0], pos[1], pos[2]])
                        print("Dangerous object " + str(dan_obj.body_id) + " position = " + str(pos) + " and danger = " + str(dan_obj.collision_danger))

                 
                # frameSize = (320, 180)
                # out = cv2.VideoWriter('output_video_epoch_'+str(idx)+'.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, frameSize)

                with open('episode_' + str(idx) + '_trajectory.csv', 'w', newline='') as csvfile2:
                    spamwriter2 = csv.writer(csvfile2, delimiter=',')


                    count=0
                    while True:
                        action = env.action_space.sample()
                        action = agent.act(state)

                        #out.write((state['rgb']*255).astype(np.uint8))
                        cv2.imwrite("epoch_"+str(idx)+"_"+str(count)+".jpg", (state['rgb']*255).astype(np.uint8))
                        # print((state['rgb']*255).astype(np.uint8)) # shape (180, 320, 3)

                        # For each timestep, log the robot position so we can make a trajectory later
                        robot_id = env.robots[0].robot_ids[0]
                        pos, _ = p.getBasePositionAndOrientation(robot_id)
                        spamwriter2.writerow(['Robot Current Pos', robot_id, pos[0], pos[1], pos[2]])

                        state, reward, done, info = env.step(action)
                        episode_return += reward
                        count += 1
                        if done:
                            break
                    # out.release()

                metrics['episode_return'] += episode_return
                for key in metrics:
                    if key in info:
                        metrics[key] += info[key]

        for key in metrics:
            metrics[key] /= total_num_episodes
            print('Avg {}: {}'.format(key, metrics[key]))


if __name__ == '__main__':
    challenge = Challenge()
    challenge.submit(None)
