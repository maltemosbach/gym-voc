import gym
from gym import spaces
import numpy as np


class Dummy(gym.GoalEnv):
    def __init__(self, num_actions: int = 2):
        self.current_episode = 0
        self.in_episode_step = -1
        self.time_limit = 50

        self.action_space = spaces.Box(-1.0, 1.0, shape=(num_actions,),
                                       dtype="float32")
        obs = self._get_obs()
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape,
                    dtype="float32"
                ),
                achieved_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape,
                    dtype="float32"
                ),
                observation=spaces.Box(
                    -np.inf, np.inf, shape=obs["observation"].shape,
                    dtype="float32"
                ),
            )
        )

    def reset(self):
        self.in_episode_step = 0
        self.current_episode += 1
        obs = self._get_obs()
        return obs

    def step(self, action):
        self.in_episode_step += 1
        obs = self._get_obs()
        done = False
        reward = np.array([0.]).astype(np.float32)[0]
        info = {}
        return obs, reward, done, info

    def render(self, mode="human", width=500, height=500):
        if mode == "rgb_array":
            return np.zeros((width, height), dtype=np.uint8)
        elif mode == "human":
            pass

    def compute_reward(self, achieved_goal, desired_goal, info):
        return np.array([0.]).astype(np.float32)[0]

    def compute_done(self, achieved_goal, goal, info):
        return False

    def _get_obs(self):
        obs = np.random.rand(4)
        obs[2] = self.in_episode_step
        obs[3] = self.current_episode
        desired_goal = np.random.rand(2)
        achieved_goal = obs[0:2]
        return {"observation": obs, "desired_goal": desired_goal,
                "achieved_goal": achieved_goal}

    # Methods for hierarchical agents
    def set_subgoal_pos(self, goals) -> None:
        pass