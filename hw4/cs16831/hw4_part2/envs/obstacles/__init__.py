from gym.envs.registration import register

register(
    id='obstacles-hw4_part2-v0',
    entry_point='cs16831.hw4_part2.envs.obstacles:Obstacles',
    max_episode_steps=500,
)
from cs16831.hw4_part2.envs.obstacles.obstacles_env import Obstacles
