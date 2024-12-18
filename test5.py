import os
import numpy as np
import random
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from rlgym import make
from rlgym.utils.obs_builders import DefaultObs
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym.utils.reward_functions import RewardFunction
from rlgym_tools.extra_action_parsers.kbm_act import KBMAction


class GoalReward(RewardFunction):
    def reset(self, initial_state):
        self.previous_ball_position = initial_state.ball.position
        self.previous_touch = False

    def get_reward(self, player, state, previous_action):
        goal_position = np.array([0, 5120, 0])
        ball_position = state.ball.position
        distance_to_goal = np.linalg.norm(goal_position - ball_position)

        reward = -distance_to_goal

        if np.linalg.norm(ball_position - self.previous_ball_position) > 10:
            reward += 10
            self.previous_touch = True
        else:
            reward -= 1
            self.previous_touch = False

        if ball_position[1] > 5100 and abs(ball_position[0]) < 1000: 
            reward += 500 

        if distance_to_goal < 1000:  
            reward += 50  

        self.previous_ball_position = ball_position
        return reward


class ActionRepeatParser(KBMAction):
    def __init__(self, repeat_duration=15):
        """
        Custom action parser to repeat actions for a specified number of ticks.
        :param repeat_duration: Number of ticks to repeat each action.
        """
        super().__init__()
        self.repeat_duration = repeat_duration
        self.action_buffer = None
        self.action_repeat_counter = 0

    def parse_actions(self, raw_actions, state):
        """
        Override the parse_actions method to handle action repetition.
        :param raw_actions: Raw actions to parse.
        :param state: Current game state.
        :return: Repeated or new action based on the repeat counter.
        """
        if self.action_repeat_counter == 0:
            self.action_buffer = super().parse_actions(raw_actions, state)
            self.action_repeat_counter = self.repeat_duration
        
        self.action_repeat_counter -= 1
        return self.action_buffer


def make_env():
    def _init():
        env = make(
            reward_fn=GoalReward(),
            obs_builder=DefaultObs(),
            action_parser=ActionRepeatParser(repeat_duration=15),  # Use your custom parser
            terminal_conditions=[TimeoutCondition(300), GoalScoredCondition()],
            game_speed=150,
        )
        return env
    return _init


class ProgressCallback:
    def __init__(self, print_interval=10000):
        self.print_interval = print_interval

    def __call__(self, locals_, globals_):
        if locals_['self'].num_timesteps % self.print_interval == 0:
            print(f"Progress: {locals_['self'].num_timesteps} timesteps")
            current_lr = locals_['self'].policy.optimizer.param_groups[0]['lr']
            print(f"Current Learning Rate: {current_lr}")
        return True


def lr_schedule_fn(current_timestep: int) -> float:
    initial_lr = 1e-3
    final_lr = 1e-5
    total_timesteps = 10_000_000

    # Linear schedule for learning rate
    lr = initial_lr - (initial_lr - final_lr) * (current_timestep / total_timesteps)
    
    # Ensure that the learning rate never goes below final_lr
    return max(lr, final_lr)


if __name__ == "__main__":
    # Ensure the device is set to CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    set_random_seed(42)
    env = SubprocVecEnv([make_env() for _ in range(1)])
    env = VecMonitor(env)

    model_path = "rocket_league_goal_agent2.zip"
    if os.path.exists(model_path):
        print(f"Loading the model from {model_path}...")
        model = PPO.load(model_path, env=env, device=device)  
    else:
        print("No saved model found. Initializing a new model...")

        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="./ppo_rlgym_logs/",
            learning_rate=lr_schedule_fn,  
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
            device="cpu"  
        )

    print("Training started...")
    model.learn(total_timesteps=10_000_000, callback=ProgressCallback(print_interval=10000))
    print("Training completed!")

    model.save("rocket_league_goal_agent2")
    print("Model updated and saved as rocket_league_goal_agent2.zip")

    env.close()
