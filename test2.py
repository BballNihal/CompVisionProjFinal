import rlgym
from rlgym.utils.obs_builders import DefaultObs
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym.utils.reward_functions.common_rewards import VelocityPlayerToBallReward
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.action_parsers.default_act import DefaultAction
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import numpy as np
import random


class GoalReward(RewardFunction):
    def reset(self, initial_state, optional_data=None):
        self.previous_ball_position = initial_state.ball.position

    def get_reward(self, player, state, previous_action):
        goal_position = np.array([0, 5120, 0]) 
        ball_position = state.ball.position
        distance_to_goal = np.linalg.norm(goal_position - ball_position)

        reward = -distance_to_goal

        
        if ball_position[1] > 5100: 
            reward += 100

        self.previous_ball_position = ball_position
        return reward


class ActionRepeatParser(DefaultAction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_buffer = None
        self.action_repeat_counter = 0
        self.repeat_duration = 15  

    def get_action(self, state):
        if self.action_repeat_counter == 0:
            
            self.action_buffer = super().get_action(state)  
        self.action_repeat_counter += 1

        if self.action_repeat_counter >= self.repeat_duration:
            self.action_repeat_counter = 0  

        return self.action_buffer  


# Environment setup
def make_env():
    def _init():
        return rlgym.make(
            reward_fn=GoalReward(),
            obs_builder=DefaultObs(),  
            action_parser=ActionRepeatParser(),  
            terminal_conditions=[TimeoutCondition(300), GoalScoredCondition()],  
        )
    return _init


if __name__ == "__main__":
    
    env = SubprocVecEnv([make_env() for _ in range(1)])  
    env = VecMonitor(env)  

    
    model = PPO(
        "MlpPolicy", 
        env,
        verbose=1,  
        tensorboard_log="./ppo_rlgym_logs/",  
    )

    class ProgressCallback:
        def __init__(self, print_interval=10000):
            self.print_interval = print_interval

        def __call__(self, locals_, globals_):
            if locals_['self'].num_timesteps % self.print_interval == 0:
                print(f"Progress: {locals_['self'].num_timesteps} timesteps")
            return True


    progress_callback = ProgressCallback(print_interval=10000)

 
    print("Training started...")
    model.learn(total_timesteps=1_000_000, callback=progress_callback)  
    print("Training completed!")


    model.save("rocket_league_goal_agent")
    print("Model saved as rocket_league_goal_agent.zip")

    env.close()
