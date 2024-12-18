import numpy as np
import rlgym
from rlgym.utils.obs_builders.advanced_obs import AdvancedObs
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.terminal_conditions import TerminalCondition
from rlgym.utils.action_parsers import ActionParser
from rlgym.utils.state_setters import StateSetter
from stable_baselines3 import PPO

import torch
import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import random

#Get all possible actions. discrete actions with parsed flip. flip/jump not necessary for dribbling. needless complication. removed it
class MaskedActionParser(ActionParser):
    def __init__(self, base_parser):
        super().__init__()
        self.base_parser = base_parser

    def parse_actions(self, actions, state):
        actions = np.array(actions)  
        
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        
        # action masks
        mask = np.ones_like(actions)
        mask[:, 5] = 0  # Jump mask
        # mask[:, 6] = 0  # boost mask
        
        filtered_actions = actions * mask
        
        # Use the base parser to convert filtered actions
        return self.base_parser.parse_actions(filtered_actions, state)

    def get_action_space(self):
        return self.base_parser.get_action_space()

# Custom reward function for encouraging dribbling
class DribbleReward(RewardFunction):
    def __init__(self):
        self.episode_length = 0
        self.total_reward = 0

    def reset(self, initial_state):
        self.episode_length = 0
        self.total_reward = 0

    def get_reward(self, player, state, previous_action):
        ball_velocity = state.ball.linear_velocity
        car_to_ball = state.ball.position - player.car_data.position
        car_to_ball_distance = np.linalg.norm(car_to_ball)
        reward = 0

        # ball pocession reward
        if car_to_ball_distance < 110:
            reward += 1 - (car_to_ball_distance / 110)  

        # far penalty
        reward -= 1*((car_to_ball_distance - 110) / 110)  

        # Dribbling reward? Maybe we use this once car knows how to dribble a bit already. Too soon.
        # reward += np.dot(player.car_data.linear_velocity, ball_velocity) / 1000

        # Negative reward for staying still
        car_speed = np.linalg.norm(player.car_data.linear_velocity)
        stillness_penalty = -2 * max(0, 1 - car_speed / 100)
        reward += stillness_penalty

        # Episode duration based reward. Made this quadratic so it has more weight than the other rewards.
        self.episode_length += 1
        reward += self.episode_length 

        self.total_reward += reward
        return reward

# terminal condition to terminate when ball touches ground
class BallTouchGroundCondition(TerminalCondition):
    def reset(self, initial_state):
        pass

    def is_terminal(self, current_state):
        return current_state.ball.position[2] <= 107.5 #experimentally found ball radius to be 107.5

# Custom state setter // This is old. needs a bit more complexity. car just drive in a straight line.
# class BallAboveCarStateSetter(StateSetter):
#     def reset(self, state_wrapper):
#         state_wrapper.ball.set_pos(0, 0, 210)
#         state_wrapper.ball.set_lin_vel(0, 0, 0)
#         state_wrapper.ball.set_ang_vel(0, 0, 0)

#         for car in state_wrapper.cars:
#             car.set_pos(0, 0, 17)
#             car.set_lin_vel(0, 0, 0)
#             car.set_ang_vel(0, 0, 0)


# New state setter. Idea here is to give the car an uncomfortable situation to start with so that it learns to adapt to different ways ball falls
class BallAboveCarStateSetter(StateSetter):
    def reset(self, state_wrapper):
        # Randomly place the ball somewhere on the nose of the car. if too far back, env is impossible. previous random was -70 to 70, changed it because it might have been making impossible envs
        random_x = random.uniform(-50, 50)
        state_wrapper.ball.set_pos(random_x, 100, 200)
        state_wrapper.ball.set_lin_vel(0, 0, 0)
        state_wrapper.ball.set_ang_vel(0, 0, 0)

        # car position/rotation
        for car in state_wrapper.cars:
            car.set_pos(0, 0, 17)
            car.set_lin_vel(0, 0, 0)
            car.set_ang_vel(0, 0, 0)

            # rotating so its facing goal // not that it matters here
            car.set_rot(0, np.pi/2, 0) 
            car.boost = 1

# to log progress. using tensorboard
class ProgressCallback:
    def __init__(self, print_interval=10000, log_dir="./ppo_rlgym_dribble9"):
        self.print_interval = print_interval
        self.writer = SummaryWriter(log_dir)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        self.total_steps = 0

    def __call__(self, locals_, globals_):
        if 'self' in locals_ and hasattr(locals_['self'], 'num_timesteps'):
            model = locals_['self']
            self.total_steps = model.num_timesteps

            # episode metrics
            if 'infos' in locals_:
                infos = locals_['infos']
                for info in infos:
                    if 'episode' in info:  
                        self.episode_rewards.append(info['episode']['r'])  # Episode reward
                        self.episode_lengths.append(info['episode']['l'])  # Episode length
                        self.episode_count += 1

                        self.writer.add_scalar("Episode/Reward", info['episode']['r'], self.episode_count)
                        self.writer.add_scalar("Episode/Length", info['episode']['l'], self.episode_count)

            # Printing extra params fro debugging purposes
            if self.total_steps % self.print_interval == 0:
                print(f"Progress: {self.total_steps} timesteps")
                current_lr = model.policy.optimizer.param_groups[0]['lr']
                print(f"Current Learning Rate: {current_lr}")
                avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
                avg_length = np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0
                print(f"Average Reward (last 100 episodes): {avg_reward}")
                print(f"Average Length (last 100 episodes): {avg_length}")

                # Log averages to TensorBoard
                self.writer.add_scalar("Episode/Average_Reward", avg_reward, self.total_steps)
                self.writer.add_scalar("Episode/Average_Length", avg_length, self.total_steps)

        return True
    
    
# learning rate scheduler // stopped using this because it just makes more sense to manually do this sense working with large amounts of steps
def lr_schedule_fn(current_timestep: int) -> float:
    initial_lr = 1e-3
    final_lr = 1e-5
    total_timesteps = 1000000

    # Linear decay
    lr = initial_lr * (1 - current_timestep / total_timesteps)
    
    # Ensure that the learning rate never goes below final_lr
    return max(lr, final_lr)

# Creating the RLGym environment
def create_env():
    # start with base discrete action parser from rlgym library
    base_action_parser = rlgym.utils.action_parsers.DiscreteAction()
    
    # wrap it with the masked version
    masked_action_parser = MaskedActionParser(base_action_parser)
    
    return rlgym.make(
        obs_builder=AdvancedObs(),
        reward_fn=DribbleReward(),
        terminal_conditions=[BallTouchGroundCondition()],
        action_parser=masked_action_parser,
        state_setter=BallAboveCarStateSetter(),
        game_speed=100,
)

def main():
    # environment
    env = create_env()
    model_path = "ppo_dribble_agent_no_jump4.zip"
    if os.path.exists(model_path):
        custom_objects = { 'learning_rate': 3e-4 }
        model = PPO.load(model_path, env=env, device="cpu", custom_objects=custom_objects)
    else:
        print("No saved model found. Initializing a new model...")
    # defining the PPO model
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="./ppo_rlgym_dribble",
            learning_rate=1e-3,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
            device="gpu"
        )

    # logging/graphs set
    progress_callback = ProgressCallback(print_interval=10000)

    # training model! yippie!
    model.learn(total_timesteps=50000, callback=progress_callback)
    
    # Save model
    model.save("ppo_dribble_agent_no_jump4")

    # Close environment
    env.close()

if __name__ == "__main__":
    main()