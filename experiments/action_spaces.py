import carla
import gym
import numpy as np

def discrete_actions(experiment, num_throttles, num_steers):
    def compute_action(self, action):
        steer = action // num_throttles
        throttle = action % num_throttles

        control = carla.VehicleControl()
        control.steer = -1 + (steer / (num_steers - 1)) * 2
        if throttle == 0:
            #control.brake = 1
            control.throttle = 0
            control.brake = 0
        else:
            control.brake = 0
            control.throttle = throttle / (num_throttles - 1)
        
        control.brake = 0
        control.reverse = False
        control.hand_brake = False

        self.last_action = control
        self.verbose = False
        if self.verbose:
            print(f"Action: {action} ({throttle}, {steer}) Control: ", control)
        return control

    
    def get_action_space(self):
        num_actions = num_steers * num_throttles
        action_space = gym.spaces.Discrete(num_actions)

        return action_space 
    
    experiment.compute_action = compute_action
    experiment.get_action_space = get_action_space

    return experiment

def continuous_actions(experiment):
    def get_action_space(self):
        action_space = gym.spaces.Box(
            np.array([-1, 0,]).astype(np.float32),
            np.array([+1, +1,]).astype(np.float32),
        )

        return action_space
    
    def compute_action(self, action):
        control = carla.VehicleControl()
        control.steer = action[0].item()
        control.throttle = action[1].item()
        control.brake = 0
        control.reverse = False
        control.hand_brake = False

        self.last_action = control

        return control
    
    experiment.compute_action = compute_action
    experiment.get_action_space = get_action_space

    return experiment