#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import math
import numpy as np
from gym.spaces import Box, Discrete

import carla

from rllib_integration.base_experiment import BaseExperiment
from rllib_integration.helper import post_process_image

def distance_vehicle(waypoint, vehicle_transform):
    """
    Returns the 2D distance from a waypoint to a vehicle
        :param waypoint: actual waypoint
        :param vehicle_transform: transform of the target vehicle
    """
    loc = vehicle_transform.location
    x = waypoint.transform.location.x - loc.x
    y = waypoint.transform.location.y - loc.y

    return math.sqrt(x * x + y * y)


class Experiment(BaseExperiment):
    def __init__(self, config={}):
        super().__init__(config)  # Creates a self.config with the experiment configuration

        self.frame_stack = self.config["others"]["framestack"]
        self.max_time_idle = self.config["others"]["max_time_idle"]
        self.max_time_episode = self.config["others"]["max_time_episode"]
        self.allowed_types = [carla.LaneType.Driving, carla.LaneType.Parking]
        self.last_heading_deviation = 0
        self.last_action = None

    def reset(self):
        """Called at the beginning and each time the simulation is reset"""

        # Ending variables
        self.time_idle = 0
        self.time_episode = 0
        self.done_time_idle = False
        self.done_falling = False
        self.done_time_episode = False

        # hero variables
        self.last_location = None
        self.last_velocity = 0

        # Sensor stack
        self.prev_image_0 = None
        self.prev_image_1 = None
        self.prev_image_2 = None

        self.last_heading_deviation = 0

    def get_action_space(self):
        """Returns the action space, in this case, a discrete space"""
        return Discrete(len(self.get_actions()))

    def get_observation_space(self):
        num_of_channels = 3
        image_space = Box(
            low=0.0,
            high=255.0,
            shape=(
                self.config["hero"]["sensors"]["birdview"]["size"],
                self.config["hero"]["sensors"]["birdview"]["size"],
                num_of_channels * self.frame_stack,
            ),
            dtype=np.uint8,
        )
        return image_space

    def get_actions(self):
        return {
            0: [0.0, 0.00, 0.0, False, False],  # Coast
            1: [0.0, 0.00, 1.0, False, False],  # Apply Break
            2: [0.0, 0.75, 0.0, False, False],  # Right
            3: [0.0, 0.50, 0.0, False, False],  # Right
            4: [0.0, 0.25, 0.0, False, False],  # Right
            5: [0.0, -0.75, 0.0, False, False],  # Left
            6: [0.0, -0.50, 0.0, False, False],  # Left
            7: [0.0, -0.25, 0.0, False, False],  # Left
            8: [0.3, 0.00, 0.0, False, False],  # Straight
            9: [0.3, 0.75, 0.0, False, False],  # Right
            10: [0.3, 0.50, 0.0, False, False],  # Right
            11: [0.3, 0.25, 0.0, False, False],  # Right
            12: [0.3, -0.75, 0.0, False, False],  # Left
            13: [0.3, -0.50, 0.0, False, False],  # Left
            14: [0.3, -0.25, 0.0, False, False],  # Left
            15: [0.6, 0.00, 0.0, False, False],  # Straight
            16: [0.6, 0.75, 0.0, False, False],  # Right
            17: [0.6, 0.50, 0.0, False, False],  # Right
            18: [0.6, 0.25, 0.0, False, False],  # Right
            19: [0.6, -0.75, 0.0, False, False],  # Left
            20: [0.6, -0.50, 0.0, False, False],  # Left
            21: [0.6, -0.25, 0.0, False, False],  # Left
            22: [1.0, 0.00, 0.0, False, False],  # Straight
            23: [1.0, 0.75, 0.0, False, False],  # Right
            24: [1.0, 0.50, 0.0, False, False],  # Right
            25: [1.0, 0.25, 0.0, False, False],  # Right
            26: [1.0, -0.75, 0.0, False, False],  # Left
            27: [1.0, -0.50, 0.0, False, False],  # Left
            28: [1.0, -0.25, 0.0, False, False],  # Left
        }

    def compute_action(self, action):
        """Given the action, returns a carla.VehicleControl() which will be applied to the hero"""
        action_control = self.get_actions()[int(action)]

        action = carla.VehicleControl()
        action.throttle = action_control[0]
        action.steer = action_control[1]
        action.brake = action_control[2]
        action.reverse = action_control[3]
        action.hand_brake = action_control[4]

        self.last_action = action

        return action

    def get_observation(self, sensor_data):
        """Function to do all the post processing of observations (sensor data).

        :param sensor_data: dictionary {sensor_name: sensor_data}

        Should return a tuple or list with two items, the processed observations,
        as well as a variable with additional information about such observation.
        The information variable can be empty
        """
        image = post_process_image(sensor_data['birdview'][1], normalized = False, grayscale = False)

        if self.prev_image_0 is None:
            self.prev_image_0 = image
            self.prev_image_1 = self.prev_image_0
            self.prev_image_2 = self.prev_image_1

        images = image

        if self.frame_stack >= 2:
            images = np.concatenate([self.prev_image_0, images], axis=2)
        if self.frame_stack >= 3 and images is not None:
            images = np.concatenate([self.prev_image_1, images], axis=2)
        if self.frame_stack >= 4 and images is not None:
            images = np.concatenate([self.prev_image_2, images], axis=2)

        self.prev_image_2 = self.prev_image_1
        self.prev_image_1 = self.prev_image_0
        self.prev_image_0 = image

        return images, {}

    def get_speed(self, hero):
        """Computes the speed of the hero vehicle in Km/h"""
        vel = hero.get_velocity()
        return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

    def get_done_status(self, observation, core):
        """Returns whether or not the experiment has to end"""
        hero = core.hero
        self.done_time_idle = self.max_time_idle < self.time_idle
        if self.get_speed(hero) > 1.0:
            self.time_idle = 0
        else:
            self.time_idle += 1
        self.time_episode += 1
        self.done_time_episode = self.max_time_episode < self.time_episode
        self.done_falling = hero.get_location().z < -0.5
        return self.done_time_idle or self.done_falling or self.done_time_episode

    def compute_reward(self, observation, core):
        """Computes the reward"""
        def unit_vector(vector):
            return vector / np.linalg.norm(vector)
        def speed(hero):
            vel = hero.get_velocity()
            return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
        
        vel_r, heading_r, dist_r = 0, 0, 0

        world = core.world
        hero = core.hero
        map = core.map
        
        hero_loc = hero.get_location()
        hero_speed = speed(hero)

        closest_wp = map.get_waypoint(
                hero_loc,
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )
        
        dist = distance_vehicle(closest_wp, hero.get_transform())
        dist_r = -(dist / self.config["reward"]["max_dist"])
         
        if dist > self.config["reward"]["max_dist"]:
            self.off_route = True
            return -1
        
        hero_heading = hero.get_transform().get_forward_vector()
        hero_heading = np.array([
            hero_heading.x,
            hero_heading.y,
        ])

        wp_heading = closest_wp.transform.get_forward_vector()
        wp_heading = np.array([
            wp_heading.x,
            wp_heading.y,
        ])

        if hero_heading.dot(wp_heading) < 0:
            # Car is heading in the wrong direction
            return -1
        else:
            hero_u = unit_vector(hero_heading)
            wp_u = unit_vector(wp_heading)
            delta_angle = np.arccos(np.clip(np.dot(hero_u, wp_u), -1.0, 1.0))
            heading_r = 1 - delta_angle / (np.pi/2)
        
        vel_r = min(hero_speed/self.config["reward"]["max_speed"], 1)

        #print(((vel_r + heading_r ) / 2) + dist_r)
        return ((vel_r + heading_r ) / 2) + dist_r