from asyncio import queues
from distutils.command.config import config
from time import sleep
import queue

import carla
import random
import os
import yaml

def parse_config(path):
    """
    Parses the .yaml configuration file into a readable dictionary
    """
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config

def connect(config):
    os.system('sh ' + config['others']['carla_root'] + '/CarlaUE4.sh&')
    sleep(10)
    client = carla.Client('localhost', 2000)
    print("Connected to Carla")

    return client

def set_synchronous_mode(client):
    settings = client.get_world().get_settings()
    settings.fixed_delta_seconds = 0.05
    client.get_world().apply_settings(settings)
    settings = client.get_world().get_settings()
    settings.synchronous_mode = True
    client.get_world().apply_settings(settings)

def save_data(data, path, step):
    if not os.path.isdir(path):
        os.mkdir(path)
    data.save_to_disk(f'{path}/{step}.png')


def spawn_sensors(world, transform, sensor_config):
    sensors = []
    queues = []
    sensor_config = sensor_config.copy()
    
    for name in sensor_config.keys():
        blueprint = world.get_blueprint_library().find(sensor_config[name].pop('type'))
        for attribute in sensor_config[name].keys():
            blueprint.set_attribute(attribute, str(sensor_config[name][attribute]))
        
        sensor = world.spawn_actor(blueprint, transform)
        _queue = queue.Queue()
        sensor.listen(_queue.put)
        sensors.append((name, sensor))
        queues.append(_queue)

    return sensors, queues

def main():
    config = parse_config('collector_config.yaml')
    client = connect(config)
    step = 0
    towns = config['others']['maps']
    
    for t in towns:
        try:
            client.load_world(t)
            set_synchronous_mode(client)
        except:
            client = connect(config)
            client.load_world(t)
            set_synchronous_mode(client)
        world = client.get_world()
        map = world.get_map()
        waypoints = map.generate_waypoints(2)
        sensors, queues = spawn_sensors(world, waypoints[0].transform, config['sensors'])

        for wp in waypoints:
            for i in range(config['others']['samples_per_step']):
                x = random.uniform(*config['range']['x'])
                y = random.uniform(*config['range']['y'])
                z = random.uniform(*config['range']['z'])
                yaw = random.uniform(*config['range']['yaw'])
                roll = random.uniform(*config['range']['roll'])
                pitch = random.uniform(*config['range']['pitch'])

                wp_loc = wp.transform.location
                x_0, y_0, z_0 = wp_loc.x, wp_loc.y, wp_loc.z
                new_loc = carla.Location(x_0 + x, y_0 + y, z_0 + z)

                wp_rot = wp.transform.rotation
                pitch_0, yaw_0, roll_0 = wp_rot.pitch, wp_rot.yaw, wp_rot.roll
                new_rot = carla.Rotation(pitch_0 + pitch, yaw_0 + yaw, roll_0 + roll)

                new_transform = carla.Transform(new_loc, new_rot)
                
                for name, sensor in sensors:
                    sensor.set_transform(new_transform)
                
                world.tick()
                for idx, queue in enumerate(queues):
                    data = queue.get()
                    save_data(data, f'{config["others"]["save_path"]}/{sensors[idx][0]}', step)
                step += 1
        for name, sensor in sensors:
            sensors.destroy()

if __name__ == '__main__':
    main()