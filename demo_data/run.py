from agents.navigation.basic_agent import BasicAgent
from time import sleep
import queue

import carla
import random
import os
import glob
import yaml

def parse_config(path):
    """
    Parses the .yaml configuration file into a readable dictionary
    """
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config

config = parse_config('config.yaml')

# current_town = towns[0]
rgb_queue = queue.Queue()

def connect():
    os.system('sh ' + config['CARLA_ROOT'] + '/CarlaUE4.sh&')
    sleep(10)
    client = carla.Client('localhost', 2000)
    print("Connected to Carla")

    return client

def set_synchronous_mode(client):
    settings = client.get_world().get_settings()
    settings.fixed_delta_seconds = 0.05
    client.get_world().apply_settings(settings)
    settings = client.get_world().get_settings()
    settings.synchronous_mode = True # Enables synchronous mode
    client.get_trafficmanager().set_synchronous_mode(True)
    client.get_world().apply_settings(settings)

def save_data(data, path, step):
    data.save_to_disk(f'{path}/{step}.png')

def spawn_agent(world):
    vehicle = world.get_blueprint_library().find('vehicle.tesla.model3')
    spawn = world.get_map().get_spawn_points()[1]
    vehicle = world.spawn_actor(vehicle, spawn)
    agent = BasicAgent(vehicle)
    return vehicle, agent, spawn

def spawn_sensors(world, vehicle):
    blueprint = world.get_blueprint_library().find('sensor.camera.rgb')    
    blueprint.set_attribute('image_size_x', str(config['img_width']))
    blueprint.set_attribute('image_size_y', str(config['img_height']))
    blueprint.set_attribute('fov', '100')
    relative_transform = carla.Transform(
        carla.Location(1.7, 0, 1.2),
        carla.Rotation(0, 0, 0)
    )
    print(relative_transform)  
    rgb = world.spawn_actor(blueprint, relative_transform, attach_to=vehicle)
    rgb.listen(rgb_queue.put)

    return rgb

client = connect()
client.load_world(config['map'])
set_synchronous_mode(client)
world = client.get_world()
map = world.get_map()
print(map)

vehicle, agent, spawn = spawn_agent(world)
rgb = spawn_sensors(world, vehicle)
destination = random.choice(world.get_map().get_spawn_points()).location
#agent.set_destination((destination.x, destination.y, destination.z))
vehicle.set_autopilot(True)
tm = client.get_trafficmanager(8000)
if config['ignore_traffic_lights']:
    tm.ignore_lights_percentage(vehicle, 100)

world.tick()

csv_entries = []

for step in range(config['steps']):
    spectator = world.get_spectator()
    transform = vehicle.get_transform()
    spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50),
        carla.Rotation(pitch=-90)))
    rgb_data = rgb_queue.get()
    # control = agent.run_step()
    # control.manual_gear_shift = False
    # vehicle.apply_control(control)
    control = vehicle.get_control()
    csv_entry = f'{control.throttle}, {control.steer}, {control.brake}\n'
    csv_entries.append(csv_entry)
    save_data(rgb_data, f'{config["path"]}', step)
    world.tick()

f = open(f"{config['path']}/actions.csv", "w")
f.write(''.join(csv_entries))
f.close()

rgb.destroy()
vehicle.destory()

print("DONE")