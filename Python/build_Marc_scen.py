import carla
import math
import numpy as np

# Connect to the CARLA server
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# Load a map
world = client.get_world()

# Set the initial position and orientation of the vehicle
spawn_point = carla.Transform(carla.Location(x=50, y=30, z=2), carla.Rotation(yaw=180))
vehicle_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# Set the initial position and orientation of the parking entry with a barrier
entry_point = carla.Transform(carla.Location(x=55, y=35, z=0), carla.Rotation(yaw=180))
entry_bp = world.get_blueprint_library().find('static.prop.building.parking_sign_01')
entry = world.spawn_actor(entry_bp, entry_point)

# Set the initial position and orientation of the camera
camera_point = carla.Transform(carla.Location(x=52, y=33, z=4), carla.Rotation(pitch=-30, yaw=180))
camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
camera = world.spawn_actor(camera_bp, camera_point, attach_to=vehicle)

# Set the initial position and orientation of the microphone
mic_point = carla.Transform(carla.Location(x=53, y=35, z=1), carla.Rotation(pitch=0, yaw=180))
mic_bp = world.get_blueprint_library().find('sensor.other.microphone')
mic = world.spawn_actor(mic_bp, mic_point)

# Save the current state of the world
world_snapshot = world.wait_for_tick()

# Run the simulation
while True:
    world.wait_for_tick()