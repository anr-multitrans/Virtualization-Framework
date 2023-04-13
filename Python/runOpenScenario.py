import carla
import time

# Connect to the Carla server
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# Load the OpenScenario file
with open('your_scenario_file', 'r') as f:
    scenario = f.read()

# Get the world and map
world = client.get_world()
carla_map = world.get_map()

# Get the blueprint for the ego vehicle
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('vehicle.audi.etron')[0]

# Spawn the ego vehicle
spawn_point = carla_map.get_spawn_points()[0]
ego_vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# Create the OpenScenario instance and start it
scenario_instance = world.apply_scenario(scenario)
scenario_instance.start()

# Wait for the scenario to finish
while not scenario_instance.is_running():
    time.sleep(0.1)
while scenario_instance.is_running():
    time.sleep(0.1)

# Cleanup
scenario_instance.stop()
ego_vehicle.destroy()
