import carla
import time

def move_spectator(world, location, orientation):
    spectator = world.get_spectator()
    new_transform = carla.Transform(location,orientation)
    spectator.set_transform(new_transform)
    print("scenario is at")
    print(spectator.get_transform())

def move_vehicle_to_location(vehicle_actor, new_location):
    print(vehicle_actor.get_transform())
    vehicle_actor.set_transform(carla.Transform(new_location, vehicle_actor.get_transform().rotation))

# Example usage
client = carla.Client('localhost', 2000)
client.set_timeout(5.0)
world = client.get_world()

# Spawn a vehicle
spawn_point = carla.Transform(carla.Location(x=-50, y=30, z=2))
spectator_point= carla.Location(x=-50, y=30, z=1)
move_spectator(world,spectator_point, spawn_point.rotation )
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
vehicle_actor = world.spawn_actor(vehicle_bp, spawn_point)
time.sleep(10)
# Check if the vehicle was spawned successfully
if vehicle_actor is not None:
    # Move the vehicle to a new location
    new_location = carla.Location(x=-55, y=45, z=2)  # Specify the desired new location
    move_vehicle_to_location(vehicle_actor, new_location)
    print("new_location")
    print(vehicle_actor.get_transform())
else:
    print("Failed to spawn the vehicle.")

time.sleep(10)
vehicle_actor.destroy()

# Rest of the code...
