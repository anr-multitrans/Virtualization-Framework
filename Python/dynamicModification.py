import carla
import time
import random
# Connect to the simulator
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# Get the blueprint for a building
blueprint_library = world.get_blueprint_library()
pedestrian_bp = blueprint_library.find('walker.pedestrian.0021')
building_bp = blueprint_library.find('static.prop.atm')
building5_bp = blueprint_library.find('static.prop.fountain')

# Get the current player
player = world.get_spectator()

actors = world.get_actors()

# find the actor you want to destroy
#actor_to_destroy = None
#actor_to_keep=None
#for actor in actors:
#	print(actor)
#	if actor.id == 81:
#		actor_to_keep = actor

# find the actor you want to destroy
#actor_to_keep.destroy()
#actor_to_destroy.destroy()

# Spawn the building
building_transform= carla.Transform(carla.Location(x=35,y=35,z=0))
walker_transform = carla.Transform(carla.Location(x=38, y=38, z=0))
actor = world.spawn_actor(pedestrian_bp, walker_transform)

# Move the player to view the building
#player.set_transform(building_transform)
#player_transform = carla.Transform(carla.Location(x=-20, y=-20, z=20))

# Wait for a few seconds
time.sleep(5)
target_location = random.choice(world.get_map().get_spawn_points()).location

# Continuously update the pedestrian's location until it reaches the target location
control = carla.WalkerControl()

# Set the speed and direction of the pedestrian
control.speed = 1.0  # meters per second
control.direction = carla.Vector3D(x=1, y=0, z=0)  # walking in x direction

# Apply the control to the pedestrian
actor.apply_control(control)
#pedestrian_control = carla.WalkerControl()
#pedestrian_control.speed = 1.0  # Set the speed to 1 m/s
#building_actor.apply_walk_control(pedestrian_control)
#for i in range(100):
#	actor_transform = building_actor.get_transform()
#	new_location = actor_transform.location + carla.Location(x=0.2)
#	new_transform = carla.Transform(new_location, actor_transform.rotation)
#	building_actor.set_transform(new_transform)
#	time.sleep(0.2)
# Destroy the building
time.sleep(10)
actor.destroy()

# Move the player back to the original position
#player_transform = carla.Transform(carla.Location(x=-40, y=-40, z=0))
#player.set_transform(player_transform)

# Wait for a few seconds
time.sleep(5)

# Spawn the building again
building_actor = world.spawn_actor(building5_bp, carla.Transform(carla.Location(x=44, y=44, z=0)))

# Move the player to view the building
# player.set_transform(building_transform)

# Wait for a few seconds
time.sleep(5)

# Destroy the building again
building_actor.destroy()

# Move the player back to the original position
# player.set_transform(player_transform)

# Wait for a few seconds
time.sleep(5)

# End the simulation
