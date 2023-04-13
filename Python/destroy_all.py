import carla

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# Get the world object
world = client.get_world()

# Get a list of all actors in the simulation
actor_list = world.get_actors()

# Loop through the list of actors and destroy them
for actor in actor_list:
    actor.destroy()