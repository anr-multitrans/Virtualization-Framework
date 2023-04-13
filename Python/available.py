import carla

# Connect to the CARLA server
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# Get the blueprint library
blueprint_library = client.get_world().get_blueprint_library()

# Print the names of all available vehicle blueprints
for blueprint in blueprint_library:
    print(blueprint.id)