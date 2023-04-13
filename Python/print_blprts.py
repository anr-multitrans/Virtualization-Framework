import carla

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

world = client.get_world()
blueprint_library = world.get_blueprint_library()

# print the available blueprints
blueprints = blueprint_library#.filter('misc')
for blueprint in blueprints:
    print(blueprint.id)

# add a building
building_bp = blueprint_library.find('static.building1')
spawn_transform = carla.Transform(carla.Location(x=50, y=50, z=0))
building = world.spawn_actor(building_bp, spawn_transform)