import carla
from carla import LaneType
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

world = client.get_world()
carla_map = world.get_map()
topology = carla_map.get_topology()

# Iterate over the waypoints and identify the static objects
for waypoint_tuple in topology:
    waypoint = waypoint_tuple[0]
    if True: #waypoint.lane_type == LaneType.Static or waypoint.lane_type == LaneType.Parking:
        # Print information about the static object
        print(f"Object type: {waypoint.lane_type} | Location: {waypoint.transform.location} | Rotation: {waypoint.transform.rotation}")

# Get all actors in the world
actors = world.get_actors()
for actor in actors:
    if 'vehicle' in actor.type_id:
        print('Vehicle:', actor.id, actor.get_transform().location, actor.type_id, actor.bounding_box)
    elif 'traffic' in actor.type_id:
        print('Traffic Sign:', actor.id, actor.get_transform().location, actor.type_id, actor.bounding_box)
    elif 'static' in actor.type_id:
        print('Static Object:', actor.id, actor.get_transform().location, actor.type_id, actor.bounding_box)
    else:
        print('Other Object:', actor.id, actor.get_transform().location, actor.type_id, actor.bounding_box)





vehicles=world.get_actors().filter('vehicle*')
actors = world.get_actors().filter('traffic*')
#actors.extend(vehicles)
print(actors)
print(vehicles)
# Print information for each actor
for actor in actors:
    # Get actor's location and bounding box
    location = actor.get_location()
    bbox = actor.bounding_box

    # Get actor's blueprint id
    if hasattr(actor, 'attributes'):
        # For dynamic objects
        blueprint_id = actor.attributes.get('object_type', 'unknown')
    else:
        # For static objects
        blueprint_id = actor.type_id
    # Print actor information
    print('Actor ID: {}, Blueprint ID: {}, Location: {}, Bounding Box: {}'.format(actor.id, blueprint_id, location, bbox))

for actor in vehicles:
    # Get actor's location and bounding box
    location = actor.get_location()
    bbox = actor.bounding_box

    # Get actor's blueprint id
    if hasattr(actor, 'attributes'):
        # For dynamic objects
        blueprint_id = actor.attributes.get('object_type', 'unknown')
    else:
        # For static objects
        blueprint_id = actor.type_id

    # Print actor information
    print('Actor ID: {}, Blueprint ID: {}, Location: {}, Bounding Box: {}'.format(actor.id, blueprint_id, location, bbox))