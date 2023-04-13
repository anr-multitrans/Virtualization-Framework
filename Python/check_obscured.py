import carla

# Connect to the Carla server
client = carla.Client('localhost', 2000)
client.set_timeout(20.0)

# Get the world object
world = client.get_world()

# Define the start and end points
start_point = carla.Location(x=0, y=0, z=2)
end_point = carla.Location(x=100, y=0, z=2)

# Create bounding boxes for the start and end points
start_bbox = carla.BoundingBox(start_point, carla.Vector3D(x=1, y=1, z=1))
end_bbox = carla.BoundingBox(end_point, carla.Vector3D(x=1, y=1, z=1))

# Get all actors in the world
actors = world.get_actors()

# Loop through all actors and check if their bounding box overlaps with the start and end points
for actor in actors:
    if actor.bounding_box.overlap(start_bbox) and actor.bounding_box.overlap(end_bbox):
        print(actor.type_id, actor.bounding_box.location)