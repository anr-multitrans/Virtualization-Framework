import carla

# Connect to the CARLA server and get a reference to the world object
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# Get the bounding box of the first Car object in the world
bounding_box_set = world.get_level_bbs(carla.CityObjectLabel.Any)


# Define the filter to retrieve actors within the bounding box
filter = 'vehicle*'  # or 'walker.*', 'traffic_light.*', etc.

# Retrieve actors that match the filter
actor_list = world.get_actors().filter(filter)
print(actor_list)
for bbox in bounding_box_set:
# Loop through the filtered actors and print the blueprint for each actor within the bounding box
    for actor in actor_list:
        type_id = actor.type_id
        print(type_id)
        if  type_id!='spectator':
            
            blueprint_library = world.get_blueprint_library()
            blueprint = blueprint_library.find(type_id)
            print(f"Actor {actor.id} has blueprint: {blueprint}")
            
            # Get the actor's location and check if it is within the bounding box
            location = actor.get_location()
            transform = actor.get_transform()
            if bbox.contains(location, transform):
                print(f"Actor {actor.id} is within the bounding box.")
            

#This code retrieves all the actors in the world that match the given filter, and then checks if 