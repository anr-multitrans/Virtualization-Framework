import os
import sys
import carla
from tools import move_view, build_projection_matrix, is_visible, bb_labels, semantic_tags
import random
import time
import json
from queue import Queue
from queue import Empty

selected_labels=['Bicycle','Bus','Car','Pole','Motorcycle','Rider','Pedestrians','traffic_light','traffic_sign','Truck', 'Static', 'Dynamic']
spawned_objects=[]
def spawn_objects(world, transform):
    try:
        # Define a distance of 20 meters in front of the spectator
        distance_in_front = 20  # You can adjust this distance as needed

        # Calculate the location for the random object
        forward_vector = transform.get_forward_vector()
        location_in_front = transform.location + distance_in_front * forward_vector

        
        # Create a random blueprint for the object
        blueprint_library = world.get_blueprint_library()
        available_blueprints = blueprint_library.filter("vehicle.*")  # You can change the filter to suit your object type
        random_blueprint = random.choice(available_blueprints)

        # Randomly move the object within a 10-meter radius
        random_offset = carla.Location(random.uniform(-5, 5), random.uniform(-5, 5), 0)
        object_location = location_in_front + random_offset

        # Randomly rotate the object
        random_rotation = carla.Rotation(yaw=random.uniform(0, 360), pitch=random.uniform(-15, 15), roll=random.uniform(-15, 15))

        # Spawn the random object with the new location and rotation
        spawned_object = world.spawn_actor(random_blueprint, carla.Transform(object_location, random_rotation))
        #semantic_tags =['static']
        #spawned_object.set_attribute('semantic_tags', semantic_tags)
        # Freeze the simulation for 1 second
        world.tick()
        time.sleep(1.0)

        # Destroy the spawned object
        spawned_objects.append(spawned_object)

    except Exception as e:
        print(f"Error: {e}")
def sensor_callback(sensor_data, sensor_queue, sensor_name, actor):
    #print(sensor_name)
    if(sensor_data.frame%20 == 0):
        all_actors = world.get_actors()

        #actors_with_specific_label = [actor for actor in all_actors if actor.attributes.get('semantic_description') == target_label]

        transform=actor.get_transform()
        K= build_projection_matrix(1200, 800, 90)

        bbs = { label: [(bb, bb.transform.location.distance(transform.location)) for bb in world.get_environment_objects(object_type=bb_labels[label]) 
        if is_visible(bb.bounding_box, transform, K)
        ] for label in selected_labels }
       
        for bb_actor in all_actors:
            # Check if any of the target tags are in the actor's semantic tags
            for target_tag in selected_labels:

                if semantic_tags[target_tag.lower()]['id'] in bb_actor.semantic_tags or (target_tag == 'Static' and len(bb_actor.semantic_tags)==0):
                    # Get the actor's transform and bounding box extent
                    
                    bounding_box = bb_actor.bounding_box
                    if True:#is_visible(bounding_box, transform, K):
                        # Calculate the distance from the actor to the bounding box
                        distance_to_bb = transform.location.distance(bounding_box.location)
                        # Append the bounding box and its location to the corresponding label in the dictionary
                        bbs[target_tag].append((bb_actor, distance_to_bb))
        for bb_actor in spawned_objects:
            if bb_actor not in all_actors:
                for target_tag in selected_labels:
                    if semantic_tags[target_tag.lower()]['id'] in bb_actor.semantic_tags:
                        # Get the actor's transform and bounding box extent
                        
                        bounding_box = bb_actor.bounding_box
                        if True:#is_visible(bounding_box, transform, K):
                            # Calculate the distance from the actor to the bounding box
                            distance_to_bb = transform.location.distance(bounding_box.location)
                            # Append the bounding box and its location to the corresponding label in the dictionary
                            bbs[target_tag].append((bb_actor, distance_to_bb))

        sensor_queue.put((sensor_data.frame, sensor_data, sensor_name,bbs,transform))  
        #if BB_3D_Mode and len(bbs)>0 :
        #     sensor_queue.put((sensor_data.frame, bbs, 'bbs'))
        print(bbs)
        for actor in spawned_objects:
            if all(element not in actor.semantic_tags for element in {6,7,8}):
            #if (6 not in actor.semantic_tags) and (7 not in actor.semantic_tags):
                print(f'{actor.id}:{actor.semantic_tags}')


os.makedirs('images_folder/rgb', exist_ok=True)
os.makedirs('images_folder/semantic_segmentation', exist_ok=True)
os.makedirs('images_folder/instance_segmentation', exist_ok=True)
os.makedirs('images_folder/simulation_objects',exist_ok=True)
os.makedirs('images_folder/labels', exist_ok=True)
scenario_length=100
max_distance = 50
client = carla.Client('localhost', 2000)
client.set_timeout(30.0)
client.load_world('Town12')
world= client.get_world()
settings = world.get_settings()
settings.fixed_delta_seconds = 0.2
settings.synchronous_mode = True
world.apply_settings(settings)
spectator = world.get_spectator()
print(spectator.get_transform())
spectator_location =carla.Location(x=148.278259, y=-83.464737, z=332.188904)
spectator_rotation = carla.Rotation(pitch=0, yaw=178.328812, roll=0)
#spectator_location =carla.Location(x=-666.873047, y=-480.319824, z=154.421280)
#spectator_rotation = carla.Rotation(pitch=-4.973757, yaw=-138.874847, roll=0.000212)
spectator_transform = carla.Transform(spectator_location, spectator_rotation)
spectator.set_transform(spectator_transform)

blueprint_library = world.get_blueprint_library()
l= carla.Location(x=-666.873047, y=-480.319824, z=154.421280)
r = carla.Rotation(pitch=-4.973757, yaw=-138.874847, roll=0.000212)
#l= carla.Location(x=-110.291763, y=97.093193, z=3.002939)
#r= carla.Rotation(pitch=-8.006648, yaw=66.636604, roll=0.000058)
vehicle_transform = carla.Transform(spectator_location,spectator_rotation)
# Create the sensor queue
#sensor_queue = Queue()
keywords= ['barrel', 'bin', 'clothcontainer','container','glasscontainer','box','trashbag','colacan','garbage','platformgarbage','trashcan','bench','gardenlamp','pergola','plasticchair','plastictable','slide','swing', 'swingcouch', 'table', 'trampoline','barbeque','clothesline','doghouse','gnome','wateringcan','haybale','plantpot','plasticbag','shoppingbag','shoppingcart','shoppingtrolley','briefcase','guitarcase','travelcase','helmet','mobile','purse','barrier','cone','ironplank','warning','brokentile','dirtdebris','foodcart','kiosk_01','fountain','maptable','advertisement','streetsign','busstop','atm','mailbox','streetfountain','vendingmachine','calibrator']
ego_bp = blueprint_library.find('vehicle.mercedes.sprinter')
ego_transform = carla.Transform(l, r)
ego_vehicle = world.spawn_actor(ego_bp, vehicle_transform)
sensor_params = {
        'rgb': {
            'type': 'sensor.camera.rgb',
            'attributes': {
                'image_size_x': 1820,
                'image_size_y': 1240,
                'fov': 90,
                'sensor_tick': 0.1
            }
        },
        'semantic_segmentation': {
            'type': 'sensor.camera.semantic_segmentation',
            'attributes': {
                'image_size_x': 1820,
                'image_size_y': 1240,
                'fov': 90,
                'sensor_tick': 0.1
            }
        },
        'instance_segmentation': {
            'type': 'sensor.camera.instance_segmentation',
            'attributes': {
                'image_size_x': 1820,
                'image_size_y': 1240,
                'fov': 90,
                'sensor_tick': 0.1
            }
        }
    }


# ...

# Spawn sensors on ego vehicle

sensor_list = []
sensor_queue = Queue()
for sensor_name, params in sensor_params.items():
    bp = blueprint_library.find(params['type'])
    bp.set_attribute('image_size_x', str(params['attributes']['image_size_x']))
    bp.set_attribute('image_size_y', str(params['attributes']['image_size_y']))
    bp.set_attribute('fov', str(params['attributes']['fov']))
    sensor_transform = carla.Transform(carla.Location(x=2.5, z=2.2))
    sensor = world.spawn_actor(bp, sensor_transform, attach_to=ego_vehicle)
    sensor_list.append((sensor_name, sensor))
    sensor.listen(lambda image, sensor_name=sensor_name: sensor_callback(image, sensor_queue, sensor_name,sensor))

spawn_objects(world, vehicle_transform)
ego_vehicle.set_autopilot(True)
target_tag="Static"
t=semantic_tags[target_tag.lower()]['id']
o=world.get_environment_objects(object_type=bb_labels[target_tag])
a={}
time.sleep(2)
all_actors = spawned_objects
for bb_actor in all_actors:
            # Check if any of the target tags are in the actor's semantic tags
    if t in bb_actor.semantic_tags:
        # Get the actor's transform and bounding box extent
        
        print(bb_actor)
#print(t)
#print(o)
for i in range(scenario_length):
    ego_vehicle.set_autopilot(True)
    transform=ego_vehicle.get_transform()
    w_frame = world.get_snapshot().frame
    print("\nWorld's frame: %d" % w_frame)
    #print(transform)
    if i%20==0:
        spawn_objects(world,transform)
    else:
        world.tick()
    objects= None
    image_rgb = None
    image_semantic = None
    image_instance = None
    camera_transform = None
    try:
        
        for _ in range(len(sensor_list)):
            s_frame, s_data, s_name,bbs_w, camera_transform = sensor_queue.get(True, 1.0)
            if s_name == 'rgb':
                image_rgb = s_data
            elif s_name == 'semantic_segmentation':
                image_semantic = s_data
                image_semantic.convert(carla.ColorConverter.CityScapesPalette)
                # Process semantic segmentation image
                #update_camera_view(image=image_semantic,view=semantic_label ,desired_width=image_width, desired_height=image_height,city_scape_convert=True
            elif s_name == 'instance_segmentation':
                image_instance = s_data
                # Process instance segmentation image
                #update_camera_view(image=image_instance,view=instance_label ,desired_width=image_width, desired_height=image_height)
                #    elif s_name == "bbs":
            if objects is None:
                objects= bbs_w

            



    except Empty:
        print("    Some of the sensor information is missed")

    if image_rgb is None:
        print("rgb")
    elif image_semantic is None:
        print("semantic_segmentation")
    elif image_instance is None:
        print("instance_segmentation")
    elif objects is None:
        print("bbs of simulation objects")
    else:
        image_instance.save_to_disk(f'images_folder/instance_segmentation/image_{w_frame}.png')
        image_semantic.save_to_disk(f'images_folder/semantic_segmentation/image_{w_frame}.png')
        image_rgb.save_to_disk(f'images_folder/rgb/image_{w_frame}.png')
        filtered_objects = {}
        for tag in objects:
            filtered_objects[tag]=[]
            for env_object,dist in objects[tag]:
                print(type(env_object))
                obj_name = env_object.name.lower()  # Convert object name to lowercase
                
                obj_transform = env_object.transform
                obj_location = obj_transform.location
                ray = obj_location - camera_transform.location
                obj_direction = obj_location -  camera_transform.location
                distance = dist#obj_location.distance(camera.get_transform().location) 

                if distance < max_distance and forward_vector.dot(obj_direction) > 0:
                    verts = [v for v in  env_object.bounding_box.get_world_vertices(carla.Transform())]
                    bounding_box_dict = {
                        'location': {
                            'x': env_object.bounding_box.location.x,
                            'y': env_object.bounding_box.location.y,
                            'z': env_object.bounding_box.location.z
                        },
                        'extent': {
                            'x': env_object.bounding_box.extent.x,
                            'y': env_object.bounding_box.extent.y,
                            'z': env_object.bounding_box.extent.z
                        },
                        'vertices':  [{'x': v.x, 'y': v.y, 'z': v.z} for v in verts]
                    }
                    filtered_objects[tag].append({
                        'id': env_object.id,
                        'name': env_object.name,
                        'type': env_object.type,
                        'distance': distance,
                        'bounding_box': bounding_box_dict,
                        'transform': {
                            'location': {
                                'x': obj_location.x,
                                'y': obj_location.y,
                                'z': obj_location.z
                            },
                            'rotation': {
                                'pitch': obj_transform.rotation.pitch,
                                'yaw': obj_transform.rotation.yaw,
                                'roll': obj_transform.rotation.roll
                            }
                        }
                    })

        # Store camera sensor's transform and parameters
        camera_transform_info = {
            'location': {
                'x': camera_transform.location.x,
                'y': camera_transform.location.y,
                'z': camera_transform.location.z
            },
            'rotation': {
                'pitch': camera_transform.rotation.pitch,
                'yaw': camera_transform.rotation.yaw,
                'roll': camera_transform.rotation.roll
            }
        }
        
        camera_params = sensor_params['rgb']['attributes']
        
        # Create a dictionary to hold all the information
        info_dict = {
            'camera_transform': camera_transform_info,
            'camera_parameters': camera_params,
            'filtered_objects': filtered_objects
        }
        
        # Save information to a JSON file
        with open(f'images_folder/simulation_objects/image_{w_frame}.json', 'w') as json_file:
            json.dump(info_dict, json_file, indent=4)
        

    move_view(world.get_spectator(),transform)
ego_vehicle.destroy()
for o in spawned_objects:
    o.destroy()

settings = world.get_settings()
settings.synchronous_mode = False
world.apply_settings(settings)



