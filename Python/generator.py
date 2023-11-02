import os
import sys
import carla
from tools import move_view, semantic_tags, load_simulation_config, check_carla_server, launch_carla_server, compare_images, merge_segmentation_images, find_color_regions_contours
import random
import time
import json
from queue import Queue
from queue import Empty
import threading
import numpy as np
import cv2
from datetime import datetime
import argparse

global_vars = {'camera_ready': False}

# Get the current date and time
current_datetime = datetime.now()

# Extract the Unix timestamp as an integer
unix_timestamp = int(current_datetime.timestamp())

#selected_labels=['Bicycle','Bus','Car','Pole','Motorcycle','Rider','Pedestrians','traffic_light','traffic_sign','Truck', 'Static', 'Dynamic']
spawned_objects=[]

motorcycles = ["vehicle.harley-davidson.low_rider", "vehicle.kawasaki.ninja","vehicle.vespa.zx125","vehicle.yamaha.yzf"]
bicycles = ["vehicle.bh.crossbike", "vehicle.diamondback.century","vehicle.gazelle.omafiets"]

blueprints_to_exclude = motorcycles + bicycles
sensor_queue = Queue()

#spawned_objects = []
def init_carla():
        i = 0
        client = check_carla_server()
        if client is None:
            print("Starting Carla Server")
            t = threading.Thread(target=launch_carla_server)
            t.start()

        while client is None and i < 50:
            i += 1
            client = check_carla_server()
            time.sleep(1)
        if client is None:
            print("Impossible to connect: Please check that Carla is installed correctly and try again later")
        else:
            world = client.get_world()
            print("connected")

def try_to_spawn_object(world, location, offset_range, category ):
    blueprints=[]
    blueprint_library = world.get_blueprint_library()
    vehicle_blueprints = blueprint_library.filter('vehicle.*')
    offset_z=0
    # Filter out the blueprints 
    
    if category == "vehicle":
        blueprints = [bp for bp in vehicle_blueprints if bp.id not in blueprints_to_exclude]
        offset_z = 0.5
        #print(blueprints)
    elif category == "motorcycle":
        blueprints = [bp for bp in vehicle_blueprints if bp.id in motorcycles]
        #offset_z = 1
    elif category == "bicycle":
        blueprints = [bp for bp in vehicle_blueprints if bp.id in bicycles]
        #offset_z = 1
    elif category == "pedestrian":
        blueprints = blueprint_library.filter('walker.*')
    elif category == "other":
        blueprints = blueprint_library.filter('static.prop*')
    else:

        blueprints = blueprint_library.filter(f"*{category}*")


    random_blueprint = random.choice(blueprints)

    # Randomly move the object within a 10-meter radius
    random_offset = carla.Location(random.uniform(-offset_range, offset_range), random.uniform(-offset_range, offset_range), offset_z)
    object_location = location + random_offset

    # Randomly rotate the object
    random_rotation = carla.Rotation(yaw=random.uniform(0, 360), pitch=random.uniform(-5, 5), roll=random.uniform(-5, 5))
    #print(f'{random_blueprint.id} at {object_location}')
    # Spawn the random object with the new location and rotation
    spawned_object = world.try_spawn_actor( random_blueprint, carla.Transform(object_location, random_rotation))
    return spawned_object

def spawn_objects(world, transform, simulation_config):
    try:
        vehicles_count = simulation_config['objects']['vehicle']
        pedestrians_count = simulation_config['objects']['pedestrian']
        cyclists_count = simulation_config['objects']['cyclist']
        motorcycle_count = simulation_config['objects']['motorcycle']
        other_object_count= simulation_config['objects']['other']
        specific_objects = simulation_config['objects']['specific']

        distribution_range = simulation_config['distribution_range']
        
        # Define a distance of 20 meters in front of the spectator
        distance_in_front = distribution_range/2  # You can adjust this distance as needed

        # Calculate the location for the random object
        forward_vector = transform.get_forward_vector()
        location_in_front = transform.location + distance_in_front * forward_vector

        for i in range(vehicles_count):
            spawned_object= try_to_spawn_object(world, location_in_front, distance_in_front,"vehicle")
            if spawned_object is not None:
                spawned_objects.append(spawned_object)
        for i in range(pedestrians_count):
            spawned_object= try_to_spawn_object(world, location_in_front, distance_in_front,"pedestrian")
            if spawned_object is not None:
                spawned_objects.append(spawned_object)
        for i in range(cyclists_count):
            spawned_object= try_to_spawn_object(world, location_in_front, distance_in_front,"bicycle")
            if spawned_object is not None:
                spawned_objects.append(spawned_object)
        for i in range(motorcycle_count):
            spawned_object= try_to_spawn_object(world, location_in_front, distance_in_front,"motorcycle")
            if spawned_object is not None:
                spawned_objects.append(spawned_object)
        for i in range(other_object_count):
            spawned_object= try_to_spawn_object(world, location_in_front, distance_in_front,"other")
            if spawned_object is not None:
                spawned_objects.append(spawned_object)
        for object_id, object_count in specific_objects.items():
            for i in range(object_count):
                spawned_object= try_to_spawn_object(world, location_in_front, distance_in_front,object_id)
                if spawned_object is not None:
                    spawned_objects.append(spawned_object)
            # Process each specific object with its ID and count
            #print(f"Object ID: {object_id}, Count: {object_count}")
        world.tick()

    except Exception as e:
        print(f"Error: {e}")
def sensor_callback(sensor_data, sensor_queue, sensor_name, actor):
    #print(f'call_back_of {sensor_name}')
    #print(global_vars['camera_ready'])
    if global_vars['camera_ready']:
        bbs={}
        transform =actor.get_transform()
        sensor_queue.put((sensor_data.frame, sensor_data, sensor_name,bbs,transform))  

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
def main(config_file, global_vars, sensor_queue):

    try:
        init_carla()
    except Exception as e:
        print("Impossible to initiate carla connection")
        raise 
    os.makedirs('images_folder/rgb', exist_ok=True)
    os.makedirs('images_folder/semantic_segmentation', exist_ok=True)
    os.makedirs('images_folder/instance_segmentation', exist_ok=True)
    os.makedirs('images_folder/simulation_objects',exist_ok=True)
    os.makedirs('images_folder/labels', exist_ok=True)
    map_id=None

    simulation_config= load_simulation_config(config_file)
    scenario_length=simulation_config['nb_images']
    max_distance = simulation_config['distribution_range']
    if 'map' in simulation_config:
        map_id=simulation_config['map']
    client = carla.Client('localhost', 2000)
    client.set_timeout(30.0)
    #client.get_world()
    if map_id is not None:
        try:
            client.load_world(map_id)
        except:
            pass
    world= client.get_world()
    default_settings = world.get_settings()#carla.WorldSettings(
        #fixed_delta_seconds=0.01)
    default_settings.synchronous_mode=False
    synchronous_settings=world.get_settings()
    synchronous_settings= carla.WorldSettings(
        fixed_delta_seconds=0.1,  # Adjust this value
        synchronous_mode=True,
        no_rendering_mode=False,
        max_substep_delta_time=0.05,  # Adjust this value
        max_substeps=10  # Adjust this value
    )
    blueprint_library = world.get_blueprint_library()
    map = world.get_map()

    # Choose a random spawn point on the map
    spawn_points = map.get_spawn_points()
    
    ego_vehicle =None
    sensor_list = []
    vego_bp =blueprint_library.find('vehicle.mercedes.sprinter')
    while ego_vehicle is None:
        spawn_point = random.choice(spawn_points)
        ego_vehicle = world.try_spawn_actor(vego_bp, spawn_point)
        
    for sensor_name, params in sensor_params.items():
        bp = blueprint_library.find(params['type'])
        bp.set_attribute('image_size_x', str(params['attributes']['image_size_x']))
        bp.set_attribute('image_size_y', str(params['attributes']['image_size_y']))
        bp.set_attribute('fov', str(params['attributes']['fov']))
        sensor_transform = carla.Transform(carla.Location(x=2.5, z=2.2))
        sensor = world.spawn_actor(bp, sensor_transform, attach_to=ego_vehicle)
        sensor_list.append((sensor_name, sensor))
        sensor.listen(lambda image, sensor_name=sensor_name: sensor_callback(image, sensor_queue, sensor_name,sensor))
    print(scenario_length)
    last_s_frame= None
    for i in range(scenario_length) :
        sensor_queue = Queue()
        spectator = world.get_spectator()
        spawn_point = random.choice(spawn_points)
        move_view(spectator,spawn_point)
        if ego_vehicle is not None:
            ego_vehicle.set_transform(spawn_point)
            data_ready=False
            time.sleep(1)
            
            
            spawn_objects(world,ego_vehicle.get_transform(),simulation_config)
            world.tick()
            time.sleep(1)
            global_vars['camera_ready']=True
            print("Camera is ready")
            print(global_vars['camera_ready'])
            #time.sleep(10)
            world.apply_settings(synchronous_settings)
            
            while not data_ready:

                w_frame = world.get_snapshot().frame
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
                
                   
                count_failure=0
                if image_rgb is None:
                    print("rgb")
                    count_failure += 1
                if image_semantic is None:
                    print("semantic_segmentation")
                    count_failure += 1
                if image_instance is None:
                    print("instance_segmentation")
                    count_failure += 1
                if objects is None:
                    print("bbs of simulation objects")
                    count_failure += 1
                if count_failure==0:
                    #np_img = np.frombuffer(image_rgb.raw_data, dtype=np.dtype("uint8"))
                    #count=500
                    #if last_s_frame is not None:
                    #    difference = cv2.subtract(np_img, last_s_frame)
                    #    #b, g, r = cv2.split(difference)
                    #    count= cv2.countNonZero(difference)
                    #if count>300:
                    #equal_mask = cv2.equal(image_rgb, last_s_frame)

                    # If all pixels are equal, the mask will be all True
                    #are_identical= equal_mask.all()
                    if True:
                        data_ready=True
                        global_vars['camera_ready'] = False
                        image_instance.save_to_disk(f'images_folder/instance_segmentation/image_{unix_timestamp}_{i}.png')
                        image_semantic.save_to_disk(f'images_folder/semantic_segmentation/image_{unix_timestamp}_{i}.png')
                        image_rgb.save_to_disk(f'images_folder/rgb/image_{unix_timestamp}_{i}.png')

                        array_sem = np.frombuffer(image_semantic.raw_data, dtype=np.dtype("uint8"))
                        array_sem = np.reshape(array_sem, (image_semantic.height, image_semantic.width, 4))

                        array_inst = np.frombuffer(image_instance.raw_data, dtype=np.dtype("uint8"))
                        array_inst = np.reshape(array_inst, (image_instance.height, image_instance.width, 4))
                        # Call the merge function
                        merged_image = merge_segmentation_images(array_inst, array_sem)

                        # Save the merged image to a file
                        #cv2.imwrite('merged_image_1000.png', merged_image)

                        #image = cv2.imread('merged_image_1000.png')
                        #image= image.astype(np.uint8)
                        bounding_boxes = find_color_regions_contours(merged_image)

                        #for box in bounding_boxes:
                        #    print("Bounding Box:", box)
                        input_image_path = f'images_folder/rgb/image_{unix_timestamp}_{i}.png'  # Replace with the actual image path
                        output_image_path = f'images_folder/labels/image_{unix_timestamp}_{i}.json' 

                        #draw_filtered_bounding_boxes(input_image_path, bounding_boxes, output_image_path,labels_of_interest)
                        filtered_objects = {}
                        for tag in objects:
                            filtered_objects[tag]=[]
                            for env_object,dist in objects[tag]:
                                #print(type(env_object))
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
                        with open(f'images_folder/simulation_objects/image_{unix_timestamp}_{i}.json', 'w') as json_file:
                            json.dump(info_dict, json_file, indent=4)
                        #final_bbs={}
                        #for o in bounding_boxes:
                        #    #print(o)
                        #    v={
                        #    'min_x': o['min_x'],
                        #    'max_x': o['max_x'],
                        #    'min_y': o['min_y'],
                        #    'max_y': o['max_y'],
                        #    #'color': tuple(color),
                        #    'label': o['label']
                        #    }
                        #    final_bbs[o['id']]=f"{v}"
                        #final_bbs = final_bbs.tolist()
                        #print(final_bbs)

                        #for key, value in final_bbs.items():
                        #    try:
                        #        json.dumps(value)
                        #    except TypeError:
                        #        print(f"Non-serializable value for key: {key}: {type(value)}")

                        with open(f'images_folder/labels/image_{unix_timestamp}_{i}.json', 'w') as json_file:
                            json.dump(bounding_boxes, json_file, indent=4)
                        last_s_frame = image_rgb

                world.tick()  


                
            
            for o in spawned_objects:
                if o.is_alive:
                    o.destroy()

            global_vars['camera_ready'] = False
            world.apply_settings(default_settings)
            time.sleep(2)

    ego_vehicle.destroy()

    settings = world.get_settings()
    settings.synchronous_mode = False
    world.apply_settings(settings)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creating a dataset starting from parametter config file")

    # Define arguments with default values
    parser.add_argument("arg1", nargs='?', default="simulation_config.json", help="config_file path")
    

    args = parser.parse_args()
    main(args.arg1, global_vars,sensor_queue)



