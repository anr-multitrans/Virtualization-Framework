import carla
import numpy as np
import json
import time
from queue import Queue, Empty
import os
import cv2
import random
from PyQt5.QtGui import QDesktopServices, QIcon
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
import threading
from tools import  move_view, bb_labels, is_visible, build_projection_matrix,copyImageData,get_image_point,CLASS_MAPPING

sensor_queue = Queue()
selected_labels=['Bicycle','Bus','Car','Motorcycle','Rider','Pedestrians','traffic_light','traffic_sign','Truck']

# Create a threading Lock to synchronize access to the current_tick variable
#current_tick_lock = threading.Lock()

def update_progress_bar(current_tick,progress_bar, max_ticks):
    

    if current_tick < max_ticks:
        #time.sleep(tick_interval)  # Sleep for a short time (adjust as needed)
        # Use a lock to safely update the current_tick variable
        progress = (current_tick / max_ticks) * 100
        progress_bar.setValue(progress)

class ActorInfo:
    def __init__(self, actor):
        actor_tags=' '.join(actor.semantic_tags)
        self.transform = actor.get_transform()
        self.bounding_box = actor.bounding_box
        self.id = actor.id
        self.name = actor.type_id+actor_tags  # You can change this to actor-specific name if available
        self.type = actor.type_id 

def image_to_qimage(sensor_data):
    """
    Convert CARLA sensor data to a QImage.

    Args:
        sensor_data: CARLA sensor data.

    Returns:
        QImage: Converted QImage.
    """
    if not sensor_data:
        return QImage()  # Return an empty QImage if sensor_data is None

    # Get the sensor data as a numpy array (assuming sensor_data is a numpy array)
    image_data = sensor_data.to_array()

    # Get the width and height of the image
    width = image_data.shape[1]
    height = image_data.shape[0]

    # Convert the image data to a QImage
    qimage = QImage(
        image_data.data,
        width,
        height,
        image_data.strides[0],  # bytes per line
        QImage.Format_RGB888  # You may need to adjust the format depending on your sensor data format
    )

    return qimage
# Create a class for synchronous mode
class CarlaSyncMode:
    def __init__(self, world, *sensors, sensor_queue, fps=20, rgb_label=None, semantic_label=None, instance_label=None, scenario_length=100000, BB_3D_Mode=True, selected_labels = selected_labels, simulation =None):
        self.world = world
        self.sensors = sensors
        self.fps = fps
        self.delta_seconds = 1.0 / fps
        self.sensor_queue = sensor_queue
        self.rgb_label= rgb_label
        self.semantic_label=semantic_label
        self.instance_label=instance_label
        self.scenario_length=scenario_length
        self.BB_3D_Mode=BB_3D_Mode
        self.selected_labels=selected_labels
        self.simulation=simulation

    def __enter__(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0/self.fps
        self.world.apply_settings(settings)
        self.frame = None

        for sensor_name, sensor in self.sensors:
            sensor.listen(lambda image, sensor_name=sensor_name: sensor_callback(image, self.sensor_queue, sensor_name, self.rgb_label, self.semantic_label, self.instance_label,self.BB_3D_Mode,self.world,sensor,self.selected_labels,self.simulation))
            #world.tick()

        return self

    def __exit__(self, *args, **kwargs):
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)

        for _,sensor in self.sensors:
            sensor.destroy()

    def tick(self, timeout):
        self.frame = self.world.tick()

def sensor_callback(sensor_data, sensor_queue, sensor_name, rgb_label=None, semantic_label=None, instance_label=None, BB_3D_Mode=True, world = None, actor = None, selected_labels=[], window=None):
    #print(sensor_name)
    if window.isRunning():
        bbs=[]
        if sensor_data.frame%20==0:
            
            all_actors = world.get_actors()
            #actors_with_specific_label = [actor for actor in all_actors if actor.attributes.get('semantic_description') == target_label]

            transform=actor.get_transform()
            K= build_projection_matrix(1200, 800, 90)
            bbs = { label: [(bb, bb.location.distance(transform.location)) for bb in world.get_level_bbs(bb_labels[label]) if
                        is_visible(bb, transform, K)] 
                        for label in window.get_selected_labels()
                    }
            for bb_actor in all_actors:
                # Check if any of the target tags are in the actor's semantic tags
                for target_tag in window.get_selected_labels():
                    if target_tag in actor.semantic_tags:
                        # Get the actor's transform and bounding box extent
                        
                        bounding_box = bb_actor.bounding_box
                        if is_visible(bounding_box, transform, K):
                            # Calculate the distance from the actor to the bounding box
                            distance_to_bb = transform.location.distance(bounding_box.location)
                            # Append the bounding box and its location to the corresponding label in the dictionary
                            bbs[target_tag].append((bounding_box, distance_to_bb))

        if sensor_name == 'rgb':
            #bbs={label: [world.get_level_bbs(bb_labels[label])] for label in selected_labels}
            
            #print(len(bbs))
            update_camera_view(image=sensor_data, view=rgb_label, desired_width=1200, desired_height=800)
            if BB_3D_Mode and sensor_data.frame%20==0:
                update_camera_view3D(image=sensor_data, view=instance_label, desired_width=1200, desired_height=800,bbs_world=bbs, actor=actor)
            #if rgb_label is not None:
            #    rgb_pixmap = QPixmap.fromImage(image_to_qimage(sensor_data))  # Convert sensor_data to QImage
            #    rgb_label.emit(rgb_pixmap)

        if sensor_name == 'semantic_segmentation':
            update_camera_view(image=sensor_data, view=semantic_label, desired_width=1200, desired_height=800, city_scape_convert=True)
            #if semantic_label is not None:
            #    semantic_pixmap = QPixmap.fromImage(image_to_qimage(sensor_data))  # Convert sensor_data to QImage
            #    semantic_label.emit(semantic_pixmap)

        if sensor_name == 'instance_segmentation' and not BB_3D_Mode:
            update_camera_view(image=sensor_data, view=instance_label, desired_width=1200, desired_height=800)
            #if instance_label is not None:
            #    instance_pixmap = QPixmap.fromImage(image_to_qimage(sensor_data))  # Convert sensor_data to QImage
            #    instance_label.emit(instance_pixmap)
        if(sensor_data.frame%20 == 0):
            sensor_queue.put((sensor_data.frame, sensor_data, sensor_name,bbs))  
            #if BB_3D_Mode and len(bbs)>0 :
            #     sensor_queue.put((sensor_data.frame, bbs, 'bbs'))
            

def main():
    run_carla_simulation()

def run_carla_simulation(rgb_label=None, semantic_label=None, instance_label=None, register=True, progress_bar=None, image_width=600, image_height=400,max_tick=100000, threeD=True, window=None):
    print("test")
    print(threeD)
    client = carla.Client('localhost', 2000)
    client.set_timeout(30.0)
    client.load_world('Town11')
    world= client.get_world()
    spectator = world.get_spectator()
    print(spectator.get_transform())
    spectator_location =carla.Location(x=-666.873047, y=-480.319824, z=154.421280)
    spectator_rotation = carla.Rotation(pitch=-4.973757, yaw=-138.874847, roll=0.000212)
    spectator_transform = carla.Transform(spectator_location, spectator_rotation)
    spectator.set_transform(spectator_transform)
    
    vehicle_transform = random.choice(world.get_map().get_spawn_points())

    blueprint_library = world.get_blueprint_library()
    l= carla.Location(x=-666.873047, y=-480.319824, z=154.421280)
    r = carla.Rotation(pitch=-4.973757, yaw=-138.874847, roll=0.000212)
    #l= carla.Location(x=-110.291763, y=97.093193, z=3.002939)
    #r= carla.Rotation(pitch=-8.006648, yaw=66.636604, roll=0.000058)
    # Create the sensor queue
    #sensor_queue = Queue()
    keywords= ['barrel', 'bin', 'clothcontainer','container','glasscontainer','box','trashbag','colacan','garbage','platformgarbage','trashcan','bench','gardenlamp','pergola','plasticchair','plastictable','slide','swing', 'swingcouch', 'table', 'trampoline','barbeque','clothesline','doghouse','gnome','wateringcan','haybale','plantpot','plasticbag','shoppingbag','shoppingcart','shoppingtrolley','briefcase','guitarcase','travelcase','helmet','mobile','purse','barrier','cone','ironplank','warning','brokentile','dirtdebris','foodcart','kiosk_01','fountain','maptable','advertisement','streetsign','busstop','atm','mailbox','streetfountain','vendingmachine','calibrator']
    ego_bp = blueprint_library.find('vehicle.mercedes.sprinter')
    ego_transform = carla.Transform(l, r)
    ego_vehicle = world.spawn_actor(ego_bp, vehicle_transform)
    

            # ...
            
    
    spectator = world.get_spectator()
    spectator.set_transform(ego_transform)
    
    # Define sensor parameters
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
    for sensor_name, params in sensor_params.items():
        bp = blueprint_library.find(params['type'])
        bp.set_attribute('image_size_x', str(params['attributes']['image_size_x']))
        bp.set_attribute('image_size_y', str(params['attributes']['image_size_y']))
        bp.set_attribute('fov', str(params['attributes']['fov']))
        sensor_transform = carla.Transform(carla.Location(x=2.5, z=2.2))
        sensor = world.spawn_actor(bp, sensor_transform, attach_to=ego_vehicle)
        sensor_list.append((sensor_name, sensor))

    # ...
    camera = None
    if len(sensor_list) > 0 :
        _,camera= sensor_list[0]

    if window.is_corner_case:
        pass

    else:
        try:
            ego_vehicle.set_autopilot(True)
            
             # Set up CARLA client
            
            #world.unload_map_layer(carla.MapLayer.Buildings)
            #world.unload_map_layer(carla.MapLayer.Foliage)
            #world.unload_map_layer(carla.MapLayer.Props)
            #world.unload_map_layer(carla.MapLayer.Decals)

            blueprint_library = world.get_blueprint_library()          


            # Enable synchronous mode and capture synchronized sensor data
            #(self, world, *sensors, sensor_queue, fps=30)
            with CarlaSyncMode(world, *sensor_list, sensor_queue=sensor_queue, fps=1, rgb_label=rgb_label, semantic_label=semantic_label, 
                instance_label=instance_label, scenario_length=max_tick,selected_labels=window.get_selected_labels(),BB_3D_Mode=threeD,simulation=window) as sync_mode:

                os.makedirs('images/rgb', exist_ok=True)
                os.makedirs('images/semantic_segmentation', exist_ok=True)
                os.makedirs('images/instance_segmentation', exist_ok=True)
                os.makedirs('images/simulation_objects',exist_ok=True)
                os.makedirs('images/labels', exist_ok=True)


                num_objects = 0

                # List to store spawned objects
                spawned_objects = []

                # Define the range of possible locations for spawning
                spawn_min_x = -200
                spawn_max_x = 200
                spawn_min_y = -200
                spawn_max_y = 200
                spawn_z = 0.5

                # Spawn random objects
                for _ in range(num_objects):
                    # Choose a random static prop blueprint
                    static_prop_blueprints = [bp for bp in blueprint_library.filter('static.prop.*')]
                    selected_blueprint = random.choice(static_prop_blueprints)
                    
                    # Choose a random spawn location
                    spawn_x = random.uniform(spawn_min_x, spawn_max_x)
                    spawn_y = random.uniform(spawn_min_y, spawn_max_y)
                    spawn_location = carla.Location(x=spawn_x, y=spawn_y, z=spawn_z)
                    
                    # Spawn the object and add it to the list
                    spawned_object = world.spawn_actor(selected_blueprint, carla.Transform(spawn_location))
                    spawned_objects.append(spawned_object)
                s_frame = 0
                i=0
                while i in range(100000000) and window.isRunning():
                    move_view(world.get_spectator(),ego_vehicle.get_transform())
                    sync_mode.tick(timeout=12.0)
                    ego_vehicle.set_autopilot=True
                    if i % 50 == 0:
                        
                        image_rgb = None
                        image_semantic = None
                        image_instance = None
                        bbs=None
                        #time.sleep(1)
                        s_frame=""
                        try:
                            for _ in range(len(sensor_list)):
                                s_frame, s_data, s_name,bbs_w = sensor_queue.get(True, 1.0)
                                if s_frame%100==0:
                                    if s_name == 'rgb':
                                        image_rgb = s_data
                                       

                                        # Process rgb image
                                        if camera is not None:
                                            world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
                                            forward_vec = camera.get_transform().get_forward_vector()
                                            # Get images and filter objects
                                            data = []
                                            

                                            filtered_objects = {}
                                            camera_transform = camera.get_transform()
                                            forward_vector = camera_transform.get_forward_vector()
                                            camera_location = camera_transform.location
                                            max_distance = 100
                                            world.tick()
                                            environment_objects={}
                                            #instance_names = [member.name for member in carla.CityObjectLabel]
                                            instance_names = {
                                                                
                                                                
                                                                "Fences":carla.CityObjectLabel.Fences,
                                                                "Other":carla.CityObjectLabel.Other,
                                                                "Pedestrians":carla.CityObjectLabel.Pedestrians,
                                                                "Poles":carla.CityObjectLabel.Poles,
                                                                
                                                                "Static":carla.CityObjectLabel.Static,
                                                                "Dynamic":carla.CityObjectLabel.Dynamic
                                                                
                                                                
                                                    
                                                           }

                                                            # Loop over the names of instances
                                            for semantic_tag in instance_names:
                                                if semantic_tag not in environment_objects:
                                                    environment_objects[semantic_tag] = []

                                                # Get environment objects of the current semantic tag
                                                env_objects = list(world.get_environment_objects(object_type=instance_names[semantic_tag]))

                                                # Filter objects that are in front of the camera and within the visible range
                                                #max_distance = visible_range  # Use visible_range or a different value as needed

                                                filtered_env_objects = []

                                                for obj in env_objects:
                                                    object_location = obj.transform.location
                                                    camera_to_object = object_location - camera_location

                                                    # Calculate the dot product to check if the object is in front of the camera
                                                    dot_product = camera_to_object.dot(forward_vector)
                                             
                                                    # Calculate the distance between the camera and the object
                                                    distance = camera_location.distance(object_location)#.magnitude()
                                            
                                                    if dot_product > 0 and distance <= max_distance:
                                                        filtered_env_objects.append(obj)

                                                environment_objects[semantic_tag].extend(filtered_env_objects)
                                            #environment_objects = list(world.get_environment_objects())

                                            # Merge the two lists while converting actor attributes
                                            all_objects = []

                                            # Convert actors to match the structure of environment objects
                                            for actor in spawned_objects:
                                                    tag = actor.semantic_tags[0]
                                                    
                                                    # Calculate actor location
                                                    actor_location = actor.get_location()
                                                    
                                                    # Calculate the vector from the camera to the actor
                                                    camera_to_actor = actor_location - camera_location
                                                    
                                                    # Calculate the dot product to check if the actor is in front of the camera
                                                    dot_product = np.dot(forward_vector, camera_to_actor)
                                                    
                                                    # Calculate the distance between the camera and the actor
                                                    distance = distance = camera_location.distance(actor_location)
                                                    
                                                    if dot_product > 0 and distance <= max_distance:
                                                        # Actor is in front of the camera and at least 50 meters away
                                                        actor_info = ActorInfo(actor)
                                                        
                                                        # Append the actor info to the appropriate tag in the dictionary
                                                        if tag not in environment_objects:
                                                            environment_objects[tag] = []
                                                        environment_objects[tag].append(actor_info)
                                                #all_objects.append(actor_info)

                                            # Add environment objects directly to the merged list
                                            all_objects=environment_objects
                                            for tag in all_objects:
                                                filtered_objects[tag]=[]
                                                for env_object in all_objects[tag]:
                                                    obj_name = env_object.name.lower()  # Convert object name to lowercase
                                                    contains_keyword = any(keyword in obj_name for keyword in keywords)
                                                    
                                                    if contains_keyword:
                                                        obj_transform = env_object.transform
                                                        obj_location = obj_transform.location
                                                        ray = obj_location - camera.get_transform().location
                                                        obj_direction = obj_location - camera_location
                                                        distance = obj_location.distance(camera.get_transform().location) 

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
                                            with open(f'images/simulation_objects/image_{s_frame}.json', 'w') as json_file:
                                                json.dump(info_dict, json_file, indent=4)
                                        #update_camera_view(image=image_rgb,view=rgb_label ,desired_width=image_width, desired_height=image_height)
                                        if register and image_rgb is not None:
                                            image_rgb.save_to_disk(f'images/rgb/image_{s_frame}.png')
                                    elif s_name == 'semantic_segmentation':
                                        image_semantic = s_data
                                        # Process semantic segmentation image
                                        #update_camera_view(image=image_semantic,view=semantic_label ,desired_width=image_width, desired_height=image_height,city_scape_convert=True)
                                        if register and image_semantic is not None:
                                            image_semantic.save_to_disk(f'images/semantic_segmentation/image_{s_frame}.png')
                                    elif s_name == 'instance_segmentation':
                                        image_instance = s_data
                                        # Process instance segmentation image
                                        #update_camera_view(image=image_instance,view=instance_label ,desired_width=image_width, desired_height=image_height)
                                        if register and image_instance is not None:
                                            image_instance.save_to_disk(f'images/instance_segmentation/image_{s_frame}.png')
                                #    elif s_name == "bbs":
                                #        bbs=s_data

                                # ... (rest of your object filtering and JSON creation code)
                                #if image_rgb is not None and bbs is not None:
                                #    update_camera_view3D(image=image_rgb, view=instance_label, desired_width=1200, desired_height=800,bbs_world=bbs, actor=camera)

                        except Empty:
                            print("Some of the sensor information is missed")
                        if image_rgb is None:
                            print("rgb")
                        elif image_semantic is None:
                            print("semantic_segmentation")
                        elif image_instance is None:
                            print("instance_segmentation")
                        # ... (rest of your object filtering and JSON creation code)
                        
                    
                    time.sleep(1.0/10)
                    i+=1

                time.sleep(2)
                for sensor_name, sensor in sensor_list:
                    sensor.stop()  
                for obj in spawned_objects:
                    obj.destroy()
            # Clean up
            # ...

            # Destroy ego vehicle
            ego_vehicle.destroy()

        except KeyboardInterrupt:
            print(' - Exited by user.')

def run_carla_simulation_simple(rgb_label=None, semantic_label=None, instance_label=None, register=True, progress_bar=None, image_width=600, image_height=400,max_tick=100000, threeD=True, window=None):
    print("test")
    print(threeD)
    if True:
        try:
            
             # Set up CARLA client
            client = carla.Client('localhost', 2000)
            client.set_timeout(5.0)
            world = client.load_world('Town11')
            #world.unload_map_layer(carla.MapLayer.Buildings)
            #world.unload_map_layer(carla.MapLayer.Foliage)
            #world.unload_map_layer(carla.MapLayer.Props)
            #world.unload_map_layer(carla.MapLayer.Decals)

            blueprint_library = world.get_blueprint_library()

            # ...
            l= carla.Location(x=-110.291763, y=97.093193, z=3.002939)
            r= carla.Rotation(pitch=-8.006648, yaw=66.636604, roll=0.000058)


            # Create the sensor queue
            #sensor_queue = Queue()
            keywords= ['barrel', 'bin', 'clothcontainer','container','glasscontainer','box','trashbag','colacan','garbage','platformgarbage','trashcan','bench','gardenlamp','pergola','plasticchair','plastictable','slide','swing', 'swingcouch', 'table', 'trampoline','barbeque','clothesline','doghouse','gnome','wateringcan','haybale','plantpot','plasticbag','shoppingbag','shoppingcart','shoppingtrolley','briefcase','guitarcase','travelcase','helmet','mobile','purse','barrier','cone','ironplank','warning','brokentile','dirtdebris','foodcart','kiosk_01','fountain','maptable','advertisement','streetsign','busstop','atm','mailbox','streetfountain','vendingmachine','calibrator']
            ego_bp = blueprint_library.find('vehicle.mercedes.sprinter')
            ego_transform = carla.Transform(l, r)
            ego_vehicle = world.spawn_actor(ego_bp, ego_transform)
            ego_vehicle.set_autopilot(True)
            spectator = world.get_spectator()
            spectator.set_transform(ego_transform)
            
            # Define sensor parameters
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
            for sensor_name, params in sensor_params.items():
                bp = blueprint_library.find(params['type'])
                bp.set_attribute('image_size_x', str(params['attributes']['image_size_x']))
                bp.set_attribute('image_size_y', str(params['attributes']['image_size_y']))
                bp.set_attribute('fov', str(params['attributes']['fov']))
                sensor_transform = carla.Transform(carla.Location(x=2.5, z=2.2))
                sensor = world.spawn_actor(bp, sensor_transform, attach_to=ego_vehicle)
                sensor_list.append((sensor_name, sensor))

            # ...
            camera = None
            if len(sensor_list) > 0 :
                _,camera= sensor_list[0]


            # Enable synchronous mode and capture synchronized sensor data
            #(self, world, *sensors, sensor_queue, fps=30)
            with CarlaSyncMode(world, *sensor_list, sensor_queue=sensor_queue, fps=1, rgb_label=rgb_label, semantic_label=semantic_label, 
                instance_label=instance_label, scenario_length=max_tick,selected_labels=window.get_selected_labels(),BB_3D_Mode=threeD,simulation=window) as sync_mode:

                os.makedirs('images/rgb', exist_ok=True)
                os.makedirs('images/semantic_segmentation', exist_ok=True)
                os.makedirs('images/instance_segmentation', exist_ok=True)
                os.makedirs('images/simulation_objects',exist_ok=True)
                os.makedirs('images/labels', exist_ok=True)


                num_objects = 0

                # List to store spawned objects
                spawned_objects = []

                # Define the range of possible locations for spawning
                spawn_min_x = -200
                spawn_max_x = 200
                spawn_min_y = -200
                spawn_max_y = 200
                spawn_z = 0.5

                # Spawn random objects
                for _ in range(num_objects):
                    # Choose a random static prop blueprint
                    static_prop_blueprints = [bp for bp in blueprint_library.filter('static.prop.*')]
                    selected_blueprint = random.choice(static_prop_blueprints)
                    
                    # Choose a random spawn location
                    spawn_x = random.uniform(spawn_min_x, spawn_max_x)
                    spawn_y = random.uniform(spawn_min_y, spawn_max_y)
                    spawn_location = carla.Location(x=spawn_x, y=spawn_y, z=spawn_z)
                    
                    # Spawn the object and add it to the list
                    spawned_object = world.spawn_actor(selected_blueprint, carla.Transform(spawn_location))
                    spawned_objects.append(spawned_object)
                s_frame = 0
                i=0
                stop_since=0
                last_transform=ego_vehicle.get_transform()
                while i in range(100000000) and window.isRunning():
                    move_view(world.get_spectator(),ego_vehicle.get_transform())
                    sync_mode.tick(timeout=12.0)
                    ego_vehicle.set_autopilot=True
                    if i % 100 == 0:
                        
                        image_rgb = None
                        image_semantic = None
                        image_instance = None
                        bbs=None
                        #time.sleep(1)
                        s_frame=""
                        try:
                            for _ in range(len(sensor_list)):
                                s_frame, s_data, s_name,bbs_w = sensor_queue.get(True, 1.0)
                                if s_name == 'rgb':
                                    image_rgb = s_data
                                   

                                    # Process rgb image
                                    #update_camera_view(image=image_rgb,view=rgb_label ,desired_width=image_width, desired_height=image_height)
                                    if register and image_rgb is not None:
                                        image_rgb.save_to_disk(f'images/rgb/image_{s_frame+6957}.png')
                                elif s_name == 'semantic_segmentation':
                                    image_semantic = s_data
                                    # Process semantic segmentation image
                                    #update_camera_view(image=image_semantic,view=semantic_label ,desired_width=image_width, desired_height=image_height,city_scape_convert=True)
                                    if register and image_semantic is not None:
                                        image_semantic.save_to_disk(f'images/semantic_segmentation/image_{s_frame+6957}.png')
                                elif s_name == 'instance_segmentation':
                                    image_instance = s_data
                                    # Process instance segmentation image
                                    #update_camera_view(image=image_instance,view=instance_label ,desired_width=image_width, desired_height=image_height)
                                    if register and image_instance is not None:
                                        image_instance.save_to_disk(f'images/instance_segmentation/image_{s_frame+6957}.png')
                            #    elif s_name == "bbs":
                            #        bbs=s_data

                            # ... (rest of your object filtering and JSON creation code)
                            #if image_rgb is not None and bbs is not None:
                            #    update_camera_view3D(image=image_rgb, view=instance_label, desired_width=1200, desired_height=800,bbs_world=bbs, actor=camera)

                        except Empty:
                            print("Some of the sensor information is missed")
                        if image_rgb is None:
                            print("rgb")
                        elif image_semantic is None:
                            print("semantic_segmentation")
                        elif image_instance is None:
                            print("instance_segmentation")
                        # ... (rest of your object filtering and JSON creation code)
                        if camera is not None:
                            world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
                            forward_vec = camera.get_transform().get_forward_vector()
                            # Get images and filter objects
                            data = []
                            

                            filtered_objects = []
                            camera_transform = camera.get_transform()
                            forward_vector = camera_transform.get_forward_vector()
                            camera_location = camera_transform.location
                            max_distance = 100
                            world.tick()
                            if ego_vehicle.get_transform == last_transform:
                                stop_since += 1
                            else:
                                stop_since = 0
                            if stop_since>10:
                                stop_since = 0
                                ego_vehicle.apply_control(carla.VehicleControl(throttle=-1.0, steer=2.0))


                            environment_objects = list(world.get_environment_objects())

                            # Merge the two lists while converting actor attributes
                            all_objects = []

                            # Convert actors to match the structure of environment objects
                            for actor in spawned_objects:
                                actor_info =ActorInfo(actor)
                                all_objects.append(actor_info)

                            # Add environment objects directly to the merged list
                            all_objects.extend(environment_objects)
                            for env_object in all_objects:
                                obj_name = env_object.name.lower()  # Convert object name to lowercase
                                contains_keyword = any(keyword in obj_name for keyword in keywords)
                                
                                if contains_keyword:
                                    obj_transform = env_object.transform
                                    obj_location = obj_transform.location
                                    ray = obj_location - camera.get_transform().location
                                    obj_direction = obj_location - camera_location
                                    distance = obj_location.distance(camera.get_transform().location) 

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
                                        filtered_objects.append({
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
                            with open(f'images/simulation_objects/image_{s_frame+6957}.json', 'w') as json_file:
                                json.dump(info_dict, json_file, indent=4)
                    
                    time.sleep(0.1)
                    i+=1
                    last_transform=ego_vehicle.get_transform()

                time.sleep(10)
                for sensor_name, sensor in sensor_list:
                    sensor.stop()  
                for obj in spawned_objects:
                    obj.destroy()
            # Clean up
            # ...

            # Destroy ego vehicle
            ego_vehicle.destroy()

        except KeyboardInterrupt:
            print(' - Exited by user.')


def update_camera_view(image, view, desired_width=300, desired_height=200, city_scape_convert=False):
    ##print(view)
    if view is not None:
        if city_scape_convert:
            image.convert(carla.ColorConverter.CityScapesPalette)
        np_img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        np_img = np_img.reshape((image.height, image.width, 4))
        np_img = np_img[..., :3]  # Remove the alpha channel

        # Create a QImage and a QPainter
        q_image = QImage(bytes(np_img.data), image.width, image.height, QImage.Format_RGB888).rgbSwapped()
        
        # Scale the QImage to the desired dimensions
        scaled_q_image = q_image.scaled(desired_width, desired_height, Qt.AspectRatioMode.KeepAspectRatio)

        # Create a QPixmap from the scaled QImage
        pixmap = QPixmap.fromImage(scaled_q_image)

        # Set the QPixmap on the view
        view.setPixmap(pixmap)

    return image


def update_camera_view3D(image, view, desired_width=600, desired_height=400, city_scape_convert=False, bbs_world = None, actor=None):
    ##print(view)
    #print(bbs_world)
    if view is not None:
        transform = actor.get_transform()
        K=[]
        bbs=[]
        if bbs_world is not None and actor is not None:
            K= build_projection_matrix(image.width, image.height, 90)
            bbs = bbs_world
            #{label: [(bb, bb.location.distance(transform.location)) for bb in bbs_list if is_visible(bb, transform, K)]
            #        for label, bbs_list in bbs_world.items()
            #        if label in selected_labels}
            #bbs = { label: [(bb, bb.location.distance(transform.location)) for bb in bbs_world[label] if
            #        is_visible(bb, transform, K)] 
            #        for label in selected_labels
            #        }
        ##print(bbs)
        world_2_camera = np.array(actor.get_transform().get_inverse_matrix())
        edges = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5], [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]
        img = np.frombuffer(copyImageData(image), dtype=np.dtype(
            "uint8"))  # np.frombuffer(copy.deepcopy(image.raw_data), dtype=np.dtype("uint8"))
        
        # rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        img = np.reshape(img, (image.height, image.width, 4))
        #img = img[..., :3]  # Remove the alpha channel
        # carla.CityObjectLabel.Car
        bb_id = 0
        processed_boxes = {label: [] for label in bbs.keys()}
        for label, bb_set in bbs.items():
            #print(label)
            object_color = CLASS_MAPPING[label.lower()]
            object_color = (object_color[2], object_color[1], object_color[0])
            for bb,dist in bb_set:
                try:

                    #print(bb)
                    edge_array = []
                    corners = bb.get_world_vertices(carla.Transform())
                    distance = dist
                    corners = [get_image_point(corner, K, world_2_camera) for corner in corners]
                    
                    
                    verts = [v for v in bb.get_world_vertices(carla.Transform())]
                    
                    corners = np.array(corners, dtype=int)
                    x_min, y_min = np.min(corners, axis=0).astype(int)
                    x_max, y_max = np.max(corners, axis=0).astype(int)
                    #included = False
                    # Draw edges of the bounding box into the camera output
                    if x_min<x_max and y_min<y_max:
                        for edge in edges:
                            try:
                                p1 = get_image_point(verts[edge[0]], K, world_2_camera)
                                p2 = get_image_point(verts[edge[1]], K, world_2_camera)
                                # Draw the edges into the camera output
                                #print(p1)
                                #print(p2)
                                if p1[0]>=0 and p1[1]>=0 and p2[0]>=0 and p2[1]>=0:

                                    x1 = max(int(p1[0]),0)
                                    x1 = min(x1, image.width)
                                    x2 = max(int(p2[0]),0)
                                    x2 = min(x2, image.width)
                                    y1 = max(int(p1[1]),0)
                                    y1 = min(y1, image.height)
                                    y2 = max(int(p2[1]),0)
                                    y2 = min(y2, image.height)
                                    #print(f'[{x1},{y1}] to [{x2},{y2}]')
                                    if((x1 != x2) or (y1 != y2)) and not is_line_crossing_image(x1,y1,x2,y2,image.width,image.height): #x1>0 and x1 <image.width and x2>0 and x2<image.width and y1>0 and y1<image.height and y2>0 and y2<image.height:
                                        cv2.line(img, (x1, y1), (x2, y2), object_color, 2)
                                        edge_array.append((p1.tolist(), p2.tolist()))
                            except Exception as e:
                                print(e)
                                continue
                            # label = 'vehicle'  # replace with the appropriate label for each object type
                        cv2.putText(img, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, object_color, 2)
                        #json_entry = {"id": bb_id, "label": label, "edges": edge_array}
                        #json_array.append(json_entry)
                        bb_id += 1
                except Exception as e:
                    print(e)
                    continue









        
        
        img = img[..., :3]
        # Create a QImage and a QPainter
        q_image = QImage(bytes(img.data), image.width, image.height, QImage.Format_RGB888).rgbSwapped()
        
        # Scale the QImage to the desired dimensions
        scaled_q_image = q_image.scaled(desired_width, desired_height, Qt.AspectRatioMode.KeepAspectRatio)

        # Create a QPixmap from the scaled QImage
        pixmap = QPixmap.fromImage(scaled_q_image)

        # Set the QPixmap on the view
        view.setPixmap(pixmap)

    return image

def is_line_crossing_image(x1, y1, x2, y2, width, height):
    # Check if the line can cross the image
    return abs(x1 - x2) >= width or abs(y1 - y2) >= height


def update_camera_view_old(image, view, desired_width=600, desired_height=400,city_scape_convert=False):
    #print(view)
    if  view is not None:
        if city_scape_convert:
            image.convert(carla.ColorConverter.CityScapesPalette)
        np_img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        np_img = np_img.reshape((image.height, image.width, 4))
        np_img = np_img[..., :3]  # Remove the alpha channel
        
        q_image = QtGui.QImage(bytes(np_img.data), image.width, image.height, QtGui.QImage.Format_RGB888).rgbSwapped()
        
        scaled_q_image = q_image.scaled(desired_width, desired_height, Qt.AspectRatioMode.KeepAspectRatio)
        pixmap = QtGui.QPixmap.fromImage(scaled_q_image)
        view.setPixmap(pixmap)
        #view.emit(pixmap)
    return image
   


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')
