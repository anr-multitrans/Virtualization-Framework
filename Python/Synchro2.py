import carla
import numpy as np
import json
import time
from queue import Queue, Empty
import os
import random

sensor_queue = Queue()
# Create a class for synchronous mode
class CarlaSyncMode:
    def __init__(self, world, *sensors, sensor_queue, fps=30):
        self.world = world
        self.sensors = sensors
        self.fps = fps
        self.delta_seconds = 1.0 / fps
        self.sensor_queue = sensor_queue
        

    def __enter__(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1
        self.world.apply_settings(settings)
        self.frame = None

        for sensor_name, sensor in self.sensors:
            sensor.listen(lambda image, sensor_name=sensor_name: sensor_callback(image, self.sensor_queue, sensor_name))
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

def sensor_callback(sensor_data, sensor_queue, sensor_name):
    if(sensor_data.frame%20 == 0):
        sensor_queue.put((sensor_data.frame, sensor_data, sensor_name))

def main():
    # Set up CARLA client
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    world = client.get_world()
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
    with CarlaSyncMode(world, *sensor_list, sensor_queue=sensor_queue, fps=1) as sync_mode:

        os.makedirs('images/rgb', exist_ok=True)
        os.makedirs('images/semantic_segmentation', exist_ok=True)
        os.makedirs('images/instance_segmentation', exist_ok=True)
        os.makedirs('images/simulation_objects',exist_ok=True)
        os.makedirs('labels', exist_ok=True)


        num_objects = 100

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
        for i in range(3500):
            sync_mode.tick(timeout=12.0)
            ego_vehicle.set_autopilot=True
            if i % 20 == 0:
                image_rgb = None
                image_semantic = None
                image_instance = None
                #time.sleep(1)
                s_frame=""
                try:
                    for _ in range(len(sensor_list)):
                        s_frame, s_data, s_name = sensor_queue.get(True, 1.0)
                        if s_name == 'rgb':
                            image_rgb = s_data
                            # Process rgb image
                            image_rgb.save_to_disk(f'images/rgb/image_{s_frame}.png')
                        elif s_name == 'semantic_segmentation':
                            image_semantic = s_data
                            # Process semantic segmentation image
                            image_semantic.save_to_disk(f'images/semantic_segmentation/image_{s_frame}.png', carla.ColorConverter.CityScapesPalette)
                        elif s_name == 'instance_segmentation':
                            image_instance = s_data
                            # Process instance segmentation image
                            image_instance.save_to_disk(f'images/instance_segmentation/image_{s_frame}.png')

                    # ... (rest of your object filtering and JSON creation code)

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
                    for env_object in world.get_environment_objects():
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
                    with open(f'images/simulation_objects/image_{s_frame}.json', 'w') as json_file:
                        json.dump(info_dict, json_file, indent=4)
            
        time.sleep(10)
        for obj in spawned_objects:
            obj.destroy()
    # Clean up
    # ...

    # Destroy ego vehicle
    ego_vehicle.destroy()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')
