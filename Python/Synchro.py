import carla
import numpy as np
import cv2
import json
import time

# Create a class for synchronous mode
class CarlaSyncMode:
    def __init__(self, world, *sensors, fps=30):
        self.world = world
        self.sensors = sensors
        self.fps = fps
        self.delta_seconds = 1.0 / fps

    def __enter__(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.delta_seconds
        self.world.apply_settings(settings)
        self.frame = None

        for sensor_name, sensor in sensor_list:
            sensor.listen(lambda image, sensor_type=sensor_name: data.append({'sensor_type': sensor_type, 'image': image}))
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

    def _sensor_callback(self, data):
        self.frame = data

# Set up CARLA client
client = carla.Client('localhost', 2000)
client.set_timeout(5.0)

# Load the world and blueprint library
world = client.get_world()
blueprint_library = world.get_blueprint_library()

l= carla.Location(x=-110.291763, y=97.093193, z=3.002939)
r= carla.Rotation(pitch=-8.006648, yaw=-114.636604, roll=0.000058)

# Spawn ego vehicle
keywords= ['barrel', 'bin', 'clothcontainer','container','glasscontainer','box','trashbag',
'colacan','garbage','platformgarbage','trashcan','bench','gardenlamp','pergola','plasticchair',
'plastictable','slide','swing', 'swingcouch', 'table', 'trampoline','barbeque','clothesline',
'doghouse','gnome','wateringcan','haybale','plantpot','plasticbag','shoppingbag','shoppingcart',
'shoppingtrolley','briefcase','guitarcase','travelcase','helmet','mobile','purse','barrier',
'cone','ironplank','warning','brokentile','dirtdebris','foodcart','kiosk_01','fountain','maptable',
'advertisement','streetsign','busstop','atm','mailbox','streetfountain','vendingmachine','calibrator']
ego_bp = blueprint_library.find('vehicle.tesla.model3')
ego_transform = carla.Transform(l, r)
ego_vehicle = world.spawn_actor(ego_bp, ego_transform)

# Define sensor parameters
sensor_params = {
    'rgb': {
        'type': 'sensor.camera.rgb',
        'attributes': {
            'image_size_x': 800,
            'image_size_y': 600,
            'fov': 90,
            'sensor_tick': 0.1
        }
    },
    'semantic_segmentation': {
        'type': 'sensor.camera.semantic_segmentation',
        'attributes': {
            'image_size_x': 800,
            'image_size_y': 600,
            'fov': 90,
            'sensor_tick': 0.1
        }
    },
    'instance_segmentation': {
        'type': 'sensor.camera.instance_segmentation',
        'attributes': {
            'image_size_x': 800,
            'image_size_y': 600,
            'fov': 90,
            'sensor_tick': 0.1
        }
    }
}

# Spawn sensors on ego vehicle
sensor_list = []
for sensor_name, params in sensor_params.items():
    bp = blueprint_library.find(params['type'])
    bp.set_attribute('image_size_x', str(params['attributes']['image_size_x']))
    bp.set_attribute('image_size_y', str(params['attributes']['image_size_y']))
    bp.set_attribute('fov', str(params['attributes']['fov']))
    sensor_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
    sensor = world.spawn_actor(bp, sensor_transform, attach_to=ego_vehicle)
    sensor_list.append((sensor_name, sensor))
    
camera = None
if len(sensor_list) > 0 :
    _,camera= sensor_list[0]

# ...



# Enable synchronous mode and capture synchronized sensor data
with CarlaSyncMode(world, *sensor_list, fps=1) as sync_mode:
    for i in range(15):
        sync_mode.tick(timeout=2.0)
        
        if camera is not None:
            world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
            forward_vec = camera.get_transform().get_forward_vector()
            # Get images and filter objects
            data = []
            

            filtered_objects = []
            camera_transform = camera.get_transform()
            forward_vector = camera_transform.get_forward_vector()
            camera_location = camera_transform.location
            max_distance = 50
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
            with open('sensor_and_objects_info.json', 'w') as json_file:
                json.dump(info_dict, json_file, indent=4)
               

        # Process sensor data
        if sync_mode.frame:
            image_rgb = None
            image_semantic = None
            image_instance = None
            
            for sensor_type ,sensor in sync_mode.frame:
                if sensor_type == 'sensor.camera.rgb':
                    image_rgb = sensor.get_data()
                elif sensor_type == 'sensor.camera.semantic_segmentation':
                    image_semantic = sensor.get_data()
                elif sensor_type == 'sensor.camera.instance_segmentation':
                    image_instance = sensor.get_data()

            if image_rgb is not None:
                # Process rgb image
                image.save_to_disk(f'image_rgb_{sync_mode.frame}.png')
                pass
            
            if image_semantic is not None:
                # Process semantic segmentation image
                image.save_to_disk(f'image_semantic_{sync_mode.frame}.png', carla.ColorConverter.CityScapesPalette)
                pass
            
            if image_instance is not None:
                # Process instance segmentation image
                image.save_to_disk(f'image_instance_{sync_mode.frame}.png')
                pass

            time.sleep(2)

            # Your object filtering and JSON creation code here

# Clean up
# ...

# Destroy ego vehicle
ego_vehicle.destroy()
