import carla
import numpy as np
import cv2
import json
import time

# Set up CARLA client
client = carla.Client('localhost', 2000)
client.set_timeout(5.0)

# Load the world and blueprint library
world = client.get_world()
blueprint_library = world.get_blueprint_library()
#l= carla.Location(x=-78.752174, y=136.815399, z=2.714144)
#r= carla.Rotation(pitch=1.375518, yaw=-14.460558, roll=0.000025)

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
sensors = []
sensor_types = [] 
for sensor_name, params in sensor_params.items():
    bp = blueprint_library.find(params['type'])
    bp.set_attribute('image_size_x', str(params['attributes']['image_size_x']))
    bp.set_attribute('image_size_y', str(params['attributes']['image_size_y']))
    bp.set_attribute('fov', str(params['attributes']['fov']))
    sensor_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
    sensor = world.spawn_actor(bp, sensor_transform, attach_to=ego_vehicle)
    sensors.append(sensor)
    sensor_types.append(sensor_name)
camera = None
if len(sensors) > 0 :
    camera= sensors[0]

if camera is not None:
    world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
    forward_vec = camera.get_transform().get_forward_vector()
    # Get images and filter objects
    data = []
    

    filtered_objects = []
    camera_transform = sensors[0].get_transform()
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
       
    
    for idx, sensor in enumerate(sensors):
        sensor.listen(lambda image, idx=idx: data.append({'sensor': sensor_types[idx], 'image': image}))    
        world.tick()
    # Save images
    for idx, entry in enumerate(data):
        sensor_type = entry["sensor"]
        image = entry['image']
        if sensor_type == 'semantic_segmentation':
            image.save_to_disk(f'image_{sensor_type}_{idx}.png', carla.ColorConverter.CityScapesPalette)
        else:
            image.save_to_disk(f'image_{sensor_type}_{idx}.png')

for i in range(5):
    world.tick()
    time.sleep(20)
# Clean up
for sensor in sensors:
    sensor.destroy()

# Destroy ego vehicle
ego_vehicle.destroy()