import carla
import random
import time
import os
import json
from carla import ColorConverter

# Define the dimensions of the camera image
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# Connect to the CARLA simulator
client = carla.Client('localhost', 2000)
client.set_timeout(10)

# Get the world and map
world = client.get_world()
map = world.get_map()

# Choose a random spawn point on the map
spawn_points = map.get_spawn_points()
spawn_point = random.choice(spawn_points)

# Spawn a vehicle at the spawn point
blueprint_library = world.get_blueprint_library()
vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# Define the mapping from RGB values to class labels
CLASS_MAPPING = {
    (0, 0, 0): 'unlabeled',
    (70, 70, 70): 'building',
    (100, 40, 40): 'fence',
    (55, 90, 80): 'other',
    (220, 20, 60): 'pedestrian',
    (153, 153, 153): 'pole',
    (157, 234, 50): 'road_line',
    (128, 64, 128): 'road',
    (244, 35, 232): 'sidewalk',
    (107, 142, 35): 'vegetation',
    (0, 0, 142): 'vehicle',
    (102, 102, 156): 'wall',
    (220, 220, 0): 'traffic_sign',
    (70, 130, 180): 'sky',
    (81, 0, 81): 'ground',
    (150, 100, 100): 'bridge',
    (230, 150, 140): 'railtrack',
    (180, 165, 180): 'guardrail',
    (250, 170, 30): 'traffic_light',
    (110, 190, 160): 'static',
    (170, 120,  50): 'dynamic',
    (45,  60, 150): 'water',
    (145, 170, 100): 'terrain'
}

CLASS_MAPPING_2 = {
    (0, 0, 0): 'u',
    (70, 70, 70): 'b',
    (100, 40, 40): 'f',
    (55, 90, 80): 'o',
    (220, 20, 60): 'h',
    (153, 153, 153): 'p',
    (157, 234, 50): 'l',
    (128, 64, 128): 'r',
    (244, 35, 232): 's',
    (107, 142, 35): 'v',
    (0, 0, 142): 'c',
    (102, 102, 156): 'w',
    (220, 220, 0): 't',
    (70, 130, 180): 'S',
    (81, 0, 81): 'G',
    (150, 100, 100): 'B',
    (230, 150, 140): 't',
    (180, 165, 180): '_',
    (250, 170, 30): 'T',
    (110, 190, 160): '.',
    (170, 120,  50): '*',
    (45,  60, 150): '~',
    (145, 170, 100): '^'
}

# Add a semantic segmentation camera sensor to the vehicle
semseg_camera_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
semseg_camera_bp.set_attribute('fov', '110')
semseg_camera_bp.set_attribute('image_size_x', f'{IMAGE_WIDTH}')
semseg_camera_bp.set_attribute('image_size_y', f'{IMAGE_HEIGHT}')
semseg_camera_bp.set_attribute('sensor_tick', '0.05')
#camera_bp.set_attribute('post_processing', 'SemanticSegmentation')
camera_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
semseg_camera = world.spawn_actor(semseg_camera_bp, camera_transform, attach_to=vehicle)

class_mapping = {str(k): v for k, v in CLASS_MAPPING.items()}

with open('class_mapping.json', 'w') as outfile:
    json.dump(class_mapping, outfile)

# Create a folder to save the captured images
image_folder = '_semseg_captured_images'

if not os.path.exists(image_folder):
    os.mkdir(image_folder)

# Define a callback function to process the semantic segmentation image data
def process_semseg_image(image):
    # Save the image to a file using CityScapesPalette color converter for semantic segmentation
    process_semseg_image.counter += 1
    image.convert(carla.ColorConverter.CityScapesPalette)
    image.save_to_disk(os.path.join(image_folder, 'sem_seg_{:06d}.png'.format(process_semseg_image.counter)))
    img2=image
    # Extract the pixel data from the image
    pixel_data = image.raw_data
    data=''
    # Map the RGB values of each pixel to class labels
    classes = []
    for i in range(0, len(pixel_data), 4):
        b, g, r, a = pixel_data[i:i+4]
        class_label = CLASS_MAPPING.get((r, g, b), 'unknown')
        classes.append(class_label)
        data= data+ CLASS_MAPPING_2.get((r, g, b), 'x') + ' '
        if i%(4*640)==0:
            data=data+'\n'
        if i<=221312 & i>221308:
            print(r)
            print(g)
            print(b)
            print(a)
    with open(os.path.join(image_folder, 'sem_seg_{:06d}.txt'.format(process_semseg_image.counter)), 'w') as f:
        f.write(data)

    
    # Save the class information to a JSON file
    image_filename = 'sem_seg_{:06d}.png'.format(process_semseg_image.counter)
    json_filename = os.path.join(image_folder, 'sem_seg_{:06d}.json'.format(process_semseg_image.counter))
    with open(json_filename, 'w') as f:
        json.dump({'image': image_filename, 'classes': classes}, f)
    

process_semseg_image.counter = 0

#process_image.counter = 0

# Attach the callback functions to the camera sensors
semseg_camera.listen(process_semseg_image)

# Capture a sequence of images
for i in range(50):
    # Wait for a short time
    vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0))
    time.sleep(0.65)

# Destroy the camera and vehicle
semseg_camera.destroy()
vehicle.destroy()