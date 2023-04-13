import carla
import json
import os
import random
import time
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
spawn_point = spawn_points[0]

# Spawn a vehicle at the spawn point
blueprint_library = world.get_blueprint_library()
vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# Add a semantic segmentation camera sensor to the vehicle
semseg_camera_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
semseg_camera_bp.set_attribute('fov', '110')
semseg_camera_bp.set_attribute('image_size_x', f'{IMAGE_WIDTH}')
semseg_camera_bp.set_attribute('image_size_y', f'{IMAGE_HEIGHT}')
semseg_camera_bp.set_attribute('sensor_tick', '0.05')

# Get the label definitions and create a dictionary
label_definition = semseg_camera_bp.get_attribute('labels_definition')
label_dict = {}
for label in label_definition.split('\n'):
    if label:
        parts = label.strip().split('.')
        label_id = int(parts[0])
        label_name = parts[2]
        label_dict[label_id] = label_name

# Create a folder to save the captured images
image_folder = 'captured_images'

if not os.path.exists(image_folder):
    os.mkdir(image_folder)

# Define a callback function to process the semantic segmentation image data and create a JSON file
def process_semseg_image(image):
    # Save the image to a file
    process_semseg_image.counter += 1
    filename = os.path.join(image_folder, 'sem_seg_{:06d}.png'.format(process_semseg_image.counter))
    image.save_to_disk(filename, carla.ColorConverter.CityScapesPalette)

    # Extract the semantic segmentation data
    semseg_data = image.raw_data
    semseg_image = np.array(semseg_data, dtype=np.uint8).reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 4))
    semseg_image = semseg_image[:, :, :3] # Ignore the alpha channel

    # Convert the semantic segmentation data to grayscale mask
    semseg_mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    for i in range(IMAGE_HEIGHT):
        for j in range(IMAGE_WIDTH):
            label_id = semseg_image[i, j, 0]
            semseg_mask[i, j] = label_id

    # Create a JSON file with the label definitions and the filename of the image
    json_dict = {
        'filename': filename,
        'labels': label_dict
    }
    json_filename = os.path.join(image_folder, 'sem_seg_{:06d}.json'.format(process_semseg_image.counter))
    with open(json_filename, 'w') as f:
        json.dump(json_dict, f)

process_semseg_image.counter = 0

# Attach the callback function to the camera sensor
semseg_camera = world.spawn_actor(semseg_camera_bp, carla.Transform(), attach_to=vehicle)
semseg_camera.listen(process_semseg_image)

# Capture a sequence of images
for i in range(50):
    # Wait for a short time
    world.tick()
    vehicle
