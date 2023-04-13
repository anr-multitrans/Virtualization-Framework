import carla
import random
import time
import os

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

# Add a camera sensor to the vehicle
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('fov', '110')
camera_bp.set_attribute('image_size_x', f'{IMAGE_WIDTH}')
camera_bp.set_attribute('image_size_y', f'{IMAGE_HEIGHT}')
camera_bp.set_attribute('sensor_tick', '0.1')
camera_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

# Add a semantic segmentation camera sensor to the vehicle
semseg_camera_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
semseg_camera_bp.set_attribute('fov', '110')
semseg_camera_bp.set_attribute('image_size_x', f'{IMAGE_WIDTH}')
semseg_camera_bp.set_attribute('image_size_y', f'{IMAGE_HEIGHT}')
semseg_camera_bp.set_attribute('sensor_tick', '0.05')
#camera_bp.set_attribute('post_processing', 'SemanticSegmentation')
#camera_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
semseg_camera = world.spawn_actor(semseg_camera_bp, camera_transform, attach_to=vehicle)


# Create a folder to save the captured images
image_folder = 'captured_images'

if not os.path.exists(image_folder):
    os.mkdir(image_folder)

# Define a callback function to process the semantic segmentation image data
def process_semseg_image(image):
    # Save the image to a file using CityScapesPalette color converter for semantic segmentation
    process_semseg_image.counter += 1
    image.save_to_disk(os.path.join(image_folder, 'sem_seg_{:06d}.png'.format(process_semseg_image.counter)), carla.ColorConverter.CityScapesPalette)
    

def process_image(image):
    # Save the image to a file
    image.save_to_disk(os.path.join(image_folder, 'image_{:06d}.png'.format(process_semseg_image.counter)))
    process_image.counter += 1


process_semseg_image.counter = 0

process_image.counter = 0

# Attach the callback functions to the camera sensors
camera.listen(process_image)
semseg_camera.listen(process_semseg_image)

# Capture a sequence of images
for i in range(50):
    # Wait for a short time
    vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0))
    time.sleep(0.65)

# Destroy the camera and vehicle
semseg_camera.destroy()
vehicle.destroy()