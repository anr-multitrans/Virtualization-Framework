import carla
import random
import time
import queue
import numpy as np
import cv2

# Connect to client and set CARLA server to synchronous mode
client = carla.Client('localhost', 2000)
client.set_timeout(30.0)
world = client.get_world()
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05  # Set fixed delta seconds
world.apply_settings(settings)


# Get the map spawn points and the spectator
spawn_points = world.get_map().get_spawn_points()
spectator = world.get_spectator()

# Set the camera to some location in the map
cam_location = carla.Location(x=-46., y=152, z=18)
cam_rotation = carla.Rotation(pitch=-21, yaw=-93.4, roll=0)
camera_transform = carla.Transform(location=cam_location, rotation=cam_rotation)
spectator.set_transform(camera_transform)

# Retrieve the semantic camera blueprint and spawn the camera
instance_camera_bp = world.get_blueprint_library().find('sensor.camera.instance_segmentation')
instance_camera = world.try_spawn_actor(instance_camera_bp, camera_transform)

# Retrieve the RGB camera blueprint and spawn the camera
rgb_camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
rgb_camera = world.try_spawn_actor(rgb_camera_bp, camera_transform)

semseg_bb = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
sem_camera = world.try_spawn_actor(semseg_bb, camera_transform)

# Create queues to store the images
instance_image_queue = queue.Queue()
rgb_image_queue = queue.Queue()
seg_image_queue = queue.Queue()

# Set up callback functions to receive the images
def instance_image_callback(image):
    instance_image_queue.put(image)

def rgb_image_callback(image):
    rgb_image_queue.put(image)

def seg_image_callback(image):
    image.convert(carla.ColorConverter.CityScapesPalette)
    seg_image_queue.put(image)


# Register the callback functions
instance_camera.listen(instance_image_callback)
rgb_camera.listen(rgb_image_callback)
sem_camera.listen(seg_image_callback)

# Spawn vehicles in an 80m vicinity of the camera
vehicle_bp_library = world.get_blueprint_library().filter('*vehicle*')
radius = 80
spawned_actors = []
for spawn_point in spawn_points:
    vec = [spawn_point.location.x - cam_location.x, spawn_point.location.y - cam_location.y]
    if vec[0]*vec[0] + vec[1]*vec[1] < radius*radius:
        actor = world.try_spawn_actor(random.choice(vehicle_bp_library), spawn_point)
        if actor is not None:
            spawned_actors.append(actor)
world.tick()


# Save the image to disk
num_ticks = 2  # Number of ticks to execute

for t in range(num_ticks):
    # Tick the simulation
    world.tick()

    # Get the instance segmentation image
    instance_image = instance_image_queue.get()
    sem_image = seg_image_queue.get()

    # Get the RGB image
    rgb_image = rgb_image_queue.get()
    alpha = 0.5  # Adjust the alpha value as needed (0.0 to 1.0)
    # Perform alpha blendi


    
    # Process the instance segmentation image
    instance_array = np.array(instance_image.raw_data)
    instance_array = instance_array.reshape((instance_image.height, instance_image.width, 4))
    instance_array = instance_array[:, :, :3]  # Remove alpha channel

    rgb_array =  np.array(rgb_image.raw_data)
    rgb_array = rgb_array.reshape((rgb_image.height, rgb_image.width, 4))
    rgb_array = rgb_array[:, :, :3]  # Remove alpha channel

    rgb_image_draw = np.array(rgb_image.raw_data)
    rgb_image_draw = rgb_image_draw.reshape((rgb_image.height, rgb_image.width, 4))
    rgb_image_draw = cv2.cvtColor(rgb_image_draw, cv2.COLOR_BGRA2BGR)
    sem_array = np.array(sem_image.raw_data)
    sem_array = sem_array.reshape((sem_image.height, sem_image.width, 4))
    sem_array = sem_array[:, :, :3]  # Remove alpha channel
    blended_image = cv2.addWeighted(instance_array, alpha, sem_array, 1 - alpha, 0)
    cv2.imwrite('rgb.png',rgb_array)
    cv2.imwrite('semantic_segmentation.png',sem_array)
    cv2.imwrite('instance_segmentation.png', instance_array)
    cv2.imwrite('blended_image.png', blended_image)


    colors = set()
    for row in blended_image:
        for pixel in row:
            color = tuple(pixel)
            colors.add(color)

    for color in colors:
        #print(color)
        mask = np.all(blended_image == color, axis=-1)
        tag_id_label = f"{color[0]}_{color[1]}_{color[2]}"

        nonzero_pixels = np.argwhere(mask != 0)
        if len(nonzero_pixels) > 0:
            x, y = nonzero_pixels[0][1], nonzero_pixels[0][0]
            min_x, max_x = np.min(nonzero_pixels[:, 1]), np.max(nonzero_pixels[:, 1])
            min_y, max_y = np.min(nonzero_pixels[:, 0]), np.max(nonzero_pixels[:, 0])
            sem_color = tuple(sem_array[y, x])
            colorx = tuple(int(round(c)) for c in sem_color)
            cv2.rectangle(rgb_image_draw, (min_x, min_y), (max_x, max_y), colorx, 2)

    result_image_path = f'result_{t}.png'
    cv2.imwrite(result_image_path, rgb_image_draw)

    time.sleep(0.1)  # Wait for a short period before the next tick


# Set the simulator back to asynchronous mode
settings.synchronous_mode = False
settings.fixed_delta_seconds = None
world.apply_settings(settings)

# Destroy all actors
for actor in spawned_actors:
    actor.destroy()
instance_camera.destroy()
rgb_camera.destroy()
