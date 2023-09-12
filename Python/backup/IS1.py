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
v_loc= random.choice(spawn_points)
cam_location = carla.Location(x=-46., y=152, z=18)
cam_rotation = carla.Rotation(pitch=-21, yaw=-93.4, roll=0)
camera_transform = carla.Transform(carla.Location(x=2, z=1.5, y=0)) #carla.Transform(location=cam_location, rotation=cam_rotation)
spectator.set_transform(v_loc)

vehicle_bb=random.choice(world.get_blueprint_library().filter('vehicle*'))
vehicle_insance = world.try_spawn_actor(vehicle_bb, v_loc)
# Retrieve the semantic camera blueprint and spawn the camera
instance_camera_bp = world.get_blueprint_library().find('sensor.camera.instance_segmentation')
instance_camera = world.try_spawn_actor(instance_camera_bp, camera_transform, attach_to=vehicle_insance)

# Retrieve the RGB camera blueprint and spawn the camera
rgb_camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
rgb_camera = world.try_spawn_actor(rgb_camera_bp, camera_transform, attach_to=vehicle_insance)

semseg_bb = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
sem_camera = world.try_spawn_actor(semseg_bb, camera_transform, attach_to=vehicle_insance)

vehicle_insance.set_autopilot(True)
# Create queues to store the images
instance_image_queue = queue.Queue()
rgb_image_queue = queue.Queue()
seg_image_queue = queue.Queue()
object_names = world.get_names_of_all_objects()
for name in object_names:
    print(name)


def is_object_fragmented(instance_image, class_color, min_x, min_y, max_x, max_y):

    # Crop the instance image within the bounding box
    cropped_image = instance_image[min_y:max_y, min_x:max_x]

    nb_total= (max_y-min_y)*(max_x-min_x)

    # Compute the mask of class color pixels within the cropped image
    mask = np.all(cropped_image == class_color, axis=-1)

    # Find the contours of the class color pixels
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if there are multiple contours (areas) for the object
    if len(contours) > 5:

        # Count the number of class color pixels
        class_color_pixels = np.count_nonzero(mask)

        # Count the total number of pixels within the bounding box
        total_pixels = cropped_image.shape[0] * cropped_image.shape[1]

        # Compute the ratio of class color pixels to total pixels
        ratio = class_color_pixels / total_pixels

        # Check if the object is fragmented based on the ratio threshold (70%)
        if ratio < 0.5:
            return True

    # by default, return False
    return False
# Set up callback functions to receive the images
def instance_image_callback(image):
    instance_image_queue.put(image)

def rgb_image_callback(image):
    rgb_image_queue.put(image)

def seg_image_callback(image):
    image.convert(carla.ColorConverter.CityScapesPalette)
    seg_image_queue.put(image)

def is_big_enough(pixel_coordinates, t_width, t_height):
    #return True
    # Calculate the minimum and maximum x and y coordinates
    min_x, max_x = min(coord[0] for coord in pixel_coordinates), max(coord[0] for coord in pixel_coordinates)
    min_y, max_y = min(coord[1] for coord in pixel_coordinates), max(coord[1] for coord in pixel_coordinates)

    # Check if the contour is smaller than the desired rectangle dimensions
    if max_x - min_x + 1 < t_width or max_y - min_y + 1 < t_height:
        return False

    # Create sets of unique x and y values
    unique_x = set(coord[0] for coord in pixel_coordinates)
    unique_y = set(coord[1] for coord in pixel_coordinates)

    # Iterate over unique x and y values to check if the rectangle can fit
    # Iterate over unique x and y values to check if the rectangle can fit
    for x in unique_x:
        # Skip if the rectangle exceeds the contour's maximum x coordinate
        if x + t_width - 1 > max_x:
            continue
        for y in unique_y:
            # Skip if the rectangle exceeds the contour's maximum y coordinate
            if y + t_height - 1 > max_y:
                continue
            # Check if any pixel coordinates are within the current rectangle
            if any((x <= coord[0] <= x + t_width - 1) and (y <= coord[1] <= y + t_height - 1) for coord in pixel_coordinates):
                return True
    return False

def is_area_isolated(pixel_coordinates, t_width, t_height):
    # Calculate the minimum and maximum x and y coordinates
    min_x, max_x = min(coord[0] for coord in pixel_coordinates), max(coord[0] for coord in pixel_coordinates)
    min_y, max_y = min(coord[1] for coord in pixel_coordinates), max(coord[1] for coord in pixel_coordinates)

    # Check if the contour is smaller than the desired rectangle dimensions
    if max_x - min_x + 1 < t_width or max_y - min_y + 1 < t_height:
        return False
    width=(max_x-min_x)/2
    height=(max_y - min_y)/2

    # Create a new list of pixel coordinates within the range of min_x, max_x, min_y, max_y
    pixel_coordinates_2 = [(x, y) for x in range(min_x, max_x + 1) for y in range(min_y, max_y + 1) if (x, y) not in pixel_coordinates]

    # Check if the new area is big enough using the is_big_enough function
    return is_big_enough(pixel_coordinates_2, width, height)
def process_colors(rgb_image_draw, blended_image, sem_array, multiple_bbox_tags_colors, min_width, min_height):
    #rgb_image_draw = blended_image.copy()
    pixels = blended_image.reshape(-1, 3)
    unique_colors = np.unique(pixels, axis=0)
    colors = {tuple(color) for color in unique_colors}

    for color in colors:
        mask = np.all(blended_image == color, axis=-1)
        mask_uint8 = np.uint8(mask)
        nonzero_pixels = np.argwhere(mask != 0)

        if len(nonzero_pixels) > 0:
            x, y = nonzero_pixels[0][1], nonzero_pixels[0][0]
            sem_color = tuple(sem_array[y, x])
            colorx = tuple(int(round(c)) for c in sem_color)
            bgr_color = tuple(reversed(sem_color))

            if bgr_color in multiple_bbox_tags_colors:
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            else:
                coordinates = [(coord[1], coord[0]) for coord in nonzero_pixels]
                isolated = False#is_area_isolated(coordinates, min_width, min_height)
                if isolated:
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                else:
                    if is_big_enough(coordinates, min_width, min_height):
                        min_x, max_x = np.min(nonzero_pixels[:, 1]), np.max(nonzero_pixels[:, 1])
                        min_y, max_y = np.min(nonzero_pixels[:, 0]), np.max(nonzero_pixels[:, 0])
                        if not is_object_fragmented(blended_image, color, min_x, min_y, max_x, max_y):
                            cv2.rectangle(rgb_image_draw, (min_x, min_y), (max_x, max_y), colorx, 2)
                    continue

            for contour in contours:
                pixel_coordinates = contour.squeeze(axis=1).tolist()
                if len(pixel_coordinates) > 0:
                    if is_big_enough(pixel_coordinates, min_width, min_height):
                        min_x, max_x = min(coord[0] for coord in pixel_coordinates), max(coord[0] for coord in pixel_coordinates)
                        min_y, max_y = min(coord[1] for coord in pixel_coordinates), max(coord[1] for coord in pixel_coordinates)
                        cv2.rectangle(rgb_image_draw, (min_x, min_y), (max_x, max_y), colorx, 2)

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
            actor.set_autopilot(True)
            spawned_actors.append(actor)
world.tick()


# Save the image to disk
num_ticks = 5000  # Number of ticks to execute

# Define the semantic tags to handle with multiple bounding boxes
multiple_bbox_tags = ['sidewalks','vegetation','traffic_sign','sky','traffic_light','static','dynamic']  # Modify with your specific semantic tags
CLASS_MAPPING = {
        'unlabeled': (0, 0, 0),
        'building': (70, 70, 70),
        'fence': (100, 40, 40),
        'other': (55, 90, 80),
        'pedestrian': (220, 20, 60),
        'cyclist': (255, 0, 0),
        'pole': (153, 153, 153),
        'roadLines': (157, 234, 50),
        'roads': (128, 64, 128),
        'sidewalks': (244, 35, 232),
        'vegetation': (107, 142, 35),
        'bicycle' : (119,  11,  32),
        'bus': (0, 60, 100),
        'car': (0, 0, 142),
        'truck': (0, 0, 70),
        'motorcycle': (0, 0, 230),
        'vehicle': (0, 0, 142),
        'walls': (102, 102, 156),
        'traffic_sign': (220, 220, 0),
        'sky': (70, 130, 180),
        'ground': (81, 0, 81),
        'bridge': (150, 100, 100),
        'rail_track': (230, 150, 140),
        'guard_rail': (180, 165, 180),
        'traffic_light': (250, 170, 30),
        'static': (110, 190, 160),
        'dynamic': (170, 120, 50),
        'water': (45, 60, 150),
        'terrain': (145, 170, 100)
    }
multiple_bbox_tags_colors  = [CLASS_MAPPING[label] for label in multiple_bbox_tags]
min_width=5
min_height=5
#CLASS_MAPPING[label]
for t in range(num_ticks):
    # Tick the simulation
    world.tick()

    if t%100 == 0:
        # Get the instance segmentation image
        instance_image = instance_image_queue.get()
        sem_image = seg_image_queue.get()

        # Get the RGB image
        rgb_image = rgb_image_queue.get()
        alpha = 0.5  # Adjust the alpha value as needed (0.0 to 1.0)
        # Perform alpha blending



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

        process_colors(rgb_image_draw,blended_image,sem_array, multiple_bbox_tags_colors, min_width, min_height)
        result_image_path = f'result_{t}.png'
        cv2.imwrite(result_image_path, rgb_image_draw)

    time.sleep(0.1)  # Wait for a short period before the next tick


# Set the simulator back to asynchronous mode

    #return rgb_image_draw
settings.synchronous_mode = False
settings.fixed_delta_seconds = None
world.apply_settings(settings)
all_actors = world.get_actors()
print("")
print("_______________________________")
print("all actors")
print("_______________________________")
for actor in all_actors:
    print("")
    print(actor.id )
    print(actor.type_id)
    print(actor.parent)
    print(actor.attributes)
    print(actor.bounding_box)

print("")
print("_______________________________")
print("other objects")
print("_______________________________")
obj= world.get_environment_objects()
for o in obj:
    print("")
    print(o)
# Destroy all actors
for actor in spawned_actors:
    actor.destroy()
instance_camera.destroy()
rgb_camera.destroy()
vehicle_insance.destroy()
