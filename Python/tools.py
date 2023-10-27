import numpy as np
from queue import Queue
from collections import namedtuple
import carla
import configparser
import platform
import subprocess
import os
import psutil
import yaml
import random
import math
import time
import cv2
import sys
import json
#from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5 import QtGui
from PyQt5.QtCore import Qt  #QTimer, QThread, pyqtSignal
selected_labels=['Bicycle','Bus','Car','Motorcycle','Rider','Pedestrians','traffic_light','traffic_sign','Truck']
labels_of_interest = ['fence','cyclist','pedestrian','pole','bicycle','bus','car','truck','motorcycle','traffic_sign','traffic_light','static','dynamic']
messages = []
CLASS_MAPPING = {
    'unlabeled': (0, 0, 0),
    'building': (70, 70, 70),
    'buildings': (70, 70, 70),
    'fence': (100, 40, 40),
    'fences': (100, 40, 40),
    'other': (55, 90, 80),
    'pedestrian': (220, 20, 60),
    'pedestrians': (220, 20, 60),
    'cyclist': (255, 0, 0),
    'rider':(255,0,0),
    'pole': (153, 153, 153),
    'roadLines': (157, 234, 50),
    'roadlines': (157, 234, 50),
    'roads': (128, 64, 128),
    'sidewalks': (244, 35, 232),
    'vegetation': (107, 142, 35),
    'bicycle': (119, 11, 32),
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
    'railtrack' : (230, 150, 140),
    'guard_rail': (180, 165, 180),
    'guardrail' :  (180, 165, 180),
    'traffic_light': (250, 170, 30),
    'static': (110, 190, 160),
    'dynamic': (170, 120, 50),
    'water': (45, 60, 150),
    'terrain': (145, 170, 100)
}
multiple_bbox_tags = ['sidewalks', 'vegetation', 'traffic_sign', 'sky', 'traffic_light', 'static', 'dynamic']
def color_to_label(color, semantic_tags):
    # Extract the blue channel (B) from the color
    blue_channel = color[0]

    for tag, data in semantic_tags.items():
        if 'id' in data and data['id'] == blue_channel:
            return tag

    # If no matching label is found, return a default label
    return 'unknown'
def find_color_in_semantic_colors(color_to_find, semantic_colors, width):
    flattened_semantic_colors = semantic_colors.tolist()
    matches = np.where(np.all(semantic_colors == color_to_find, axis=1))[0]
    
    if len(matches) > 0:
        y, x = np.unravel_index(matches[0], (width, len(semantic_colors) // width))
        return x, y
    else:
        return None
def list_distinct_colors(semantic_colors):
    if len(semantic_colors.shape) == 2:
        # For (height*width, 3) shape, convert to list of tuples
        semantic_colors_list = semantic_colors.tolist()
    else:
        # For (height, width, 3) shape, flatten and convert to list of tuples
        semantic_colors_list = semantic_colors.reshape(-1, 3).tolist()

    distinct_colors = list(set(map(tuple, semantic_colors_list)))
    return distinct_colors

def list_distinct_colors_2(semantic_colors):
    semantic_colors_list = semantic_colors.tolist()
    distinct_colors = list(set(map(tuple, map(tuple, semantic_colors_list))))
    return distinct_colors

def merge_segmentation_images(instance_segmentation, semantic_segmentation):
    height, width, channels = semantic_segmentation.shape

    if channels == 4:  # Check for RGBA format and suppress the alpha channel
        semantic_segmentation = semantic_segmentation[:, :, :3]
        instance_segmentation = instance_segmentation[:, :, :3]
    
    merged_image = np.zeros((height, width, 3), dtype=np.uint8)
    semantic_colors_bgr = semantic_segmentation.reshape(-1, 3)
    instance_colors_bgr = instance_segmentation.reshape(-1, 3)
    ldst=list_distinct_colors(semantic_colors_bgr)
    print(ldst)
    semantic_to_id = {}
    for tag, data in semantic_tags.items():
        color = data['color']
        semantic_to_id[color] = (data['id'], data['id'])

    semantic_colors = semantic_colors_bgr[:, ::-1]
    instance_colors = instance_colors_bgr[:, ::-1]

    idssdd = find_color_in_semantic_colors((190, 153, 153), semantic_colors, width)
    print(idssdd)

    mask = np.isin(semantic_colors, np.array(list(semantic_to_id.keys()))).all(axis=1)
    semantic_color_ids = np.array([semantic_to_id[tuple(color)] for color in semantic_colors])

    merged_image = np.column_stack((semantic_color_ids[:, 0], instance_colors[:, 1], instance_colors[:, 2]))
    merged_image = merged_image.reshape(height, width, 3)  # 3 channels (Red, Green, Blue)

    return merged_image


def move_view(spectator,target_transform):
    distance_behind = 15  # Adjust the distance behind the target
    height_above = 15  # Adjust the height above the target

    # Calculate the spectator's position relative to the target
    rotation_yaw = target_transform.rotation.yaw  # Use the target's yaw as the rotation angle
    spectator_location = target_transform.location - carla.Location(
        x=distance_behind * math.cos(math.radians(rotation_yaw)),
        y=distance_behind * math.sin(math.radians(rotation_yaw)),
        z=-height_above
    )
    #spectator.set_location(spectator_location)

    # Set the spectator's rotation to point towards the target
    spectator_rotation = carla.Rotation(pitch=-40, yaw=rotation_yaw, roll=0)  # Adjust the rotation angles
    spectator.set_transform(carla.Transform(spectator_location,spectator_rotation))


# Sleep to allow time for the spectator camera to update

def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def convert_coordinates(reference_transform, location, orientation):
    reference_location = reference_transform.location
    reference_rotation = reference_transform.rotation

    # Convert local X and Y coordinates to global coordinates
    x = -location[0] * math.cos(math.radians(reference_rotation.yaw)) + location[1] * math.sin(
        math.radians(reference_rotation.yaw))
    y = location[0] * math.sin(math.radians(reference_rotation.yaw)) + location[1] * math.cos(
        math.radians(reference_rotation.yaw))

    # Calculate global location
    global_location = carla.Location(reference_location.x + x, reference_location.y + y, location[2] + 1)

    # Calculate global rotation
    global_pitch = 0
    global_yaw = reference_rotation.yaw + orientation[1]
    global_roll = 0
    global_rotation = carla.Rotation(global_pitch, global_yaw, global_roll)

    return global_location, global_rotation


def spawn_vehicle(world, spectator_transform, location, orientation, speed):
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = random.choice(blueprint_library.filter('vehicle.audi*'))

    spawn_location, spawn_rotation = convert_coordinates(spectator_transform, location, orientation)

    spawn_transform = carla.Transform(spawn_location, spawn_rotation)

    print("vehicle is spawned at")
    print(spawn_transform)
    vehicle = world.spawn_actor(vehicle_bp, spawn_transform)
    control = carla.VehicleControl()
    control.throttle = speed
    vehicle.apply_control(control)

    return vehicle


def spawn_pedestrian(world, spectator_transform, location, orientation, speed):
    blueprint_library = world.get_blueprint_library()
    pedestrian_bp = random.choice(
        blueprint_library.filter('walker.pedestrian.0009'))  # blueprint_library.find('walker.pedestrian.*')

    spawn_location, spawn_rotation = convert_coordinates(spectator_transform, location, orientation)

    spawn_transform = carla.Transform(spawn_location, spawn_rotation)

    print("pedistrian is spawned at")
    print(location)
    print(orientation)
    print(spawn_transform)
    pedestrian = world.spawn_actor(pedestrian_bp, spawn_transform)
    print(pedestrian.get_transform())
    control = carla.WalkerControl()
    control.speed = speed
    pedestrian.apply_control(control)
    pedestrian.set_transform(spawn_transform)

    return pedestrian


def change_vehicle_direction(vehicle, direction):
    control = carla.VehicleControl()
    control.steer = direction[0]
    control.throttle = direction[1]
    vehicle.apply_control(control)


def pedestrian_jump(pedestrian, spectator_transform, location, orientation, speed):
    new_location, new_rotation = convert_coordinates(spectator_transform, location, orientation)

    new_transform = carla.Transform(new_location, new_rotation)
    pedestrian.set_transform(new_transform)

    control = carla.WalkerControl()
    control.speed = speed
    pedestrian.apply_control(control)
    print("pedistrian jump to")
    print(pedestrian.get_transform())


def load_scenario_from_yaml(file_path):
    with open(file_path, 'r') as file:
        scenario_data = yaml.safe_load(file)
    return scenario_data


def move_spectator(world, location, orientation):
    spectator = world.get_spectator()
    new_transform = carla.Transform(carla.Location(*location), carla.Rotation(*orientation))
    spectator.set_transform(new_transform)
    world.tick()
    print("scenario is at")
    print(spectator.get_transform())
def draw_filtered_bounding_boxes(image_path, bounding_boxes, output_path, labels_of_interest=None):
    # Read the input image
    image = cv2.imread(image_path)

    # Create a copy of the image to draw on
    image_with_boxes = image.copy()

    for bbox in bounding_boxes:
        # Extract bounding box coordinates, label, and color
        min_x, max_x, min_y, max_y = bbox['min_x'], bbox['max_x'], bbox['min_y'], bbox['max_y']
        color = bbox['color']
        label = bbox.get('label', '')  # Get the label or use an empty string if not provided

        # Check if the label is in the list of labels of interest
        if labels_of_interest is not None and label not in labels_of_interest:
            continue  # Skip drawing if the label is not of interest

        # Draw the bounding box on the image
        cv2.rectangle(image_with_boxes, (min_x, min_y), (max_x, max_y), color, thickness=2)

        # Add label text if provided
        if label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image_with_boxes, label, (min_x, min_y - 10), font, 0.5, color, 2)

    # Save the image with bounding boxes to the output path
    cv2.imwrite(output_path, image_with_boxes)

def draw_bounding_boxes(image_path, bounding_boxes, output_path):
    # Read the input image
    image = cv2.imread(image_path)

    # Create a copy of the image to draw on
    image_with_boxes = image.copy()

    for bbox in bounding_boxes:
        # Extract bounding box coordinates and label
        min_x, max_x, min_y, max_y = bbox['min_x'], bbox['max_x'], bbox['min_y'], bbox['max_y']
        color = bbox['color']
        label = bbox.get('label', '')  # Get the label or use an empty string if not provided

        # Draw the bounding box on the image
        cv2.rectangle(image_with_boxes, (min_x, min_y), (max_x, max_y), color, thickness=2)

        # Add label text if provided
        if label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image_with_boxes, label, (min_x, min_y - 10), font, 0.5, color, 2)

    # Save the image with bounding boxes to the output path
    cv2.imwrite(output_path, image_with_boxes)
def find_color_regions(image):
    bounding_boxes = []
    height, width, channels = image.shape
    # Convert the image to the RGB format if it's in BGR
    if channels == 4:  # Check for RGBA format and suppress the alpha channel
        image = image[:, :, :3]

    image_colors = list_distinct_colors(image)
    
    for color in image_colors:
        mask = np.all(image == color, axis=-1)
        
        # Calculate the bounding box directly from the mask
        non_zero_pixels = np.where(mask)
        min_x, max_x = min(non_zero_pixels[1]), max(non_zero_pixels[1])
        min_y, max_y = min(non_zero_pixels[0]), max(non_zero_pixels[0])

        bounding_box = {
            'min_x': min_x,
            'max_x': max_x,
            'min_y': min_y,
            'max_y': max_y,
            'color': tuple(color),
            'label': color_to_label(color,semantic_tags)
        }
        bounding_boxes.append(bounding_box)

    return bounding_boxes
def find_color_regions_contours(image):
    bounding_boxes = []
    height, width, channels = image.shape
    min_dim = min(height, width)

    # Convert the image to the RGB format if it's in BGR
    if channels == 4:  # Check for RGBA format and suppress the alpha channel
        image = image[:, :, :3]

    image_colors = list_distinct_colors(image)

    # Convert the image to a NumPy array for more efficient processing
    image_np = np.asarray(image)

    for color in image_colors:
        mask = np.all(image_np == color, axis=-1)
        contours, _ = cv2.findContours(np.uint8(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        merged_boxes = []  # List to store merged bounding boxes

        for contour in contours:
            pixel_coordinates = contour.squeeze(axis=1)
            if pixel_coordinates.shape[0] > 0:
                min_x, max_x = pixel_coordinates[:, 0].min(), pixel_coordinates[:, 0].max()
                min_y, max_y = pixel_coordinates[:, 1].min(), pixel_coordinates[:, 1].max()

                if max_x - min_x > 3 and max_y - min_y > 3:
                    label= color_to_label(color, semantic_tags)
                    b_id=  f"b_box_{label}_{color[1]}{color[2]}"
                    bounding_box = {
                        'id': b_id,
                        'min_x': int(min_x),
                        'max_x': int(max_x),
                        'min_y': int(min_y),
                        'max_y': int(max_y),
                        #'color': tuple(color),
                        'label': label
                    }

                    # Check if this bounding box can be merged with any existing merged bounding boxes
                    merge_attempt = False
                    for merged_box in merged_boxes:
                        h_dist = max(0, min_x - merged_box['max_x'], merged_box['min_x'] - max_x)
                        v_dist = max(0, min_y - merged_box['max_y'], merged_box['min_y'] - max_y)
                        max_dist = max(h_dist, v_dist)

                        if max_dist < min_dim / 40:
                            # Merge the bounding boxes
                            merged_box['min_x'] = int(min(merged_box['min_x'], min_x))
                            merged_box['max_x'] = int(max(merged_box['max_x'], max_x))
                            merged_box['min_y'] = int(min(merged_box['min_y'], min_y))
                            merged_box['max_y'] = int(max(merged_box['max_y'], max_y))
                            merge_attempt = True
                            break

                    if not merge_attempt:
                        merged_boxes.append(bounding_box)

        bounding_boxes.extend(merged_boxes)

    return bounding_boxes
def find_color_regions_contours_1(image):

    bounding_boxes = []
    height, width, channels = image.shape
    min_dim=min(height,width)
    # Convert the image to the RGB format if it's in BGR
    if channels == 4:  # Check for RGBA format and suppress the alpha channel
        image = image[:, :, :3]
    image_bgr = image.reshape(-1, 3)
    
    image_colors = list_distinct_colors(image)
    
    for color in image_colors:
        mask = np.all(image == color, axis=-1)
        contours, _ = cv2.findContours(np.uint8(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        merged_boxes = []  # List to store merged bounding boxes

        for contour in contours:
            pixel_coordinates = contour.squeeze(axis=1).tolist()
            if len(pixel_coordinates) > 0:
                min_x, max_x = min(coord[0] for coord in pixel_coordinates), max(
                    coord[0] for coord in pixel_coordinates)
                min_y, max_y = min(coord[1] for coord in pixel_coordinates), max(
                    coord[1] for coord in pixel_coordinates)
                if max_x-min_x>3 and max_y - min_y >3:

                    bounding_box = {
                        'min_x': min_x,
                        'max_x': max_x,
                        'min_y': min_y,
                        'max_y': max_y,
                        'color': tuple(color),
                        'label': color_to_label(color,semantic_tags)
                    }

                    # Check if this bounding box can be merged with any existing merged bounding boxes
                    merge_attempt = False
                    for merged_box in merged_boxes:
                        left= (min_x < merged_box['min_x'])
                        right= (min_x > merged_box['min_x'])
                        up = (min_y < merged_box['min_y'])
                        down = (min_y > merged_box['min_y'])
                        h_dist =0
                        v_dist=0

                        if left:
                            h_dist= max(h_dist,(merged_box['min_x']-max_x))
                        if right:
                            h_dist= max(h_dist,(min_x-merged_box['max_x']))
                        if up:
                            v_dist= max(v_dist,(merged_box['min_y']-max_y))
                        if down:
                            v_dist= max(v_dist,(min_y-merged_box['max_y']))

                        max_dist= max(h_dist,v_dist)

                        if max_dist<min_dim/40:
                            # Merge the bounding boxes
                            merged_box['min_x'] = min(merged_box['min_x'], min_x)
                            merged_box['max_x'] = max(merged_box['max_x'], max_x)
                            merged_box['min_y'] = min(merged_box['min_y'], min_y)
                            merged_box['max_y'] = max(merged_box['max_y'], max_y)
                            merge_attempt = True
                            break
                    
                    if not merge_attempt:
                        merged_boxes.append(bounding_box)

        bounding_boxes.extend(merged_boxes)

    return bounding_boxes

def update_seg_camera_view(image, view, desired_width, desired_height):
    image.convert(carla.ColorConverter.CityScapesPalette)
    np_img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    np_img = np_img.reshape((image.height, image.width, 4))
    np_img = np_img[..., :3]  # Remove the alpha channel
    # q_image = QtGui.QImage(bytes(np_img.data), image.width, image.height, QtGui.QImage.Format_RGB888)
    # q_image = q_image.rgbSwapped()
    # self.semantic_image= image
    # Use the palette to create a QImage
    q_image = QtGui.QImage(bytes(np_img.data), image.width, image.height, QtGui.QImage.Format_RGB888).rgbSwapped()
    # q_image = q_image.rgbSwapped()
    # self.semsegConversion(q_image)
    # q_image.setColorTable(palette_list)
    scaled_q_image = q_image.scaled(desired_width, desired_height, Qt.AspectRatioMode.KeepAspectRatio)
    pixmap = QtGui.QPixmap.fromImage(scaled_q_image)
    # pixmap = QtGui.QPixmap.fromImage(q_image)
    view.setPixmap(pixmap)
    return image
    # self.camera_tick +=1
    # self.synchroTick()


def update_rgb_camera_view(image, view, desired_width, desired_height):
    # Convert the Image object to a QImage object
    # new_raw_data = image.raw_data#bytearray(image.raw_data)
    # new_image = ImageObj(raw_data=new_raw_data, width=image.width, height=image.height,fov=image.fov)

    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    q_image = QtGui.QImage(array.data, image.width, image.height, QtGui.QImage.Format_RGB32)
    # pixmap = QtGui.QPixmap.fromImage(q_image)
    scaled_q_image = q_image.scaled(desired_width, desired_height, Qt.AspectRatioMode.KeepAspectRatio)
    pixmap = QtGui.QPixmap.fromImage(scaled_q_image)
    view.setPixmap(pixmap)
    return get_RGB_DATA(image)


def get_RGB_DATA(image):
    new_raw_data = copyImageData(image)
    # copy.deepcopy(image.raw_data)#bytearray(image.raw_data)
    new_image = ImageObj(raw_data=new_raw_data, width=image.width, height=image.height, fov=image.fov)
    return new_image


def copyImageData(source_image):
    if source_image is None:
        return None

    source_array = np.array(source_image.raw_data)
    # source_array = source_array.reshape((source_image.height, source_image.width, 4))

    # Create a new RGB image
    new_image = np.zeros(source_array.shape,
                         dtype=np.uint8)  # np.zeros((source_image.height, source_image.width, 3), dtype=np.uint8)

    # Copy the pixel values from source_array to new_image
    new_image[:] = source_array[:]
    # new_image[:] = source_array[:, :, :3]
    return new_image


# Sensor callback.
# This is where you receive the sensor data and
# process it as you liked and the important part is that,
# at the end, it should include an element into the sensor queue.
synchro_queue = Queue()

ImageObj = namedtuple('ImageObj', ['raw_data', 'width', 'height', 'fov'])

semantic_tags={
    'unknown': {'color':(190,153,153), 'tag':0, 'id': 0},
    'not_defined': {'color':(152,251,152), 'tag':0, 'id': 0},
    'unlabeled': {'color':(0, 0, 0), 'tag':0, 'id': 0},
    'building': {'color':(70, 70, 70), 'tag':1, 'id': 10},
    'buildings': {'color':(70, 70, 70), 'tag':1, 'id': 10},
    'fence': {'color':(100, 40, 40), 'tag':2, 'id': 20},
    'fences': {'color':(100, 40, 40), 'tag':2,'id': 20},
    'other': {'color':(55, 90, 80), 'tag':3, 'id': 30},
    'pedestrian': {'color':(220, 20, 60), 'tag':4, 'id': 40},
    'pedestrians': {'color':(220, 20, 60), 'tag':4, 'id': 40},
    'cyclist': {'color':(255, 0, 0), 'tag':4, 'id': 45},
    'rider':{'color':(255, 0, 0), 'tag':4, 'id': 45},
    'pole': {'color':(153, 153, 153), 'tag':5, 'id': 50},
    'roadLines': {'color':(157, 234, 50), 'tag':6, 'id': 60},
    'roadLine': {'color':(157, 234, 50), 'tag':6, 'id': 60},
    'roadlines': {'color':(157, 234, 50), 'tag':6, 'id': 60},
    'roads': {'color':(128, 64, 128), 'tag':7, 'id': 70},
    'sidewalks': {'color':(244, 35, 232), 'tag':8, 'id': 80},
    'sidewalk': {'color':(244, 35, 232), 'tag':8, 'id': 80},
    'vegetation': {'color':(107, 142, 35), 'tag':9, 'id': 90},
    'bicycle': {'color':(119, 11, 32), 'tag':10,'id': 100},
    'bus': {'color':(0, 60, 100), 'tag':10, 'id': 102},
    'car': {'color':(0, 0, 142), 'tag':10, 'id': 104},
    'truck': {'color':(0, 0, 70), 'tag':10, 'id': 106},
    'motorcycle': {'color':(0, 0, 230), 'tag':10, 'id': 108},
    'vehicle': {'color':(0, 0, 142), 'tag':10, 'id': 104},
    'walls': {'color':(102, 102, 156), 'tag':11, 'id': 110},
    'traffic_sign': {'color':(220, 220, 0), 'tag':12, 'id': 120},
    'sky': {'color':(70, 130, 180), 'tag':13, 'id': 130},
    'ground': {'color':(81, 0, 81), 'tag':14, 'id': 140},
    'bridge': {'color':(150, 100, 100), 'tag':15, 'id': 150},
    'rail_track': {'color':(230, 150, 140), 'tag':16, 'id': 160},
    'railtrack' : {'color':(230, 150, 140), 'tag':16, 'id': 160},
    'guard_rail': {'color':(180, 165, 180), 'tag':17, 'id': 170},
    'guardrail' :  {'color':(180, 165, 180), 'tag':17, 'id': 170},
    'traffic_light': {'color':(250, 170, 30), 'tag':18, 'id': 180},
    'static': {'color':(110, 190, 160), 'tag':19, 'id': 190},
    'dynamic': {'color':(170, 120, 50), 'tag':20, 'id': 200},
    'water': {'color':(45, 60, 150), 'tag':21, 'id': 210},
    'terrain': {'color':(145, 170, 100), 'tag':22, 'id': 220}
}
bb_labels= {
    #'Any' : carla.CityObjectLabel.Any,
    'Bicycle' : carla.CityObjectLabel.Bicycle,
    #'Bridge' : carla.CityObjectLabel.Bridge,
    'Buildings' : carla.CityObjectLabel.Buildings,
    'Bus' : carla.CityObjectLabel.Bus,
    'Car' : carla.CityObjectLabel.Car,
    'Dynamic' : carla.CityObjectLabel.Dynamic,
    'Fences' : carla.CityObjectLabel.Fences,
    'Ground' : carla.CityObjectLabel.Ground,
    'GuardRail' : carla.CityObjectLabel.GuardRail,
    'Motorcycle' : carla.CityObjectLabel.Motorcycle,
    #'NONE' : carla.CityObjectLabel.NONE,
    'Other' : carla.CityObjectLabel.Other,
    'Pedestrians' : carla.CityObjectLabel.Pedestrians,
    'Pole' : carla.CityObjectLabel.Poles,
    'RailTrack' : carla.CityObjectLabel.RailTrack,
    'Rider' : carla.CityObjectLabel.Rider,
    'RoadLines' : carla.CityObjectLabel.RoadLines,
    'Roads' : carla.CityObjectLabel.Roads,
    'Sidewalks' : carla.CityObjectLabel.Sidewalks,
    'Sky' : carla.CityObjectLabel.Sky,
    'Static' : carla.CityObjectLabel.Static,
    'Terrain' : carla.CityObjectLabel.Terrain,
    'traffic_light' : carla.CityObjectLabel.TrafficLight,
    'traffic_sign' : carla.CityObjectLabel.TrafficSigns,
    #'Train' : carla.CityObjectLabel.Train,
    'Truck' : carla.CityObjectLabel.Truck,
    'Vegetation' : carla.CityObjectLabel.Vegetation,
    'Walls' : carla.CityObjectLabel.Walls,
    'Water' : carla.CityObjectLabel.Water

}


def max_projected_length(length, distance, K):
    f = K[0, 0]
    l_max = (f * length) / distance
    return l_max


def is_visible(bb, camera_transform, k):
    #forward_vec = camera_transform.get_forward_vector()
    #bb_direction = bb.location - camera_transform.location
    dist = bb.location.distance(camera_transform.location)
    #bb_width = bb.extent.z
    #if max_projected_length(bb_width, dist, k) < 2:
        # print(bb)
        # print("too small")
    ##    return False
    dist=camera_transform.location.distance(bb.location)
    if dist>1000:
        return False
    return True
    #dot_product = (forward_vec.x) * (bb_direction.x) + forward_vec.y * (bb_direction.y) + forward_vec.z * (
    #    bb_direction.z)
    # if dot_product<=0:
    # print(bb)
    # print("behind camera")
    #return dot_product > 0


def expand_bb(bounding_boxes):
    for label in ["Bicycle", "Motorcycle"]:
        if label in bounding_boxes:
            for bb in bounding_boxes[label]:
                # Expand the bounding box and append it to the modified_bbs list
                # Add a new entry for "Rider" if it doesn't exist
                if "Rider" not in bounding_boxes:
                    bounding_boxes["Rider"] = []
                # Add a new bounding box for "Rider" with the same location and dimensions
                rider_bb = carla.BoundingBox(carla.Location(bb.location), carla.Vector3D(bb.extent))
                bounding_boxes["Rider"].append(rider_bb)


def process_colors(rgb_image_draw, blended_image, sem_array, multiple_bbox_tags_colors, min_width, min_height,
                   CLASS_MAPPING):
    bounding_boxes = []

    for label, color in CLASS_MAPPING.items():
        mask = np.all(blended_image == color, axis=-1)
        if color in multiple_bbox_tags_colors:
            contours, _ = cv2.findContours(np.uint8(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            isolated = False  # is_area_isolated(mask, min_width, min_height)
            if isolated:
                contours, _ = cv2.findContours(np.uint8(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            else:
                continue

        for contour in contours:
            pixel_coordinates = contour.squeeze(axis=1).tolist()
            if len(pixel_coordinates) > 0:
                if is_big_enough(pixel_coordinates, min_width, min_height):
                    min_x, max_x = min(coord[0] for coord in pixel_coordinates), max(
                        coord[0] for coord in pixel_coordinates)
                    min_y, max_y = min(coord[1] for coord in pixel_coordinates), max(
                        coord[1] for coord in pixel_coordinates)

                    if not is_object_fragmented(blended_image, color, min_x, min_y, max_x, max_y):
                        cv2.rectangle(rgb_image_draw, (min_x, min_y), (max_x, max_y), color, 2)
                        bounding_boxes.append(((min_x, min_y), (max_x, max_y), label))

    return bounding_boxes


def is_object_fragmented(instance_image, class_color, min_x, min_y, max_x, max_y):
    # Crop the instance image within the bounding box
    cropped_image = instance_image[min_y:max_y, min_x:max_x]

    nb_total = (max_y - min_y) * (max_x - min_x)

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


def is_big_enough(pixel_coordinates, t_width, t_height):
    # return True
    # Calculate the minimum and maximum x and y coordinates
    min_x, max_x = min(coord[0] for coord in pixel_coordinates), max(coord[0] for coord in pixel_coordinates)
    min_y, max_y = min(coord[1] for coord in pixel_coordinates), max(coord[1] for coord in pixel_coordinates)

    # Check if the contour is smaller than the desired rectangle dimensions
    if ((max_x - min_x + 1) < (t_width)) or ((max_y - min_y + 1) < (t_height)):
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
            if any((x <= coord[0] <= x + t_width - 1) and (y <= coord[1] <= y + t_height - 1) for coord in
                   pixel_coordinates):
                return True
    return False


def is_area_isolated(pixel_coordinates, t_width, t_height):
    # Calculate the minimum and maximum x and y coordinates
    min_x, max_x = min(coord[0] for coord in pixel_coordinates), max(coord[0] for coord in pixel_coordinates)
    min_y, max_y = min(coord[1] for coord in pixel_coordinates), max(coord[1] for coord in pixel_coordinates)

    # Check if the contour is smaller than the desired rectangle dimensions
    w = max_x - min_x + 1
    h = max_y - min_y + 1
    if (w < t_width) or (h < t_height):
        return False
    width = (max_x - min_x) / 2
    height = (max_y - min_y) / 2

    # Create a new list of pixel coordinates within the range of min_x, max_x, min_y, max_y
    pixel_coordinates_2 = [(x, y) for x in range(min_x, max_x + 1) for y in range(min_y, max_y + 1) if
                           (x, y) not in pixel_coordinates]

    # Check if the new area is big enough using the is_big_enough function
    return is_big_enough(pixel_coordinates_2, width, height)


def sensor_callback(world, actor, sensor_data, synchro_queue, sensor_name, K=None):
    if sensor_name == 'rgb_camera':
        transform = actor.get_transform()
        # Calculate distances to camera for each bounding box
        bounding_boxes_with_distances = {
            label: [(bb, bb.location.distance(transform.location)) for bb in world.get_level_bbs(bb_labels[label]) if
                    is_visible(bb, transform, K)]
            for label in selected_labels
        }
        # Sort bounding boxes by distance from camera in place
        sorted_bounding_boxes = {
            label: [bb for bb, _ in sorted(bounding_boxes_with_distances[label], key=lambda x: x[1])]
            for label in selected_labels
        }
        bounding_boxes_3d = sorted_bounding_boxes
        expand_bb(bounding_boxes_3d)
        camera_transform = carla.Transform(transform.location, transform.rotation)
        synchro_queue.put((sensor_data.frame, "bounding_boxes", bounding_boxes_3d))
        synchro_queue.put((sensor_data.frame, "camera_transform", camera_transform))
        # print(sensor_data.frame)

    synchro_queue.put((sensor_data.frame, sensor_name, sensor_data))



config = configparser.ConfigParser()
config.read('config.ini')
carla_path = config.get('Carla', 'path')


def check_carla_server():
    print("Attempting to connect to Carla server .. ")
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)
        client.get_world()
        return client
    except Exception as e:
        # print(f"Failed to connect to Carla server: {e}")
        return None


def check_connection_status(client):
    try:
        # Check the connection status by performing a simple operation,
        # such as retrieving the world or a specific actor
        world = client.get_world()


    except Exception as e:
        print("Connection lost:", e)
        # Close the connection or kill the client here
        client.disconnect()
        client = None


def launch_carla_server():
    # launch the Carla server
    close_carla_server()
    os_name = platform.system()

    print(f"Opening CARLA from path: {carla_path}")
    if os_name == 'Windows':
        path_to_run = os.path.join(carla_path, 'CarlaUE4.exe')
        subprocess.Popen(path_to_run, cwd=carla_path)
    elif os_name == 'Linux':
        path_to_run = os.path.join(carla_path, 'CarlaUE4.sh')
        subprocess.Popen([path_to_run, '-opengl'], cwd=carla_path)
    else:
        print('Unsupported operating system')


def close_carla_server():
    # get the process list
    processes = psutil.process_iter()

    # find all processes associated with CARLA
    carla_processes = []
    for process in processes:
        if process.name().startswith('Carla'):
            carla_processes.append(process)

    # terminate all CARLA processes
    for process in carla_processes:
        process.terminate()

    # wait for all CARLA processes to terminate
    psutil.wait_procs(carla_processes)


def run_scenario(world, scenario_data, spawned_actors, ego):
    # Get the spectator position and orientation from the scenario data
    spectator_position = scenario_data[0]['location']
    spectator_orientation = scenario_data[0]['orientation']

    # Move spectator view towards the specified position
    move_spectator(world, spectator_position, spectator_orientation)

    # Get the spectator transform for relative spawning
    time.sleep(1)
    spectator = world.get_spectator()
    spectator_transform = spectator.get_transform()

    # List to keep track of spawned actors
    # spawned_actors = []
    ego.set_autopilot(False)

    # Move the ego vehicle and cameras to the spectator location and orientation
    ego.set_transform(spectator_transform)
    for event in scenario_data[1:]:
        time.sleep(1)
        event_type = event.get('type')
        if event_type == 'spawn_vehicle':
            vehicle_id = event['vehicle_id']
            vehicle = spawn_vehicle(
                world,
                spectator_transform,
                event['location'],
                event['orientation'],
                event.get('speed', 0)
            )
            spawned_actors.append((vehicle_id, vehicle))  # Add spawned vehicle to the list
        elif event_type == 'spawn_pedestrian':
            pedestrian_id = event['pedestrian_id']
            pedestrian = spawn_pedestrian(
                world,
                spectator_transform,
                event['location'],
                event['orientation'],
                event.get('speed', 0)
            )
            spawned_actors.append((pedestrian_id, pedestrian))  # Add spawned pedestrian to the list
        elif event_type == 'change_vehicle_direction':
            vehicle_id = event['vehicle_id']
            for actor_id, actor in spawned_actors:
                if actor_id == vehicle_id:
                    change_vehicle_direction(actor, event['direction'])
                    break
        elif event_type == 'pedestrian_jump':
            pedestrian_id = event['pedestrian_id']
            for actor_id, actor in spawned_actors:
                if actor_id == pedestrian_id:
                    pedestrian_jump(
                        actor,
                        spectator_transform,
                        event['location'],
                        event['orientation'],
                        event.get('speed', 0)
                    )
                    break

    # Wait for a while to observe the scenario

    # Destroy the spawned actors


def update_bounding_box_view_simple(view, camera, img, semantic_image, bounding_box_set, transform, image_ref, counter,
                                    desired_width, desired_height, bb_id=0):
    json_array = []
    for bounding_box in bounding_box_set:
        bb, label = bounding_box[0], bounding_box[1]
        # for bb,label in bounding_box_set.items():
        object_color = CLASS_MAPPING[label]
        x_min_new, y_min_new, x_max_new, y_max_new = bb
        if not (x_min_new == -1 and y_min_new == -1 and x_max_new == -1 and y_max_new == -1):
            # Draw bounding box using OpenCV's rectangle function
            cv2.rectangle(img, (x_min_new, y_min_new), (x_max_new, y_max_new), object_color, 2)

            # label = 'vehicle'  # replace with the appropriate label for each object type
            cv2.putText(img, label, (x_min_new, y_min_new - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, object_color, 1)
            json_entry = {"id": bb_id, "label": label, "x_min": int(x_min_new), "y_min": int(y_min_new),
                          "x_max": int(x_max_new), "y_max": int(y_max_new)}
            json_array.append(json_entry)
            bb_id += 1

    # Convert the image back to a QImage object and display it
    qimage = QtGui.QImage(img.data, image_ref.width, image_ref.height, QtGui.QImage.Format_RGB32)
    scaled_qimage = qimage.scaled(desired_width, desired_height, Qt.AspectRatioMode.KeepAspectRatio)
    pixmap = QtGui.QPixmap.fromImage(scaled_qimage)
    # pixmap = QtGui.QPixmap.fromImage(qimage)

    view.setPixmap(pixmap)
    counter += 1
    return json_array


def compute_bb_distance(camera_location, corners):
    # corners = bb.get_world_vertices(carla.Transform())

    # Compute the center of the bounding box in world coordinates
    bounding_box_center = carla.Location()
    for corner in corners:
        bounding_box_center.x += corner.x
        bounding_box_center.y += corner.y
        bounding_box_center.z += corner.z
    bounding_box_center.x /= len(corners)
    bounding_box_center.y /= len(corners)
    bounding_box_center.z /= len(corners)
    distance = camera_location.distance(bounding_box_center)
    return distance


def get_image_point(loc, K, w2c):
    # Calculate 2D projection of 3D coordinate

    # Format the input coordinate (loc is a carla.Position object)
    point = np.array([loc.x, loc.y, loc.z, 1])
    # transform to camera coordinates
    point_camera = np.dot(w2c, point)

    # New we must change from UE4's coordinate system to an "standard"
    # (x, y ,z) -> (y, -z, x)
    # and we remove the fourth componebonent also
    # point_camera = [point_camera[0], -point_camera[1], point_camera[2]]
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    # now project 3D->2D using the camera matrix
    point_img = np.dot(K, point_camera)
    # normalize
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img[0:2]


def update_bounding_box_view_3D(view, camera, image, semantic_image, bounding_box_set, transform, image_ref, counter,
                                desired_width, desired_height, self):
    # Convert the Image object to a QImage object
    # Define percentage to reduce bounding box size by
    reduction_percentage = 0.1
    if image is None:
        return []
    json_array = []
    world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
    edges = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5], [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]
    img = np.frombuffer(copyImageData(image), dtype=np.dtype(
        "uint8"))  # np.frombuffer(copy.deepcopy(image.raw_data), dtype=np.dtype("uint8"))
    rgb = np.frombuffer(copyImageData(image_ref), dtype=np.dtype(
        "uint8"))  # np.frombuffer(copy.deepcopy(image.raw_data), dtype=np.dtype("uint8"))
    rgb = np.reshape(rgb, (image.height, image.width, 4))
    # rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    img = np.reshape(img, (image.height, image.width, 4))
    # carla.CityObjectLabel.Car
    bb_id = 0
    processed_boxes = {label: [] for label in bounding_box_set.keys()}
    for label, bb_set in bounding_box_set.items():
        for bb in bb_set:
            try:
                corners = bb.get_world_vertices(carla.Transform())
                distance = compute_bb_distance(camera.get_transform().location, corners)
                corners = [get_image_point(corner, self.K, world_2_camera) for corner in corners]
                object_color = CLASS_MAPPING[label]
                object_color = (object_color[2], object_color[1], object_color[0])
                # Check if the bounding box is visible to the camera
                # line_of_sight = self.world.get_line_of_sight(camera.get_location(), bb.location)
                # forward_vec = self.vehicle.get_transform().get_forward_vector()
                # bb_direction = bb.location - camera.get_transform().location
                # dot_product = forward_vec.x * bb_direction.x + forward_vec.y * bb_direction.y + forward_vec.z * bb_direction.z
                # if dot_product > 0:
                # if np.dot(forward_vec, bb_direction) > 0:
                # relative_location = bb.location - camera.get_location()
                # ang=camera.get_forward_vector().get_angle(relative_location)
                # if ang < 80 and ang>=0:
                # Define percentage to reduce bounding box size by
                verts = [v for v in bb.get_world_vertices(carla.Transform())]
                #    center_point = carla.Location()
                #    for v in verts:
                #        center_point += v
                #    center_point /= len(verts)

                # Calculate new vertices by reducing distance from center point by reduction percentage
                #    new_verts = []
                #    for v in verts:
                #        direction = v - center_point
                #        new_direction = direction - direction * reduction_percentage
                #        new_v = center_point + new_direction
                #        new_verts.append(new_v)

                # Get image coordinates of new vertices
                #    corners = [self.get_image_point(corner, self.K, world_2_camera) for corner in new_verts]

                # Use NumPy to calculate min/max corners
                corners = np.array(corners, dtype=int)
                x_min, y_min = np.min(corners, axis=0).astype(int)
                x_max, y_max = np.max(corners, axis=0).astype(int)
                included = False
                # Extract the region of interest from the semantic image using the bounding box coordinates
                # Assume that 'semantic_image' is a CARLA Image object
                semantic_data = np.frombuffer(semantic_image.raw_data, dtype=np.dtype("uint8"))
                semantic_data = np.reshape(semantic_data, (semantic_image.height, semantic_image.width, 4))
                semantic_data = semantic_data[:, :, :3]
                roi = semantic_data[y_min:y_max, x_min:x_max]

                # Count the number of pixels within the bounding box coordinates that have the correct semantic color
                count = np.sum((roi == object_color).all(axis=2))

                # Compute the total number of pixels within the bounding box coordinates
                total = roi.shape[0] * roi.shape[1]

                # If the ratio of the number of pixels with the correct semantic color to the total number of pixels is greater than or equal to 0.5, process the bounding box

                if count * 2 >= total:

                    for processed_bb in processed_boxes[label]:
                        if x_min >= processed_bb[0] and x_max <= processed_bb[2] and y_min >= processed_bb[
                            1] and y_max <= processed_bb[3] and processed_bb[4] < distance:
                            included = True
                            break
                    if not included:
                        # Process the bounding box if it's not included in any previously processed box
                        processed_boxes[label].append((x_min, y_min, x_max, y_max, distance))
                        edge_array = []
                        # Draw edges of the bounding box into the camera output
                        for edge in edges:
                            try:
                                p1 = get_image_point(verts[edge[0]], self.K, world_2_camera)
                                p2 = get_image_point(verts[edge[1]], self.K, world_2_camera)
                                # Draw the edges into the camera output
                                cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), object_color, 1)
                                edge_array.append((p1.tolist(), p2.tolist()))
                            except Exception as e:
                                print(e)
                                continue
                            # label = 'vehicle'  # replace with the appropriate label for each object type
                        cv2.putText(img, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, object_color, 1)
                        json_entry = {"id": bb_id, "label": label, "edges": edge_array}
                        json_array.append(json_entry)
                        bb_id += 1
            except Exception as e:
                continue

    # record_tick(json_array,rgb,semantic_image,img)
    # Convert the image back to a QImage object and display it
    qimage = QtGui.QImage(img.data, image.width, image.height, QtGui.QImage.Format_RGB32)
    # pixmap = QtGui.QPixmap.fromImage(qimage)

    # Convert the QImage object to a QPixmap object and display it
    # pixmap = QtGui.QPixmap.fromImage(qimage)
    scaled_qimage = qimage.scaled(desired_width, desired_height, Qt.AspectRatioMode.KeepAspectRatio)
    pixmap = QtGui.QPixmap.fromImage(scaled_qimage)
    view.setPixmap(pixmap)
    counter += 1
    return json_array


def record_tick(registry, json_array, rgb, semantic_image, img_bb, is_running, is_recording):
    if is_running and is_recording:
        # Create a dictionary with the data to be stored
        # rgb_data=self.get_RGB_DATA(rgb)
        data_dict = {
            "json_array": json_array,
            "rgb": rgb,
            "semantic_image": semantic_image,
            "img_bb": img_bb,
        }
        # Append the data dictionary to the list
        registry.append(data_dict)


class ConsoleLogger:
    def __init__(self, window=None):
        self.terminal = sys.stdout

    def flush(self):
        self.terminal.flush()

    def write(self, message, window=None):
        # Handle the "INFO: ..." messages as desired
        if "connection failed" in message:
            # Do something with the message, such as logging to a file
            # or displaying it in a separate widget
            print("Found a message containing 'specific_word':", message)
            if window is not None:
                try:
                    client = window.client
                    check_connection_status(client)
                except Exception as e:
                    raise
                else:
                    pass
                finally:
                    pass

        if message.startswith("INFO: "):
            # Do something with the message, such as logging to a file
            # or displaying it in a separate widget
            # pass
            messages.append(message)

        # Forward the message to the original stdout stream
        else:
            self.terminal.write(message)

def compare_images(image1, image2):

    # Ensure the images have the same dimensions
    if image1.shape != image2.shape:
        return False

    # Calculate the absolute difference between the two images
    difference = cv2.absdiff(image1, image2)

    # Sum the absolute differences to get an overall difference score
    total_difference = difference.sum()

    # Define a threshold for similarity
    # You can adjust this threshold depending on your specific needs
    similarity_threshold = 1000

    # Compare the total difference to the threshold
    return total_difference < similarity_threshold


def load_simulation_config(config_file):
    # Default configuration values
    default_config = {
        "objects": {
            "vehicle": 10,
            "pedestrian": 15,
            "cyclist": 3,
            "motorcycle": 2,
            "specific": {},  # Initialize the specific field as an empty dictionary
            "other": 15,
            # Add more object types and quantities as needed
        },
        "distribution_range": 50,
        "nb_images": 100
    }

    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        config = {}  # Empty configuration if the file doesn't exist

    # Merge the loaded configuration with the default values
    merged_config = {**default_config, **config}
    return merged_config
instance_segmentation = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8)
semantic_segmentation = np.array([[[70, 70, 70], [100, 40, 40]], [[55, 90, 80], [220, 20, 60]]], dtype=np.uint8)

#merged_image = merge_segmentation_images(instance_segmentation, semantic_segmentation)



#print(merged_image)


instance_segmentation = cv2.imread('image_1000_inst.png')
semantic_segmentation = cv2.imread('image_1000_semseg.png')

# Ensure that the images are in the correct format (e.g., uint8)
instance_segmentation = instance_segmentation.astype(np.uint8)
semantic_segmentation = semantic_segmentation.astype(np.uint8)

# Call the merge function
#merged_image = merge_segmentation_images(instance_segmentation, semantic_segmentation)

# Save the merged image to a file
#cv2.imwrite('merged_image_1000.png', merged_image)

image = cv2.imread('merged_image_1000.png')
image= image.astype(np.uint8)
bounding_boxes = find_color_regions_contours(image)
#for box in bounding_boxes:
#    print("Bounding Box:", box)
input_image_path = 'image_1000_rgb.png'  # Replace with the actual image path
output_image_path = 'output_image_1000.png' 

#draw_filtered_bounding_boxes(input_image_path, bounding_boxes, output_image_path,labels_of_interest)