import numpy as np
from queue import Queue
from collections import namedtuple
import carla
import configparser
import platform
import subprocess
import os
import psutil

def copyImageData(source_image):
    source_array = np.array(source_image.raw_data)
    #source_array = source_array.reshape((source_image.height, source_image.width, 4))

    # Create a new RGB image
    new_image = np.zeros(source_array.shape,dtype=np.uint8)#np.zeros((source_image.height, source_image.width, 3), dtype=np.uint8)

    # Copy the pixel values from source_array to new_image
    new_image[:] = source_array[:]
    #new_image[:] = source_array[:, :, :3]
    return new_image
# Sensor callback.
# This is where you receive the sensor data and
# process it as you liked and the important part is that,
# at the end, it should include an element into the sensor queue.
synchro_queue = Queue()

ImageObj = namedtuple('ImageObj', ['raw_data', 'width', 'height', 'fov'])
selected_labels=['Bicycle','Bus','Car','Motorcycle','Rider','Static','traffic_light','traffic_sign','Truck']
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
    #'Poles' : carla.CityObjectLabel.Poles,
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
def max_projected_length(length, distance,K):
        f=K[0,0]
        l_max = (f * length) / distance
        return l_max
def is_visible(bb, camera_transform,k):
    forward_vec = camera_transform.get_forward_vector()
    bb_direction = bb.location - camera_transform.location
    dist=bb.location.distance(camera_transform.location)
    bb_width=bb.extent.z 
    if max_projected_length(bb_width, dist,k)<5 :
       # print(bb)
       # print("too small")
        return False
    dot_product = (forward_vec.x) * (bb_direction.x) + forward_vec.y * (bb_direction.y) + forward_vec.z * (bb_direction.z)
    #if dot_product<=0:
       # print(bb)
       # print("behind camera")
    return dot_product>0

def expand_bb(bounding_boxes):
    for label in ["Bicycle" , "Motorcycle"]:
        if label in bounding_boxes:
            for bb in bounding_boxes[label]:
                # Expand the bounding box and append it to the modified_bbs list
                # Add a new entry for "Rider" if it doesn't exist
                if "Rider" not in bounding_boxes:
                    bounding_boxes["Rider"] = []
                # Add a new bounding box for "Rider" with the same location and dimensions
                rider_bb = carla.BoundingBox(carla.Location(bb.location), carla.Vector3D(bb.extent))
                bounding_boxes["Rider"].append(rider_bb)

def sensor_callback(world, actor, sensor_data, synchro_queue, sensor_name,K=None):
    if(sensor_name=='rgb_camera'):
        transform=actor.get_transform()
        # Calculate distances to camera for each bounding box
        bounding_boxes_with_distances = {
            label: [(bb, bb.location.distance(transform.location)) for bb in world.get_level_bbs(bb_labels[label]) if is_visible(bb, transform,K)]
            for label in selected_labels
        }
        # Sort bounding boxes by distance from camera in place
        sorted_bounding_boxes = {
            label: [bb for bb, _ in sorted(bounding_boxes_with_distances[label], key=lambda x: x[1])]
            for label in selected_labels
        }
        bounding_boxes_3d = sorted_bounding_boxes
        expand_bb(bounding_boxes_3d)
        camera_transform= carla.Transform(transform.location, transform.rotation)
        synchro_queue.put((sensor_data.frame, "bounding_boxes",bounding_boxes_3d))
        synchro_queue.put((sensor_data.frame, "camera_transform",camera_transform))

    synchro_queue.put((sensor_data.frame, sensor_name,sensor_data))

config = configparser.ConfigParser()
config.read('config.ini')
carla_path = config.get('Carla', 'path')
def check_carla_server():
    print( "Attempting to connect to Carla server .. ")
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)
        client.get_world()
        return True
    except Exception as e:
        #print(f"Failed to connect to Carla server: {e}")
        return False


def launch_carla_server():
    # launch the Carla server
    os_name = platform.system()

    print(f"Opening CARLA from path: {carla_path}")
    if os_name == 'Windows':
        path_to_run=os.path.join(carla_path, 'CarlaUE4.exe')
        subprocess.Popen(path_to_run, cwd=carla_path)
    elif os_name == 'Linux':
        path_to_run=os.path.join(carla_path, 'CarlaUE4.sh')
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