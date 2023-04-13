import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, QPushButton, QFileDialog, QLabel, QHBoxLayout, QComboBox, QGridLayout, QCheckBox, QGroupBox
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from PyQt5 import QtGui
from carla import ColorConverter
import carla
import random
import numpy as np
import cv2
import keyboard
from matplotlib import pyplot as plt
import os
from queue import Queue
from queue import Empty
from collections import namedtuple

# Sensor callback.
# This is where you receive the sensor data and
# process it as you liked and the important part is that,
# at the end, it should include an element into the sensor queue.
synchro_queue = Queue()

ImageObj = namedtuple('ImageObj', ['raw_data', 'width', 'height', 'fov'])
selected_labels=[]
bb_labels= {
    'Any' : carla.CityObjectLabel.Any,
    'Bicycle' : carla.CityObjectLabel.Bicycle,
    'Bridge' : carla.CityObjectLabel.Bridge,
    'Buildings' : carla.CityObjectLabel.Buildings,
    'Bus' : carla.CityObjectLabel.Bus,
    'Car' : carla.CityObjectLabel.Car,
    'Dynamic' : carla.CityObjectLabel.Dynamic,
    'Fences' : carla.CityObjectLabel.Fences,
    'Ground' : carla.CityObjectLabel.Ground,
    'GuardRail' : carla.CityObjectLabel.GuardRail,
    'Motorcycle' : carla.CityObjectLabel.Motorcycle,
    'NONE' : carla.CityObjectLabel.NONE,
    'Other' : carla.CityObjectLabel.Other,
    'Pedestrians' : carla.CityObjectLabel.Pedestrians,
    'Poles' : carla.CityObjectLabel.Poles,
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
    'Train' : carla.CityObjectLabel.Train,
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


# Main Window

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.x=0
         # Set up the main window
        self.setWindowTitle("MultiTrans Virtualization Framework")
        self.setGeometry(100, 100, 800, 600)
        bb_drawn = True
        self.global_imabe_error=0
        #while not bb_drawn:
        # Initialize the CARLA client and world
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(30.0)
        self.client.load_world('Town04')
        self.world = self.client.get_world()
        #☺self.context= self.world.get_snapshot()
        self.camera_tick = 0
        self.ThreeD = False
        #self.semantic_image= None
        # We create all the sensors and keep them in a list for convenience.
        self.synchro_list = []
        # Adjust the graphics settings
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True
        settings.no_rendering_mode = False
        settings.quality_level = 'Ultra'
        settings.resolution = (1920, 1080)
        settings.anti_aliasing = '16x'
        settings.shadow_quality = 'Epic'
        settings.particles_quality_level = 'High'
        self.world.apply_settings(settings)
        # initiate bounding_box_labels
        self.selected_labels=selected_labels
        self.bb_labels= bb_labels
        group_box = QGroupBox('City Object Labels')
        # Create the checkboxes and add them to the group box
        checkbox_layout =  QGridLayout()
        for i, (label, value) in enumerate(bb_labels.items()):
            checkbox = QCheckBox(label)
            checkbox.setChecked(False)
            checkbox.stateChanged.connect(lambda state, label=label: self.checkbox_state_changed(state, label))
            row = i // 4  # display 4 checkboxes per row
            col = i % 4
            checkbox_layout.addWidget(checkbox, row, col)
        group_box.setLayout(checkbox_layout)
        # Set the weather parameters
        weather = carla.WeatherParameters(
            cloudiness=random.uniform(0.0, 50.0),
            precipitation=random.uniform(-50, 0),
            precipitation_deposits=random.uniform(0.0, 50.0),
            wind_intensity=random.uniform(0.0, 50.0),
            sun_azimuth_angle=random.uniform(45.0, 135.0),
            sun_altitude_angle=random.uniform(45.0, 145.0),
            fog_density=random.uniform(0.0, 25.0),
            fog_distance=random.uniform(0.0, 200.0),
            fog_falloff=random.uniform(0.0, 200.0),
            wetness=random.uniform(0.0, 50.0),
            #puddles=random.uniform(0.0, 50.0),
            scattering_intensity=random.uniform(0.0, 50.0), 
            mie_scattering_scale=random.uniform(0.0, 50.0), 
            rayleigh_scattering_scale=0.03310000151395798, 
            dust_storm=random.uniform(0.0, 25.0),
            #snow_depth=random.uniform(0.0, 50.0),
            #ice_adherence=random.uniform(0.0, 50.0),
            #precipitation_type=carla.WeatherParameters.PrecipitationType.Snow
            #is_wet=True

        )

        self.label_rgb = QLabel(self)
        self.label_seg = QLabel(self)
        self.label_bounding = QLabel(self)
        
        # Set up the palette of editing a scenario
        add_btn = QPushButton(QIcon("icons/add.png"), "")
        select_btn = QPushButton(QIcon("icons/select.png"), "")
        remove_btn = QPushButton(QIcon("icons/remove.png"), "")
        modify_btn = QPushButton(QIcon("icons/modify.png"), "")
        timeline_btn = QPushButton(QIcon("icons/timeline.png"), "")
        new_event_btn = QPushButton(QIcon("icons/new_event.png"), "")
        palette_layout = QHBoxLayout()
        palette_layout.addWidget(add_btn)
        palette_layout.addWidget(select_btn)
        palette_layout.addWidget(remove_btn)
        palette_layout.addWidget(modify_btn)
        palette_layout.addWidget(timeline_btn)
        palette_layout.addWidget(new_event_btn)
        palette_widget = QWidget()
        palette_widget.setLayout(palette_layout)

        # Set up the menu bar
        menu_bar = self.menuBar()
        scenario_menu = menu_bar.addMenu("Scenario")
        start_action = scenario_menu.addAction("Start")
        stop_action = scenario_menu.addAction("Stop")
        pause_action = scenario_menu.addAction("Pause")
        record_action = scenario_menu.addAction("Record")

        # Connect the menu bar actions to functions
        start_action.triggered.connect(self.start_scenario)
        stop_action.triggered.connect(self.stop_scenario)
        pause_action.triggered.connect(self.pause_scenario)
        record_action.triggered.connect(self.record_scenario)

        # Set up the layout for the main window
        central_widget = QWidget()
        layout = QGridLayout()
        layout.addWidget(self.label_rgb, 0, 0)
        layout.addWidget(self.label_seg, 0, 1)
        layout.addWidget(self.label_bounding, 0, 2)
        layout.addWidget(palette_widget, 1, 0, 1, 3)
        layout.addWidget(group_box)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)



        self.K = self.build_projection_matrix(1280, 1024, 90)

        vehicle_bp = random.choice(self.world.get_blueprint_library().filter('vehicle*'))
        #carla.Transform(carla.Location(x=35,y=35,z=0))

        points=self.world.get_map().get_spawn_points()
        
        
        vehicle_transform =random.choice(self.world.get_map().get_spawn_points()) #carla.Transform(carla.Location(x=-64.644844, y=24.471010, z=0.600000))
        
        self.vehicle = self.world.spawn_actor(vehicle_bp, vehicle_transform)

        self.rgb_camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.rgb_camera_bp.set_attribute('image_size_x', '1280')
        self.rgb_camera_bp.set_attribute('image_size_y', '1024')
        self.rgb_camera_bp.set_attribute('fov', '90')
        self.rgb_camera_transform = carla.Transform(carla.Location(x=2, z=1.5, y=0))

        self.rgb_ref_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.rgb_ref_bp.set_attribute('image_size_x', '1280')
        self.rgb_ref_bp.set_attribute('image_size_y', '1024')
        self.rgb_ref_bp.set_attribute('fov', '90')
        self.rgb_ref_transform=  self.rgb_camera_transform

        self.seg_camera_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        self.seg_camera_bp.set_attribute('image_size_x', '1280')
        self.seg_camera_bp.set_attribute('image_size_y', '1024')
        self.seg_camera_bp.set_attribute('fov', '90')
        self.seg_camera_bp.set_attribute('sensor_tick', '0.0')
        self.seg_camera_transform = self.rgb_camera_transform


        self.rgb_camera = self.world.spawn_actor(
            self.rgb_camera_bp,
            self.rgb_camera_transform,
            attach_to=self.vehicle
        )

        self.rgb_camera_ref = self.world.spawn_actor(
            self.rgb_ref_bp,
            self.rgb_ref_transform,
            attach_to=self.vehicle
        )

        self.seg_camera = self.world.spawn_actor(
            self.seg_camera_bp,
            self.seg_camera_transform,
            attach_to=self.vehicle
        )
        self.vehicle.set_autopilot(False)

        self.synchro_list.append(self.rgb_camera)
        self.synchro_list.append(self.rgb_camera_ref)
        self.synchro_list.append(self.seg_camera)
        self.synchro_list.append("bounding_boxes")
        self.synchro_list.append("camera_transform")

        # Start the timer for updating the real-time views
        self.timer = QTimer()
        #self.timer.timeout.connect(self.update_views)
        self.timer.start(50)
        self.rgb_camera.listen(lambda data: sensor_callback(self.world,self.rgb_camera, data,synchro_queue,"rgb_camera" , self.K))
        self.rgb_camera_ref.listen(lambda data: sensor_callback(self.world,self.rgb_camera_ref, data,synchro_queue,"rgb_camera_ref",self.K))
        #self.rgb_camera.listen(lambda data: self.update_bounding_box_view(data))
        self.seg_camera.listen(lambda data: sensor_callback(self.world,self.seg_camera, data,synchro_queue,"semantic_segmentation",self.K))
        self.timer = self.startTimer(100)
        self.running = True

        new_vehicle_bp = random.choice(self.world.get_blueprint_library().filter('vehicle*'))
        new_vehicle_location = self.vehicle.get_transform().location+ carla.Location(3,0,0)#carla.Transform(carla.Location(x=-67.254570, y=27.963758, z=0.600000))#
        new_vehicle_transform =random.choice(self.world.get_map().get_spawn_points()) # carla.Transform(carla.Location(x=-67.254570, y=27.963758, z=0.600000))#carla.Transform(new_vehicle_location, vehicle_transform.rotation)
        self.new_vehicle = self.world.spawn_actor(new_vehicle_bp, new_vehicle_transform)
        self.world.tick()
        self.new_vehicle.set_transform(carla.Transform(new_vehicle_location))


    def start_scenario(self):
        pass
    def stop_scenario(self):
        pass
    def pause_scenario(self):
        pass
    def record_scenario(self):
        pass
    def synchroTick(self):
        if(self.camera_tick == 3):
            self.world.tick()
            self.camera_tick= 0;
            self.vehicle.set_autopilot(False)

    def checkbox_state_changed(self, state, label):
        # Update the value of the corresponding label in the dictionary
        if state == 2:
            self.selected_labels.append(label)
        elif state == 0:
            self.selected_labels.remove(label)
        print(self.selected_labels)

    def timerEvent(self, event):

        self.world.tick()
        w_frame = self.world.get_snapshot().frame
        rgb_image=None
        segmentation_image=None
        boxes=None
        transform =None
        rgb_ref=None
        self.x+=1
        new_vehicle_location = self.vehicle.get_transform().location+ carla.Location(100-self.x,+4,0)
        #self.new_vehicle.set_transform(carla.Transform(new_vehicle_location, self.vehicle.get_transform().rotation))
        try:
            
            
            #print("Try")
            #print("sensors")
            #print(self.synchro_list)
            for _ in range(len(self.synchro_list)):
                s_frame = synchro_queue.get(True, 10.0)
                #print("    Frame: %d   Sensor: %s" % (s_frame[0], s_frame[1]))
                if s_frame[1] == "rgb_camera":
                    rgb_image=self.get_RGB_DATA(s_frame[2])
                if s_frame[1] == "rgb_camera_ref":
                    rgb_ref=self.update_rgb_camera_view(s_frame[2])
                if s_frame[1] == "semantic_segmentation":
                    segmentation_image=self.update_seg_camera_view(s_frame[2])
                if s_frame[1] == "bounding_boxes":
                    boxes= s_frame[2]
                if s_frame[1] == "camera_transform":
                    transform= s_frame[2]
                #print(s_frame[1] )
                #print(s_frame[2] )
            if rgb_image!=None and boxes!=None and transform!=None :
                #print(rgb_image.width)
                if self.ThreeD:
                    self.update_bounding_box_view_3D(self.rgb_camera,rgb_image,selected_labels)
                else:
                    self.update_bounding_box_view_smart(self.rgb_camera,rgb_image,segmentation_image,boxes,transform)
            else:
                print("at least one of these data is missing:")
                if rgb_image==None:
                    print('rgb')
                if segmentation_image==None:
                    print('segmentation_image')
                if boxes==None:
                    print('boxes')
                if transform==None:
                    print('transform')

        except Empty:
                pass
        self.vehicle.set_autopilot(True)

    def start_stop_camera(self):
        # Start or stop the camera movement
        if self.running:
            self.camera.stop()
            self.button.setText('Start')
        else:
            self.camera.listen(lambda data: self.update_camera_view(data))
            self.button.setText('Stop')
        self.running = not self.running

    def get_RGB_DATA(self,image):
        new_raw_data = image.raw_data#bytearray(image.raw_data)
        new_image = ImageObj(raw_data=new_raw_data, width=image.width, height=image.height,fov=image.fov)
        return new_image

    def update_rgb_camera_view(self, image):
        # Convert the Image object to a QImage object
        #new_raw_data = image.raw_data#bytearray(image.raw_data)
        #new_image = ImageObj(raw_data=new_raw_data, width=image.width, height=image.height,fov=image.fov)

        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        qimage = QtGui.QImage(array.data, image.width, image.height, QtGui.QImage.Format_RGB32)
        pixmap = QtGui.QPixmap.fromImage(qimage)
        self.label_rgb.setPixmap(pixmap)
        actors = self.world.get_actors()
        objects = self.selected_labels
        return image

    def build_projection_matrix(self, w, h, fov):
        focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = w / 2.0
        K[1, 2] = h / 2.0
        return K

    

    def get_image_point(self, loc, K, w2c):
        # Calculate 2D projection of 3D coordinate

        # Format the input coordinate (loc is a carla.Position object)
        point = np.array([loc.x, loc.y, loc.z, 1])
        # transform to camera coordinates
        point_camera = np.dot(w2c, point)

        # New we must change from UE4's coordinate system to an "standard"
        # (x, y ,z) -> (y, -z, x)
        # and we remove the fourth componebonent also
        #point_camera = [point_camera[0], -point_camera[1], point_camera[2]]
        point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

        # now project 3D->2D using the camera matrix
        point_img = np.dot(K, point_camera)
        # normalize
        point_img[0] /= point_img[2]
        point_img[1] /= point_img[2]

        return point_img[0:2]

    def no_tighten_bb(self,sem_seg_image, x_min, y_min, x_max, y_max, object_color):
        return x_min, y_min, x_max, y_max

    def tighten_bb(self,sem_seg_image, x_min, y_min, x_max, y_max, object_color):
        #if x_min<=0:
        #    print(f"x_min: {x_min} -> image coordinates: {np.max(x_min-1,0)}")
        #    x_min=0
        #if y_min<=0:
        #    print(f"y_min: {x_min} -> image coordinates: {np.max(x_min-1,0)}")
        #    y_min=0

        
        if sem_seg_image==None:
            #print("sem_seg_image is not valid")
            return -1,-1,-1,-1
        np_img = np.frombuffer(sem_seg_image.raw_data, dtype=np.dtype("uint8"))
        np_img = np_img.reshape((sem_seg_image.height, sem_seg_image.width, 4))
        np_img = np_img[..., :3]
        object_mask = cv2.inRange(np_img, object_color, object_color)
        # Crop the object mask to the bounding box
        object_mask_bb = object_mask[y_min:y_max, x_min:x_max]
        # Find the indices of all pixels that belong to the object
        object_pixels_indices = np.argwhere(object_mask_bb != 0)

        if len(object_pixels_indices) == 0:
            self.global_imabe_error+=1
            return -1, -1, -1, -1

        x_min_new = x_min + np.min(object_pixels_indices[:, 1])
        y_min_new = y_min + np.min(object_pixels_indices[:, 0])
        x_max_new = x_min + np.max(object_pixels_indices[:, 1])
        y_max_new = y_min + np.max(object_pixels_indices[:, 0])

        if x_max_new<0 or y_max_new<0 or x_min_new>sem_seg_image.width or y_min_new> sem_seg_image.height:
            return -1, -1, -1, -1

        #if x_min_new<0:
        #    x_min_new=0
        #if y_min_new<0:
        #    y_min_new=0
        #if x_max_new>sem_seg_image.width:
        #    x_max_new=sem_seg_image.width
        #if y_max_new>sem_seg_image.height:
        #    y_max_new=sem_seg_image.height

        return x_min_new, y_min_new, x_max_new, y_max_new

    def update_bounding_box_view_smart(self, camera, image, semantic_image, bounding_box_set, transform):
        world_2_camera = np.array(transform.get_inverse_matrix())
        #print("world_2_camera")
        edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]
        img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        #print("frombuffer")
        img = np.reshape(img, (image.height, image.width, 4))
        #print("reshape")
        #ego_bb = self.vehicle.bounding_box
        processed_boxes = {label: [] for label in bounding_box_set.keys()}
        for label, bb_set in bounding_box_set.items():
            for bb in bb_set:
                corners = bb.get_world_vertices(carla.Transform())
                corners = [self.get_image_point(corner, self.K, world_2_camera) for corner in corners]

                corners = np.array(corners, dtype=int)
                x_min, y_min = np.min(corners, axis=0)
                if x_min<0:
                    x_min=0
                if y_min <0:
                    y_min=0
                x_max, y_max = np.max(corners, axis=0)

                if label in ['Bicycle','Motorcycle','Rider','Pedestrians']:
                    #w_max = (f * (x_max - x_min)) / z
                    w,h=self.bb_max_dim[label]
                    yaw=bb.rotation.yaw
                    pitch=bb.rotation.pitch
                    w_proj = w * np.abs(np.cos(yaw))
                    h_proj = h * np.abs(np.sin(pitch))
                    z= bb.location.distance(transform.location)
                    bb_width= int(max_projected_length(w_proj,z,self.K)/2)
                    bb_h = int(max_projected_length(h_proj,z,self.K)/2)
                    x_min=x_min - bb_width
                    x_max=x_max + bb_width

                    y_min=y_min- bb_h
                    y_max=y_max+ bb_h

                # Check if the bounding box is completely included in any previously processed box
                included = False
                for processed_bb in processed_boxes[label]:
                    if x_min >= processed_bb[0] and x_max <= processed_bb[2] and y_min >= processed_bb[1] and y_max <= processed_bb[3]:
                        included = True
                        break

                if not included:
                    # Process the bounding box if it's not included in any previously processed box
                    processed_boxes[label].append((x_min, y_min, x_max, y_max))
                    # Check if bounding box is inside image dimensions
                    if x_min >= img.shape[1] or x_max < 0. or y_min >= img.shape[0] or y_max < 0:
                        continue
                    object_color=self.CLASS_MAPPING[label]
                    object_color=(object_color[2], object_color[1], object_color[0])
                    x_min_new , y_min_new, x_max_new, y_max_new = self.tighten_bb(semantic_image, x_min, y_min, x_max, y_max, object_color )
                    if not(x_min_new==-1 and y_min_new ==-1 and x_max_new==-1 and y_max_new== -1):

                        # Draw bounding box using OpenCV's rectangle function
                        cv2.rectangle(img, (x_min_new, y_min_new), (x_max_new, y_max_new), object_color, 2)

                        #label = 'vehicle'  # replace with the appropriate label for each object type
                        cv2.putText(img, label, (x_min_new, y_min_new-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, object_color, 1)

        # Convert the image back to a QImage object and display it
        qimage = QtGui.QImage(img.data, image.width, image.height, QtGui.QImage.Format_RGB32)
        pixmap = QtGui.QPixmap.fromImage(qimage)
        self.label_bounding.setPixmap(pixmap)

    def update_bounding_box_view_3D(self, camera, image, objects):
        # Convert the Image object to a QImage object
         # Define percentage to reduce bounding box size by
        reduction_percentage = 0.1

        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
        edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]
        img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        img = np.reshape(img, (image.height, image.width, 4))
        #carla.CityObjectLabel.Car
        for label in objects:
            bounding_box_set = self.world.get_level_bbs(self.bb_labels[label])
            c_objects = self.world.get_environment_objects(self.bb_labels[label])
            
            ego_bb = self.vehicle.bounding_box
            # Filter bounding boxes based on distance from camera
            bounding_box_set = [bb for bb in bounding_box_set if (bb.location.distance(camera.get_location()) < 50) and (not  bb == ego_bb) and (bb.location.distance(camera.get_location()) >2 )]
        
            for bb in bounding_box_set:
                # Check if the bounding box is visible to the camera
                #line_of_sight = self.world.get_line_of_sight(camera.get_location(), bb.location)
                forward_vec = self.vehicle.get_transform().get_forward_vector()
                bb_direction = bb.location - camera.get_transform().location
                dot_product = forward_vec.x * bb_direction.x + forward_vec.y * bb_direction.y + forward_vec.z * bb_direction.z
                if dot_product > 0:
                #if np.dot(forward_vec, bb_direction) > 0:
                #relative_location = bb.location - camera.get_location()
                #ang=camera.get_forward_vector().get_angle(relative_location)
                #if ang < 80 and ang>=0:
                # Define percentage to reduce bounding box size by
                    verts = [v for v in bb.get_world_vertices(carla.Transform())]
                    center_point = carla.Location()
                    for v in verts:
                        center_point += v
                    center_point /= len(verts)

                    # Calculate new vertices by reducing distance from center point by reduction percentage
                    new_verts = []
                    for v in verts:
                        direction = v - center_point
                        new_direction = direction - direction * reduction_percentage
                        new_v = center_point + new_direction
                        new_verts.append(new_v)

                    # Get image coordinates of new vertices
                    corners = [self.get_image_point(corner, self.K, world_2_camera) for corner in new_verts]

                    # Use NumPy to calculate min/max corners
                    corners = np.array(corners, dtype=int)
                    x_min, y_min = np.min(corners, axis=0).astype(int)

                    # Draw edges of the bounding box into the camera output
                    for edge in edges:
                        p1 = self.get_image_point(new_verts[edge[0]], self.K, world_2_camera)
                        p2 = self.get_image_point(new_verts[edge[1]], self.K, world_2_camera)
                        # Draw the edges into the camera output
                        cv2.line(img, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (0,0,255, 255), 1)

                        #label = 'vehicle'  # replace with the appropriate label for each object type
                    cv2.putText(img, label, (x_min, y_min -5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Convert the image back to a QImage object and display it
        qimage = QtGui.QImage(img.data, image.width, image.height, QtGui.QImage.Format_RGB32)
        #pixmap = QtGui.QPixmap.fromImage(qimage)

        # Convert the QImage object to a QPixmap object and display it
        pixmap = QtGui.QPixmap.fromImage(qimage)
        self.label_bounding.setPixmap(pixmap)
        self.camera_tick += 1
        self.synchroTick()

   
    def update_seg_camera_view(self, image):
        image.convert(carla.ColorConverter.CityScapesPalette)
        np_img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        np_img = np_img.reshape((image.height, image.width, 4))
        np_img = np_img[..., :3]  # Remove the alpha channel
        #qimage = QtGui.QImage(bytes(np_img.data), image.width, image.height, QtGui.QImage.Format_RGB888)
        #qimage = qimage.rgbSwapped()
        #self.semantic_image= image
    # Use the palette to create a QImage
        qimage = QtGui.QImage(bytes(np_img.data), image.width, image.height, QtGui.QImage.Format_RGB888).rgbSwapped()
        #qimage = qimage.rgbSwapped()
        #self.semsegConversion(qimage)
        #qimage.setColorTable(palette_list)
        pixmap = QtGui.QPixmap.fromImage(qimage)
        self.label_seg.setPixmap(pixmap)
        return image
        #self.camera_tick +=1
        #self.synchroTick()

    CLASS_MAPPING = {
        'unlabeled': (0, 0, 0),
        'Buildings': (70, 70, 70),
        'Fences': (100, 40, 40),
        'Other': (55, 90, 80),
        'Pedestrians': (220, 20, 60),
        'Rider': (255, 0, 0),
        'pole': (153, 153, 153),
        'RoadLines': (157, 234, 50),
        'Roads': (128, 64, 128),
        'Sidewalks': (244, 35, 232),
        'Vegetation': (107, 142, 35),
        'Bicycle' : (119,  11,  32),
        'Bus': (0, 60, 100),
        'Car': (0, 0, 142),
        'Truck': (0, 0, 70),
        'Motorcycle': (0, 0, 230),
        'vehicle': (0, 0, 142),
        'Walls': (102, 102, 156),
        'traffic_sign': (220, 220, 0),
        'Sky': (70, 130, 180),
        'ground': (81, 0, 81),
        'Bridge': (150, 100, 100),
        'RailTrack': (230, 150, 140),
        'guardrail': (180, 165, 180),
        'traffic_light': (250, 170, 30),
        'Static': (110, 190, 160),
        'Dynamic': (170, 120, 50),
        'water': (45, 60, 150),
        'terrain': (145, 170, 100)
    }
    bb_max_dim = {
        'Pedestrians' : (1,2.5),
        'Bicycle' : (4,2),
        'Motorcycle' : (4,2),
        'Rider' : (4,2.5)

    }

    def semsegConversion(self,qimage):
        # Define a dictionary that maps label colors to class names
        label_color_map = {
            (0, 0, 0): 'Unlabeled',
            (70, 70, 70): 'Building',
            (190, 153, 153): 'Fence',
            (250, 170, 160): 'Other',
            (220, 220, 0): 'Pedestrian',
            (107, 142, 35): 'Vegetation',
            (152, 251, 152): 'Terrain',
            (70, 130, 180): 'Sky',
            (220, 20, 60): 'Car',
            (255, 0, 0): 'Traffic Sign',
            (0, 0, 142): 'Traffic Light'
        }

        # Create a list of color tuples from the dictionary
        palette_list = [(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)) for color in label_color_map.keys()]
        palette_list = [color[0] << 16 | color[1] << 8 | color[2] for color in palette_list]


        #palette_list = [color[0] << 16 | color[1] << 8 | color[2] for color in label_color_map.keys()]

        # Create a color table from the list of color tuples
        qimage.setColorTable(palette_list)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    k=main_window.K
    print(k)
    camera_location= main_window.rgb_camera.get_transform().location
    w2c=np.array(main_window.rgb_camera.get_transform().get_inverse_matrix())
    p1 = camera_location+ carla.Location(5, 0, 0)
    p2 = camera_location+ carla.Location(0, 5, 0)
    p3 = camera_location+ carla.Location(5, 0, 5)
    p4 = camera_location+ carla.Location(5, 5, 0)
    p5 = camera_location+ carla.Location(5, 0, 5)
    p6 = camera_location+ carla.Location(5, 5, 5)
    p7 = camera_location+ carla.Location(-5, -5, 0)
    p8 = camera_location+ carla.Location(-5, -5, -5)
    # Test the function with the in-view point
    point_img = main_window.get_image_point(p1, k, w2c)
    print(f"point: {p1} -> image coordinates: {point_img}")
    point_img = main_window.get_image_point(p2, k, w2c)
    print(f"point: {p2} -> image coordinates: {point_img}")
    point_img = main_window.get_image_point(p3, k, w2c)
    print(f"point: {p3} -> image coordinates: {point_img}")
    point_img = main_window.get_image_point(p4, k, w2c)
    print(f"point: {p4} -> image coordinates: {point_img}")
    point_img = main_window.get_image_point(p5, k, w2c)
    print(f"point: {p5} -> image coordinates: {point_img}")
    point_img = main_window.get_image_point(p6, k, w2c)
    print(f"point: {p6} -> image coordinates: {point_img}")
    point_img = main_window.get_image_point(p7, k, w2c)
    print(f"point: {p7} -> image coordinates: {point_img}")
    point_img = main_window.get_image_point(p8, k, w2c)
    print(f"point: {p8} -> image coordinates: {point_img}")

    

    # Define a test 3D point out of the camera's field of view
    point_out_of_view = carla.Location(-20, 10, 50)

    # Test the function with the out-of-view point
    point_img = main_window.get_image_point(point_out_of_view, k, w2c)
    print(f"Out-of-view point: {point_out_of_view} -> image coordinates: {point_img}")
    main_window.show()
    sys.exit(app.exec_())








