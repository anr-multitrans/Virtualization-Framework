import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QHBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5 import QtGui
from carla import ColorConverter
import carla
import random
import numpy as np
import cv2

class CarlaUI(QWidget):
    def __init__(self):
        super().__init__()

        # Initialize the Carla client and world
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.camera_tick = 0

        # Adjust the graphics settings
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 0.05
#        settings.synchronous_mode = True
        settings.no_rendering_mode = False
        settings.quality_level = 'Ultra'
        settings.resolution = (1920, 1080)
        settings.anti_aliasing = '16x'
        settings.shadow_quality = 'Epic'
        settings.particles_quality_level = 'High'
        self.world.apply_settings(settings)
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
        #self.world.set_weather(weather)
        self.K = self.build_projection_matrix(1280, 1024, 90)
        # Create a vehicle
        vehicle_bp = random.choice(self.world.get_blueprint_library().filter('vehicle.dodge.*'))
        #vehicle.dodge.charger_2020
        vehicle_transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(vehicle_bp, vehicle_transform)

        # Attach the camera sensor to the vehicle
        self.rgb_camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.rgb_camera_bp.set_attribute('image_size_x', '1280')
        self.rgb_camera_bp.set_attribute('image_size_y', '1024')
        self.rgb_camera_bp.set_attribute('fov', '90')
        self.rgb_camera_transform = carla.Transform(carla.Location(x=1.5, z=1.4, y=0))
        self.rgb_camera_transform.rotation.yaw = +5.0
        #self.rgb_camera_transform.rotation.pitch = -25.0

        self.rgb_camera = self.world.spawn_actor(
            self.rgb_camera_bp,
            self.rgb_camera_transform,
            attach_to=self.vehicle
        )

        # Attach the semantic segmentation camera sensor to the vehicle
        self.seg_camera_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        self.seg_camera_bp.set_attribute('image_size_x', '1280')
        self.seg_camera_bp.set_attribute('image_size_y', '1024')
        self.seg_camera_bp.set_attribute('fov', '90')
        self.seg_camera_bp.set_attribute('sensor_tick', '0.0')
        #self.seg_camera_bp.set_attribute('post_processing', 'SemanticSegmentation')        
        #self.seg_camera_bp.set_attribute('semantic_segmentation', 'CityScapesPalette')
        self.seg_camera_transform = carla.Transform(carla.Location(x=1.5, z=1.4, y=0))
        self.seg_camera_transform.rotation.yaw = +5.0
        #self.seg_camera_transform.rotation.pitch = +15.0


        self.seg_camera = self.world.spawn_actor(
            self.seg_camera_bp,
            self.seg_camera_transform,
            attach_to=self.vehicle
        )
        self.vehicle.set_autopilot(True)

        # Set up the user interface
        self.label_rgb = QLabel(self)
        self.label_rgb.setFixedSize(640, 480)
        self.label_seg = QLabel(self)
        self.label_seg.setFixedSize(640, 480)
        self.label_bounding = QLabel(self)
        self.label_bounding.setFixedSize(640, 480)

        layout = QHBoxLayout(self)
        layout.addWidget(self.label_rgb)
        layout.addWidget(self.label_seg)
        layout.addWidget(self.label_bounding)
        self.setLayout(layout)

        # Start the main loop
        self.rgb_camera.listen(lambda data: self.update_rgb_camera_view(data))
        #self.rgb_camera.listen(lambda data: self.update_bounding_box_view(data))
        self.seg_camera.listen(lambda data: self.update_seg_camera_view(data))
        self.timer = self.startTimer(100)
        self.running = True
        self.world.wait_for_tick()
        if(self.camera_tick == 3):
            self.world.tick()
            self.camera_tick= 0;

        #for i in range(200):
        #    self.world.tick()

    def timerEvent(self, event):
        pass
        #control = self.vehicle.get_control()
        #control.throttle = 0.5
        #control.steer = 0.1
        #self.vehicle.apply_control(control)
        #self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))

    def start_stop_camera(self):
        # Start or stop the camera movement
        if self.running:
            self.camera.stop()
            self.button.setText('Start')
        else:
            self.camera.listen(lambda data: self.update_camera_view(data))
            self.button.setText('Stop')
        self.running = not self.running

    def update_rgb_camera_view(self, image):
        # Convert the Image object to a QImage object
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        qimage = QtGui.QImage(array.data, image.width, image.height, QtGui.QImage.Format_RGB32)
        pixmap = QtGui.QPixmap.fromImage(qimage)
        self.label_rgb.setPixmap(pixmap)
        actors = self.world.get_actors()
        # Filter the list to get only objects with a bounding box
        objects = [actor for actor in actors if actor.type_id.startswith('vehicle') or actor.type_id.startswith('walker')]
        self.update_bounding_box_view(self.rgb_camera,image,objects)
        self.camera_tick +=1

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

    def update_bounding_box_view_1(self, camera, image, objects):
        # Convert the Image object to a QImage object
        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
        
        img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        img = np.reshape(img, (image.height, image.width, 4))
        #img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4)) 
        bounding_box_set = self.world.get_level_bbs(carla.CityObjectLabel.TrafficLight)
        bounding_box_set.extend(self.world.get_level_bbs(carla.CityObjectLabel.TrafficSigns))
        #bounding_box_set.extend(self.world.get_actors())
        for npc in self.world.get_actors():#.filter('*vehicle*'):
            if npc.id != self.vehicle.id:
                bb = npc.bounding_box
                dist = npc.get_transform().location.distance(self.vehicle.get_transform().location)

                # Filter for the vehicles within 50m
                if dist < 50:

                # Calculate the dot product between the forward vector
                # of the vehicle and the vector between the vehicle
                # and the other vehicle. We threshold this dot product
                # to limit to drawing bounding boxes IN FRONT OF THE CAMERA
                    forward_vec = self.vehicle.get_transform().get_forward_vector()
                    ray = npc.get_transform().location - self.vehicle.get_transform().location
    
                    if forward_vec.dot(ray) > 1:
                        p1 = self.get_image_point(bb.location, self.K, world_2_camera)#http://host.robots.ox.ac.uk/pascal/VOC/
                        verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                        x_max = -10000
                        x_min = 10000
                        y_max = -10000
                        y_min = 10000
    
                        for vert in verts:
                            p = self.get_image_point(vert, self.K, world_2_camera)
                            # Find the rightmost vertex
                            if p[0] > x_max:
                                x_max = p[0]
                            # Find the leftmost vertex
                            if p[0] < x_min:
                                x_min = p[0]
                            # Find the highest vertex
                            if p[1] > y_max:
                                y_max = p[1]
                            # Find the lowest  vertex
                            if p[1] < y_min:
                                y_min = p[1]
    
                        cv2.line(img, (int(x_min),int(y_min)), (int(x_max),int(y_min)), (0,0,255, 255), 1)
                        cv2.line(img, (int(x_min),int(y_max)), (int(x_max),int(y_max)), (0,0,255, 255), 1)
                        cv2.line(img, (int(x_min),int(y_min)), (int(x_min),int(y_max)), (0,0,255, 255), 1)
                        cv2.line(img, (int(x_max),int(y_min)), (int(x_max),int(y_max)), (0,0,255, 255), 1)
        #looking for other non-actor vehicles
        bounding_box_set.extend(self.world.get_level_bbs(carla.CityObjectLabel.Vehicles))
        for bb in bounding_box_set:
        # Filter for distance from ego vehicle
            if bb.location.distance(self.vehicle.get_transform().location) < 50:
                forward_vec = self.vehicle.get_transform().get_forward_vector()
                ray = bb.location - vehicle.get_transform().location

                if forward_vec.dot(ray) > 1:
                    p1 = get_image_point(bb.location, K, world_2_camera)#http://host.robots.ox.ac.uk/pascal/VOC/
                    verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                    x_max = -10000
                    x_min = 10000
                    y_max = -10000
                    y_min = 10000

                    for vert in verts:
                        p = self.get_image_point(vert, K, world_2_camera)
                        # Find the rightmost vertex
                        if p[0] > x_max:
                            x_max = p[0]
                        # Find the leftmost vertex
                        if p[0] < x_min:
                            x_min = p[0]
                        # Find the highest vertex
                        if p[1] > y_max:
                            y_max = p[1]
                        # Find the lowest  vertex
                        if p[1] < y_min:
                            y_min = p[1]

                    cv2.line(img, (int(x_min),int(y_min)), (int(x_max),int(y_min)), (0,0,255, 255), 1)
                    cv2.line(img, (int(x_min),int(y_max)), (int(x_max),int(y_max)), (0,0,255, 255), 1)
                    cv2.line(img, (int(x_min),int(y_min)), (int(x_min),int(y_max)), (0,0,255, 255), 1)
                    cv2.line(img, (int(x_max),int(y_min)), (int(x_max),int(y_max)), (0,0,255, 255), 1)






        # Draw the bounding boxes of objects in the image
        
        qimage = QtGui.QImage(img.data, image.width, image.height, QtGui.QImage.Format_RGB32)
        #pixmap = QtGui.QPixmap.fromImage(qimage)

        # Convert the QImage object to a QPixmap object and display it
        pixmap = QtGui.QPixmap.fromImage(qimage)
        self.label_bounding.setPixmap(pixmap)
        self.camera_tick +=1

    def update_bounding_box_view_2(self, camera, image, objects):
        # Convert the Image object to a QImage object
        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
        edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]
        img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        img = np.reshape(img, (image.height, image.width, 4))
        bounding_box_set = self.world.get_level_bbs(carla.CityObjectLabel.Car)
        #bounding_box_set.extend(self.world.get_level_bbs(carla.CityObjectLabel.TrafficSigns))
        #self.world.get_level_bbs(carla.CityObjectLabel.Car)
        # Check if vehicle is visible in camera view before drawing its bounding box
        for npc in self.world.get_actors().filter('vehicle5645446*'):
            if npc.id != self.vehicle.id:
                bb = npc.bounding_box
                dist = npc.get_transform().location.distance(self.vehicle.get_transform().location)
    
            # Filter for the vehicles within 50m
                if dist < 50:
    
                # Calculate the dot product between the forward vector
                    # of the vehicle and the vector between the vehicle
                    # and the other vehicle. We threshold this dot product
                    # to limit to drawing bounding boxes IN FRONT OF THE CAMERA
                    forward_vec = self.vehicle.get_transform().get_forward_vector()
                    ray = npc.get_transform().location - self.vehicle.get_transform().location
    
                    if forward_vec.dot(ray) > 1:
                        corners = bb.get_world_vertices(npc.get_transform())
                        corners = [self.get_image_point(corner, self.K, world_2_camera) for corner in corners]
    
                        # Check if any corner is outside image dimensions
                        if all(0 <= corner[0] < img.shape[1] and 0 <= corner[1] < img.shape[0] for corner in corners):
                            x_min, y_min = np.min(corners, axis=0).astype(int)
                            x_max, y_max = np.max(corners, axis=0).astype(int)
    
                            cv2.line(img, (x_min, y_min), (x_max, y_min), (0, 0, 255, 255), 1)
                            cv2.line(img, (x_min, y_max), (x_max, y_max), (0, 0, 255, 255), 1)
                            cv2.line(img, (x_min, y_min), (x_min, y_max), (0, 0, 255, 255), 1)
                            cv2.line(img, (x_max, y_min), (x_max, y_max), (0, 0, 255, 255), 1)
                            label = 'vehicle'  # replace with the appropriate label for each object type
                            cv2.putText(img, label, (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255, 255), 1)

        #looking for other non-actor vehicles
        #bounding_box_set.extend(self.world.get_level_bbs(carla.CityObjectLabel.Car))
        for bb in bounding_box_set:

        # Filter for distance from ego vehicle
            if bb.location.distance(self.vehicle.get_transform().location) < 50:
    
                # Calculate the dot product between the forward vector
                # of the vehicle and the vector between the vehicle
                # and the bounding box. We threshold this dot product
                # to limit to drawing bounding boxes IN FRONT OF THE CAMERA
                forward_vec = self.vehicle.get_transform().get_forward_vector()
                ray = bb.location - self.vehicle.get_transform().location

                if forward_vec.dot(ray) > 1:
                        corners = bb.get_world_vertices(carla.Transform())
                        corners = [self.get_image_point(corner, self.K, world_2_camera) for corner in corners]
    
                        # Check if any corner is outside image dimensions
                        if all(0 <= corner[0] < img.shape[1] and 0 <= corner[1] < img.shape[0] for corner in corners):
                            x_min, y_min = np.min(corners, axis=0).astype(int)
                            x_max, y_max = np.max(corners, axis=0).astype(int)
    
                            cv2.line(img, (x_min, y_min), (x_max, y_min), (0, 0, 255, 255), 1)
                            cv2.line(img, (x_min, y_max), (x_max, y_max), (0, 0, 255, 255), 1)
                            cv2.line(img, (x_min, y_min), (x_min, y_max), (0, 0, 255, 255), 1)
                            cv2.line(img, (x_max, y_min), (x_max, y_max), (0, 0, 255, 255), 1)
                            label = 'vehicle'  # replace with the appropriate label for each object type
                            cv2.putText(img, label, (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255, 255), 1)

    
                #if forward_vec.dot(ray) > 1:
                #    # Cycle through the vertices
                #    verts = [v for v in bb.get_world_vertices(carla.Transform())]
                #    for edge in edges:
                #        # Join the vertices into edges
                #        p1 = self.get_image_point(verts[edge[0]], self.K, world_2_camera)
                #        p2 = self.get_image_point(verts[edge[1]],  self.K, world_2_camera)
                #        # Draw the edges into the camera output
                #        cv2.line(img, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (0,0,255, 255), 1)
    
        # Convert the image back to a QImage object and display it
        qimage = QtGui.QImage(img.data, image.width, image.height, QtGui.QImage.Format_RGB32)
        #pixmap = QtGui.QPixmap.fromImage(qimage)

        # Convert the QImage object to a QPixmap object and display it
        pixmap = QtGui.QPixmap.fromImage(qimage)
        self.label_bounding.setPixmap(pixmap)
        self.camera_tick +=1

    def update_bounding_box_view(self, camera, image, objects):
        # Convert the Image object to a QImage object
        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
        edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]
        img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        img = np.reshape(img, (image.height, image.width, 4))
        bounding_box_set = self.world.get_level_bbs(carla.CityObjectLabel.Car)
        ego_bb = self.vehicle.bounding_box
        # Filter bounding boxes based on distance from camera
        bounding_box_set = [bb for bb in bounding_box_set if (bb.location.distance(camera.get_location()) < 50) and (not  bb == ego_bb)]
    
        for bb in bounding_box_set:
            # Check if the bounding box is visible to the camera
            line_of_sight = self.world.get_line_of_sight(camera.get_location(), bb.location)
            if line_of_sight.has_value() and line_of_sight.value().hit_actor.id == bb.actor.id:
                corners = bb.get_world_vertices(carla.Transform())
                corners = [self.get_image_point(corner, self.K, world_2_camera) for corner in corners]
        
                # Use NumPy to calculate min/max corners
                corners = np.array(corners, dtype=int)
                x_min, y_min = np.min(corners, axis=0)
                x_max, y_max = np.max(corners, axis=0)
        
                # Check if bounding box is inside image dimensions
                if x_min >= img.shape[1] or x_max < 0 or y_min >= img.shape[0] or y_max < 0:
                    continue
            
                # Draw bounding box using OpenCV's rectangle function
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        
                label = 'vehicle'  # replace with the appropriate label for each object type
                cv2.putText(img, label, (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Convert the image back to a QImage object and display it
        qimage = QtGui.QImage(img.data, image.width, image.height, QtGui.QImage.Format_RGB32)
        #pixmap = QtGui.QPixmap.fromImage(qimage)

        # Convert the QImage object to a QPixmap object and display it
        pixmap = QtGui.QPixmap.fromImage(qimage)
        self.label_bounding.setPixmap(pixmap)
        self.camera_tick += 1


    def update_seg_camera_view(self, image):
        image.convert(carla.ColorConverter.CityScapesPalette)
        np_img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        np_img = np_img.reshape((image.height, image.width, 4))
        np_img = np_img[..., :3]  # Remove the alpha channel
        #qimage = QtGui.QImage(bytes(np_img.data), image.width, image.height, QtGui.QImage.Format_RGB888)
        #qimage = qimage.rgbSwapped()

    # Use the palette to create a QImage
        qimage = QtGui.QImage(bytes(np_img.data), image.width, image.height, QtGui.QImage.Format_RGB888).rgbSwapped()
        #qimage = qimage.rgbSwapped()
        #self.semsegConversion(qimage)
        #qimage.setColorTable(palette_list)
        pixmap = QtGui.QPixmap.fromImage(qimage)
        self.label_seg.setPixmap(pixmap)
        self.camera_tick +=1

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
    ui = CarlaUI()
    ui.show()
    sys.exit(app.exec_())