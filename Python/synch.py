import carla
import numpy as np
import cv2
import random
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow
from PyQt5.QtGui import QPixmap, QImage

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CARLA Sensor Viewer")

        # Set up the CARLA client and connect to the CARLA server.
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)

        # Get the CARLA world and map.
        self.world = self.client.get_world()
        self.map = self.world.get_map()

        # Set up the synchronous mode.
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1
        self.world.apply_settings(settings)
        self.context = self.world

        # Create a vehicle
        vehicle_bp = random.choice(self.world.get_blueprint_library().filter('vehicle.dodge.*'))
        #vehicle.dodge.charger_2020
        vehicle_transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(vehicle_bp, vehicle_transform)

        # Spawn the RGB camera.
        rgb_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        rgb_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.rgb_sensor = self.world.spawn_actor(rgb_bp, rgb_transform, attach_to=self.vehicle)

        # Spawn the segmentation camera.
        seg_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        seg_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.seg_sensor = self.world.spawn_actor(seg_bp, seg_transform, attach_to=self.vehicle)

        # Set up the PyQT interface.
        self.rgb_label = QLabel(self)
        self.rgb_label.move(0, 0)
        self.seg_label = QLabel(self)
        self.seg_label.move(640, 0)
        self.setGeometry(0, 0, 1280, 720)
        self.show()

        # Start the sensor data loop.
        self.timer = self.startTimer(100)

    def timerEvent(self, event):
        # Advance the simulation and capture data from the sensors.
        self.world.tick()
        rgb_data = self.rgb_sensor.get()
        seg_data = self.seg_sensor.get()

        # Convert the sensor data to images and display them in the PyQT interface.
        rgb_image = np.array(rgb_data.raw_data)
        rgb_image = rgb_image.reshape((rgb_data.height, rgb_data.width, 4))
        rgb_image = rgb_image[:, :, :3]
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        rgb_image = QImage(rgb_image.data, rgb_image.shape[1], rgb_image.shape[0], QImage.Format_RGB888)
        self.rgb_label.setPixmap(QPixmap.fromImage(rgb_image))

        seg_image = np.array(seg_data.raw_data)
        seg_image = seg_image.reshape((seg_data.height, seg_data.width, 4))
        seg_image = seg_image[:, :, :3]
        seg_image = cv2.cvtColor(seg_image, cv2.COLOR_BGR2RGB)
        seg_image = QImage(seg_image.data, seg_image.shape[1], seg_image.shape[0], QImage.Format_RGB888)
        self.seg_label.setPixmap(QPixmap.fromImage(seg_image))
if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    app.exec_()
