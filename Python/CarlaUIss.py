import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QHBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5 import QtGui
from carla import ColorConverter
import carla
import random
import numpy as np

class CarlaUI(QWidget):
    def __init__(self):
        super().__init__()

        # Initialize the Carla client and world
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        # Create a vehicle
        vehicle_bp = random.choice(self.world.get_blueprint_library().filter('vehicle.*'))
        vehicle_transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(vehicle_bp, vehicle_transform)

        # Attach the semantic segmentation camera sensor to the vehicle
        self.seg_camera_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        self.seg_camera_bp.set_attribute('image_size_x', '640')
        self.seg_camera_bp.set_attribute('image_size_y', '480')
        self.seg_camera_bp.set_attribute('fov', '90')
        self.seg_camera_bp.set_attribute('sensor_tick', '0.0')
        self.seg_camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))

        self.seg_camera = self.world.spawn_actor(
            self.seg_camera_bp,
            self.seg_camera_transform,
            attach_to=self.vehicle
        )

        # Set up the user interface
        self.seg_camera_view = QLabel(self)
        self.seg_camera_view.setFixedSize(640, 480)
        layout = QHBoxLayout(self)
        layout.addWidget(self.seg_camera_view)
        self.setLayout(layout)

        # Start the main loop
        self.seg_camera.listen(lambda data: self.update_seg_camera_view(data))
        self.timer = self.startTimer(100)
        self.running = True

    def timerEvent(self, event):
        #pass
        self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))

    def update_seg_camera_view(self, image):
        np_img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        np_img = np_img.reshape((image.height, image.width, 4))
        np_img = np_img[..., :3]  # Remove the alpha channel
        qimage = QtGui.QImage(bytes(np_img.data), image.width, image.height, QtGui.QImage.Format_RGB888)
        qimage = qimage.rgbSwapped()
        color_table = [rgb[0] << 16 | rgb[1] << 8 | rgb[2] for rgb in ColorConverter().get_color_palette()]
        qimage.setColorTable(color_table)
        qimage = qimage.convertToFormat(QtGui.QImage.Format_Indexed8)
        pixmap = QtGui.QPixmap.fromImage(qimage)
        self.seg_camera_view.setPixmap(pixmap)


class ColorConverter:
    CityScapesPalette = [        [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
        [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152],
        [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
        [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32], [0, 0, 0]
    ]
    
    def get_color_palette(self):
        return self.CityScapesPalette
  
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = CarlaUI()
    ui.show()
    sys.exit(app.exec_())