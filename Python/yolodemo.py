import carla
import torch
import numpy as np
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QHBoxLayout
from PyQt5.QtGui import QPixmap
from carla import ColorConverter
import random
import numpy as np

class ObjectDetector(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # Connect to the CARLA simulator and retrieve the object detection sensor
        self.client = carla.Client("localhost", 2000)
        self.world = self.client.get_world()
        self.sensor = self.world.get_blueprint_library().find('sensor.other.yolov5').clone()
        self.sensor_transform = carla.Transform(carla.Location(x=1.5, y=0.0, z=2.4))
        self.sensor = self.world.spawn_actor(self.sensor, self.sensor_transform)

        # Set up the PyTorch model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).fuse().eval()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Set up the PyQt widget
        self.label = QtWidgets.QLabel(self)
        self.label.setFixedSize(640, 480)
        self.show()

        # Connect to the sensor's data stream
        self.sensor.listen(lambda data: self.update(data))

    def update(self, data):
        # Convert the sensor data to a PIL Image
        image = Image.frombytes('RGBA', (data.width, data.height), bytes(data.raw_data), 'raw', 'RGBA')

        # Run the PyTorch model to detect objects in the image
        results = self.model(image.to(self.device))

        # Convert the PyTorch results to a list of bounding boxes and class labels
        bboxes = results.xyxy[0].cpu().numpy().tolist()
        labels = results.names[results.pred[0].cpu().numpy()]

        # Draw the bounding boxes and class labels on the image
        draw = ImageDraw.Draw(image)
        for bbox, label in zip(bboxes, labels):
            draw.rectangle(bbox, outline=(255, 0, 0), width=2)
            draw.text((bbox[0], bbox[1]), label, fill=(255, 0, 0))

        # Convert the PIL Image to a QImage and display it in the QLabel
        qimage = QtGui.QImage(image.tobytes(), image.width, image.height, QtGui.QImage.Format_RGB888).rgbSwapped()
        pixmap = QtGui.QPixmap.fromImage(qimage)
        self.label.setPixmap(pixmap)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = ObjectDetector()
    ui.show()
    sys.exit(app.exec_())