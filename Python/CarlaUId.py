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

        # Create a vehicle
        vehicle_bp = random.choice(self.world.get_blueprint_library().filter('vehicle.*'))
        vehicle_transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(vehicle_bp, vehicle_transform)

        # Initialize the object detection module
        self.model_weights = 'yolov3.weights'
        self.model_cfg = 'yolov3.cfg'
        self.labels = ['person', 'car', 'truck'] # list of object classes

        object_detection = YoloObjectDetector(self.vehicle, self.world, self.model_weights, self.model_cfg, self.labels, self)

        # Set up the user interface
        self.camera_view = QLabel(self)
        self.camera_view.setFixedSize(640, 480)
        layout = QHBoxLayout(self)
        layout.addWidget(self.camera_view)
        self.setLayout(layout)

        # Start the main loop
        #self.object_detection.listen(lambda data: self.update_camera_view(data))
        self.timer = self.startTimer(100)
        self.running = True

    def timerEvent(self, event):
        #pass
        self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))

    def update_camera_view(self, image):
        np_img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        np_img = np_img.reshape((image.height, image.width, 4))
        np_img = np_img[..., :3]  # Remove the alpha channel
        qimage = QtGui.QImage(bytes(np_img.data), image.width, image.height, QtGui.QImage.Format_RGB888)
        qimage = qimage.rgbSwapped()
        pixmap = QtGui.QPixmap.fromImage(qimage)
        self.camera_view.setPixmap(pixmap)

class YoloObjectDetector:
    def __init__(self, parent_actor, world, model_weights, model_cfg, labels,ui=None):
        self.sensor = None
        self.sensor_data = None
        self.parent_actor = parent_actor
        self.world = world
        self.model_weights = model_weights
        self.model_cfg = model_cfg
        self.labels = labels
        self.ui = ui
        
        bp_library = world.get_blueprint_library()

        # Create a RGB camera sensor
        self.rgb_camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.rgb_camera_bp.set_attribute('image_size_x', '1280')
        self.rgb_camera_bp.set_attribute('image_size_y', '1024')
        self.rgb_camera_bp.set_attribute('fov', '90')
        self.rgb_camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.sensor = self.world.spawn_actor(
            self.rgb_camera_bp,
            self.rgb_camera_transform,
            attach_to=parent_actor
        )
        model_weights = 'c:/yolo/yolov3-wider_16000.weights'
        model_cfg = 'C:/CARLA/latest/PythonAPI/examples/yolov3.cfg'
        frameWidth= 640
        frameHeight = 480
        net = cv2.dnn.readNet(model_weights, model_cfg)
        classes = []
        with open("coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()] # we put the names in to an array

        layers_names = net.getLayerNames()
        print(layers_names)
        print(len(layers_names))
        print(net.getUnconnectedOutLayers())
        print(len(net.getUnconnectedOutLayers()))

        unconnected_layers = net.getUnconnectedOutLayers()
        if len(unconnected_layers)>0:
            output_layers = [layers_names[i[0] - 1] for i in unconnected_layers if i[0] - 1 < len(layers_names)]
        else:
            print("No unconnected output layers found.")

        #output_layers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers() if i[0] - 1 < len(layers_names)]

        # Create a callback function to handle the camera data
       # self.sensor.listen(lambda data: self._process_image(data, model_weights, model_cfg))

        # Attach the sensor to the parent actor
        #self.sensor_actor = world.spawn_actor(self.sensor, carla.Transform(), attach_to=parent_actor)

    def destroy(self):
        # Stop listening and destroy the actor
        self.sensor.stop()
        self.sensor_actor.destroy()

    def get_detection(self):
        return self.sensor_data

    def _process_image(self, data, model_weights, model_cfg):
        # Convert the image to a numpy array and reshape it to the correct dimensions
        np_img = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
        np_img = np_img.reshape((data.height, data.width, 4))
        np_img = np_img[..., :3]
        #f = open(model_weights)
        #print(f.read()) 
        # Perform YOLO object detection on the image
        net = cv2.dnn.readNetFromDarknet(model_weights, model_cfg)
        blob = cv2.dnn.blobFromImage(np_img, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []
        (height, width) = np_img.shape[:2]

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = center_x - w // 2
                    y = center_y - h // 2

                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        objects = []

        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]

            label = str(self.labels[class_ids[i]])
            confidence = confidences[i]

            objects.append({'label': label, 'confidence': confidence, 'x': x, 'y': y, 'w': w, 'h': h})

        self.sensor_data = objects
        if ui != None:
            ui.update_camera_view(data)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = CarlaUI()
    ui.show()
    sys.exit(app.exec_())