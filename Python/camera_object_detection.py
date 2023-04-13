import carla
import numpy as np
import cv2

class YoloObjectDetector:
    def __init__(self, parent_actor, world, model_weights, model_cfg, labels):
        self.sensor = None
        self.sensor_data = None
        self.labels = labels
        
        bp_library = world.get_blueprint_library()

        # Create a RGB camera sensor
        self.sensor = bp_library.find('sensor.camera.rgb')
        self.sensor.set_attribute('image_size_x', '1920')
        self.sensor.set_attribute('image_size_y', '1080')
        self.sensor.set_attribute('fov', '110')
        self.sensor.set_attribute('sensor_tick', '0.05')

        # Create a callback function to handle the camera data
        self.sensor.listen(lambda data: self._process_image(data, model_weights, model_cfg))

        # Attach the sensor to the parent actor
        self.sensor_actor = world.spawn_actor(self.sensor, carla.Transform(), attach_to=parent_actor)

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

        # Perform YOLO object detection on the image
        net = cv2.dnn.readNet(model_weights, model_cfg)
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
