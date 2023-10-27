import carla
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen, QTransform
import sys
import math

# Define your type-color mapping
CLASS_MAPPING = {
    'unlabeled': (0, 0, 0),
    'fences': (100, 40, 40),
    'other': (55, 90, 80),
    'pedestrians': (220, 20, 60),
    'cyclists': (255, 0, 0),
    'poles': (153, 153, 153),
    'vegetation': (107, 142, 35),
    'bicycles': (119, 11, 32),
    'buses': (0, 60, 100),
    'cars': (0, 0, 142),
    'trucks': (0, 0, 70),
    'motorcycles': (0, 0, 230),
    'vehicles': (0, 0, 142),
    'walls': (102, 102, 156),
    'traffic_signs': (220, 220, 0),
    'bridge': (150, 100, 100),
    'rail_track': (230, 150, 140),
    'guard_rail': (180, 165, 180),
    'traffic_light': (250, 170, 30),
    'static': (110, 190, 160),
    'dynamic': (170, 120, 50),
}

class BirdseyeViewWindow(QMainWindow):
    def __init__(self, world, ego_vehicle, spatial_range, image_size):
        super().__init__()
        self.setWindowTitle("Bird's-Eye View")
        self.setGeometry(100, 100, image_size, image_size)
        self.world = world
        self.spatial_range = spatial_range
        self.image_size = image_size
        self.ego_vehicle = ego_vehicle

        # Create a central widget and a layout to hold the QLabel
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create a QLabel to display the pixmap
        self.image_label = QLabel(self)
        layout.addWidget(self.image_label)

    def paintEvent(self, event):
        pixmap = QPixmap(self.image_size, self.image_size)
        pixmap.fill(QColor(0, 0, 0))
        painter = QPainter(pixmap)
        painter.setPen(QPen(QColor(0, 255, 0), 2))

        if self.ego_vehicle is not None:

            ego_location = self.ego_vehicle.get_location()
            ego_x, ego_y = ego_location.x, ego_location.y
            ego_rotation = self.ego_vehicle.get_transform().rotation

            # Calculate the scaling factor to map spatial range to image size
            scale_factor = self.image_size / (2 * self.spatial_range)

            # Create a transformation matrix for rotation
            transform = QTransform()
            transform.translate(self.image_size / 2, self.image_size / 2)
            transform.rotate(ego_rotation.yaw)  # Negate yaw to match vehicle orientation
            transform.translate(-self.image_size / 2, -self.image_size / 2)

            # Apply the transformation to the painter
            painter.setTransform(transform)

            # Iterate through bounding boxes and render them in bird's-eye perspective
            bounding_boxes = get_bounding_boxes_around_ego_vehicle(self.world, self.ego_vehicle, self.spatial_range)
            for bbox_info in bounding_boxes:
                x, y = bbox_info['location'].x - ego_x, bbox_info['location'].y - ego_y
                x_pixel = int((x + self.spatial_range) * scale_factor)
                y_pixel = int((self.spatial_range - y) * scale_factor)

                # Handle cases where extent values are not finite
                bbox_extent_x = int(bbox_info['bbox'].extent.x) if math.isfinite(bbox_info['bbox'].extent.x) else 0
                bbox_extent_y = int(bbox_info['bbox'].extent.y) if math.isfinite(bbox_info['bbox'].extent.y) else 0
                bbox_extent_x *= scale_factor
                bbox_extent_y *= scale_factor
                #print(bbox_info['type'])
                type_str = str(bbox_info['type']).lower()
                # Filter out objects higher than the bird's eye or not of interest
                #if bbox_info['location'].z <= 0 and bbox_info['type'].lower() in CLASS_MAPPING:
                try:
                    color = QColor(*CLASS_MAPPING[type_str])
                    painter.setPen(QPen(color, 2))
                    painter.drawRect(y_pixel - bbox_extent_y, x_pixel - bbox_extent_x,
                                     y_pixel + bbox_extent_y, x_pixel + bbox_extent_x)
                except Exception as e:
                    print(e)
            self.ego_vehicle.set_autopilot(True)
        painter.end()

        # Set the pixmap to the QLabel for display
        self.image_label.setPixmap(pixmap)


def get_bounding_boxes_around_ego_vehicle(world, ego_vehicle, spatial_range):
    # Get the ego vehicle's location
    ego_location = ego_vehicle.get_location()

    # Define a bounding box around the ego vehicle within the spatial range
    bbox_x = [ego_location.x - spatial_range, ego_location.x + spatial_range]
    bbox_y = [ego_location.y - spatial_range, ego_location.y + spatial_range]

    # Create an empty list to store bounding box information
    bounding_boxes = []

    # Get bounding boxes of actors
    for actor in world.get_actors():
        if actor.id != ego_vehicle.id:
            actor_location = actor.get_location()
            if bbox_x[0] <= actor_location.x <= bbox_x[1] and bbox_y[0] <= actor_location.y <= bbox_y[1]:
                actor_type = actor.type_id
                actor_bbox = actor.bounding_box  # Bounding box extent
                actor_id = actor.id
                bounding_boxes.append({
                    'id': actor_id,
                    'type': actor_type,
                    'bbox': actor_bbox,
                    'location': actor_location
                })

    # Get bounding boxes of map level objects (static objects)
    map_objects = world.get_environment_objects()
    for map_object in map_objects:
        map_object_location = map_object.transform.location
        if bbox_x[0] <= map_object_location.x <= bbox_x[1] and bbox_y[0] <= map_object_location.y <= bbox_y[1]:
            map_object_type = map_object.type
            map_object_bbox = map_object.bounding_box  # Bounding box extent
            map_object_id = map_object.id
            bounding_boxes.append({
                'id': map_object_id,
                'type': map_object_type,
                'bbox': map_object_bbox,
                'location': map_object_location
            })

    return bounding_boxes



def main():
    app = QApplication(sys.argv)
    
    # Connect to the CARLA server and create bounding boxes
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    
    # Spawn an ego vehicle and apply autopilot
    blueprint_library = world.get_blueprint_library()
    ego_vehicle_bp = blueprint_library.find('vehicle.tesla.model3') #-53, 55, 1
    ego_vehicle_transform = carla.Transform(carla.Location(x=-53, y=55, z=2), carla.Rotation())
    ego_vehicle = world.spawn_actor(ego_vehicle_bp, ego_vehicle_transform)
    ego_vehicle.set_autopilot(True)  # Enable autopilot for the ego vehicle
    
    spatial_range = 50.0
    image_size = 1200
    
    # Create and show the PyQt window
    window = BirdseyeViewWindow(world, ego_vehicle, spatial_range, image_size)
    #window.ego_vehicle = ego_vehicle  # Pass the ego vehicle to the window
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
