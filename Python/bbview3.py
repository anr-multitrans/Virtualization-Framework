def update_bounding_box_view(self, camera, image, objects):
        # Convert the Image object to a QImage object
        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
        edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]
        img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        img = np.reshape(img, (image.height, image.width, 4))
        bounding_box_set = self.world.get_level_bbs(carla.CityObjectLabel.Car)
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
    
        # Convert the image back to a QImage object and display it
        qimage = QtGui.QImage(img.data, image.width, image.height, QtGui.QImage.Format_RGB32)
        #pixmap = QtGui.QPixmap.fromImage(qimage)

        # Convert the QImage object to a QPixmap object and display it
        pixmap = QtGui.QPixmap.fromImage(qimage)
        self.label_bounding.setPixmap(pixmap)
        self.camera_tick +=1