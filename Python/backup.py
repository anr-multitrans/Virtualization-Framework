#carla.CityObjectLabel.Car
        for label in objects:
            print("for loop")
            bounding_box_set = self.world.get_level_bbs(self.bb_labels[label])
            ego_bb = self.vehicle.bounding_box
            # Filter bounding boxes based on distance from camera
            bounding_box_set = [bb for bb in bounding_box_set if (bb.location.distance(camera.get_location()) < 50) and (not  bb == ego_bb) and (bb.location.distance(camera.get_location()) >1 )]
        
            for bb in bounding_box_set:
                print("second for loop")
                # Check if the bounding box is visible to the camera
                #line_of_sight = self.world.get_line_of_sight(camera.get_location(), bb.location)
                forward_vec = self.vehicle.get_transform().get_forward_vector()
                print('forward_vec')
                bb_direction = bb.location - camera.get_transform().location
                print('bb_direction')
                dot_product = forward_vec.x * bb_direction.x + forward_vec.y * bb_direction.y + forward_vec.z * bb_direction.z
                print('dot_product')
                if dot_product > 0:
                    print("dot_product (if)")
                #if np.dot(forward_vec, bb_direction) > 0:
                #relative_location = bb.location - camera.get_location()
                #ang=camera.get_forward_vector().get_angle(relative_location)
                #if ang < 80 and ang>=0:
                # Define percentage to reduce bounding box size by
                   

                    corners = bb.get_world_vertices(carla.Transform())
                    corners = [self.get_image_point(corner, self.K, world_2_camera) for corner in corners]

                    # Use NumPy to calculate min/max corners
                    corners = np.array(corners, dtype=int)
                    x_min, y_min = np.min(corners, axis=0)
                    x_max, y_max = np.max(corners, axis=0)

                    # Calculate new bounding box dimensions

                    
                    
                   

                    # Check if bounding box is inside image dimensions
                    if x_min >= img.shape[1] or x_max < 0 or y_min >= img.shape[0] or y_max < 0:
                        continue
                    object_color=self.CLASS_MAPPING[label]
                    object_color=(object_color[2], object_color[1], object_color[0])
                    x_min_new , y_min_new, x_max_new, y_max_new = self.tighten_bb(semantic_image, x_min, y_min, x_max, y_max, object_color )
                    if not(x_min_new==-1 and y_min_new ==-1 and x_max_new==-1 and y_max_new== -1):

                        # Draw bounding box using OpenCV's rectangle function
                        cv2.rectangle(img, (x_min_new, y_min_new), (x_max_new, y_max_new), (0, 0, 255), 2)

                        #label = 'vehicle'  # replace with the appropriate label for each object type
                        cv2.putText(img, label, (x_min_new, y_min_new-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, object_color, 1)

        # Convert the image back to a QImage object and display it
        qimage = QtGui.QImage(img.data, image.width, image.height, QtGui.QImage.Format_RGB32)
        print("qimage")
        #pixmap = QtGui.QPixmap.fromImage(qimage)

        # Convert the QImage object to a QPixmap object and display it
        pixmap = QtGui.QPixmap.fromImage(qimage)
        print("pixmap")
        self.label_bounding.setPixmap(pixmap)
        print("label_bounding")
        self.camera_tick += 1
        print("camera_tick")
        #self.synchroTick()