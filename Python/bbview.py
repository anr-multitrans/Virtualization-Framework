def update_bounding_box_view_1(self, camera, image):
        world_2_camera = np.array(camera.get_transform().get_inverse_matrix()) 
        img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        img = np.reshape(img, (image.height, image.width, 4))
        for npc in self.world.get_actors():
            if npc.id != self.vehicle.id:
                bb = npc.bounding_box
                dist = npc.get_transform().location.distance(self.vehicle.get_transform().location)
                if dist < 50:
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
                            if p[0] > x_max:
                                x_max = p[0]
                            if p[0] < x_min:
                                x_min = p[0]
                            if p[1] > y_max:
                                y_max = p[1]
                            if p[1] < y_min:
                                y_min = p[1]
                        cv2.line(img, (int(x_min),int(y_min)), (int(x_max),int(y_min)), (0,0,255, 255), 1)
                        cv2.line(img, (int(x_min),int(y_max)), (int(x_max),int(y_max)), (0,0,255, 255), 1)
                        cv2.line(img, (int(x_min),int(y_min)), (int(x_min),int(y_max)), (0,0,255, 255), 1)
                        cv2.line(img, (int(x_max),int(y_min)), (int(x_max),int(y_max)), (0,0,255, 255), 1)




        