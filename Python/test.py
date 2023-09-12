import numpy as np
import carla

def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def get_image_point(loc, K, w2c):
        # Calculate 2D projection of 3D coordinate

        # Format the input coordinate (loc is a carla.Position object)
        point = np.array([loc.x, loc.y, loc.z, 1])
        # transform to camera coordinates
        point_camera = np.dot(w2c, point)

        # New we must change from UE4's coordinate system to an "standard"
        # (x, y ,z) -> (y, -z, x)
        # and we remove the fourth componebonent also
        point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

        # now project 3D->2D using the camera matrix
        point_img = np.dot(K, point_camera)
        # normalize
        point_img[0] /= point_img[2]
        point_img[1] /= point_img[2]

        return point_img[0:2]

def match(min_x1, min_y1, max_x1, max_y1, min_x2, min_y2, max_x2, max_y2, threshold=0.3):
    r1_area = (max_x1 - min_x1) * (max_y1 - min_y1)
    r2_area = (max_x2 - min_x2) * (max_y2 - min_y2)
    
    max_area = max(r1_area, r2_area)
    min_area = min(r1_area, r2_area)
    #if max_area > 3 * min_area:
    #    return False  # Huge difference in areas
    
    intersection_min_x = max(min_x1, min_x2)
    intersection_min_y = max(min_y1, min_y2)
    intersection_max_x = min(max_x1, max_x2)
    intersection_max_y = min(max_y1, max_y2)
    print(f"min_xi={intersection_min_x}")
    print(f"min_yi={intersection_min_y}")
    print(f"max_xi={intersection_max_x}")
    print(f"max_yi={intersection_max_y}")
    if intersection_max_x <= intersection_min_x or intersection_max_y <= intersection_min_y:
        return False  # No intersection
    
    
    intersection_area = (intersection_max_x - intersection_min_x) * (intersection_max_y - intersection_min_y)
    
    return (intersection_area > threshold * r1_area and intersection_area > threshold * r2_area)

# Given camera parameters and transform
#l= carla.Location(x=-110.291763, y=97.093193, z=3.002939)
#r= carla.Rotation(pitch=-8.006648, yaw=-114.636604, roll=0.000058)
camera_transform_info = {
    'location': {
        "x": -110.291763,
        "y": 97.093193,
        "z": 3.002939
    },
    "rotation": {
        "pitch": -8.006648,
        "yaw": -114.636604,
        "roll": 0.000058
    }
}
camera_parameters = {
    "image_size_x": 800,
    "image_size_y": 600,
    "fov": 90,
    "sensor_tick": 0.1
}

# Given bounding box information
bbp = {
          "id": 1760071176472179717,
            "name": "SM_Pergola_2531_SM_0",
            "type": 21,
            "distance": 93.53719329833984,
            "bounding_box": {
                "location": {
                    "x": -27.19999885559082,
                    "y": 53.94999694824219,
                    "z": 1.69253408908844
                },
                "extent": {
                    "x": 2.5146968364715576,
                    "y": 2.5425169467926025,
                    "z": 1.7707659006118774
                },
                "vertices": [
                    {
                        "x": -29.71469497680664,
                        "y": 51.40747833251953,
                        "z": -0.0782318115234375
                    },
                    {
                        "x": -29.71469497680664,
                        "y": 51.40747833251953,
                        "z": 3.4632999897003174
                    },
                    {
                        "x": -29.71469497680664,
                        "y": 56.492515563964844,
                        "z": -0.0782318115234375
                    },
                    {
                        "x": -29.71469497680664,
                        "y": 56.492515563964844,
                        "z": 3.4632999897003174
                    },
                    {
                        "x": -24.685302734375,
                        "y": 51.40747833251953,
                        "z": -0.0782318115234375
                    },
                    {
                        "x": -24.685302734375,
                        "y": 51.40747833251953,
                        "z": 3.4632999897003174
                    },
                    {
                        "x": -24.685302734375,
                        "y": 56.492515563964844,
                        "z": -0.0782318115234375
                    },
                    {
                        "x": -24.685302734375,
                        "y": 56.492515563964844,
                        "z": 3.4632999897003174
                    }
                ]
            },
            "transform": {
                "location": {
                    "x": -27.19999885559082,
                    "y": 53.94999694824219,
                    "z": 0.14999999105930328
                },
                "rotation": {
                    "pitch": 0.0,
                    "yaw": 0.0,
                    "roll": 0.0
                }
            }
        }
rrbp = {
        "name": "",
        "id": 1,
        "min_x": 162,
        "min_y": 297,
        "max_x": 167,
        "max_y": 307,
        "base_label": "static",
        "other_labels": []
    }

# Convert camera transform to matrix
camera_transform = carla.Transform(
    carla.Location(
        x=camera_transform_info['location']['x'],
        y=camera_transform_info['location']['y'],
        z=camera_transform_info['location']['z']
    ),
    carla.Rotation(
        pitch=camera_transform_info['rotation']['pitch'],
        yaw=camera_transform_info['rotation']['yaw'],
        roll=camera_transform_info['rotation']['roll']
    )
)
camera_matrix = camera_transform.get_inverse_matrix()

# Build projection matrix
projection_matrix = build_projection_matrix(
    camera_parameters['image_size_x'],
    camera_parameters['image_size_y'],
    camera_parameters['fov']
)
bbox_center = np.array([bbp['bounding_box']['location']['x'],
                        bbp['bounding_box']['location']['y'],
                        bbp['bounding_box']['location']['z']])
        
bbox_extent = np.array([bbp['bounding_box']['extent']['x'],
                        bbp['bounding_box']['extent']['y'],
                        bbp['bounding_box']['extent']['z']])

corners = bbp['bounding_box']['vertices']

projected_corners = [get_image_point(
    carla.Location(x=c['x'], y=c['y'], z=c['z']),
    projection_matrix,
    camera_matrix
) for c in corners]
        
min_x_proj = min([p[0] for p in projected_corners])
min_y_proj = min([p[1] for p in projected_corners])
max_x_proj = max([p[0] for p in projected_corners])
max_y_proj = max([p[1] for p in projected_corners])


print(f"min_xp={min_x_proj}, min_x={rrbp['min_x']}")
print(f"min_yp={min_y_proj}, min_y={rrbp['min_y']}")
print(f"max_xp={max_x_proj}, max_x={rrbp['max_x']}")
print(f"max_yp={max_y_proj}, max_y={rrbp['max_y']}")

# Compare with rrbp
if match(min_x_proj, min_y_proj, max_x_proj, max_y_proj, rrbp['min_x'], rrbp['min_y'], rrbp['max_x'], rrbp['max_y']):#match(rrbp['min_x']-1, rrbp['min_y']+1, rrbp['max_x']-1, rrbp['max_y'], rrbp['min_x'], rrbp['min_y'], rrbp['max_x'], rrbp['max_y']):

    print("Bounding boxes match!")
else:
    print("Bounding boxes do not match.")
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# Get the spectator and print its location
spectator = world.get_spectator()
transform = spectator.get_transform()
print('Spectator location:', transform.location)
print('Spectator rotation:', transform.rotation)
# Get the spectator and set its location
#spectator = world.get_spectator()
location = carla.Location(x=-119.11104583740234, y=86.80431365966797, z=1.5218690633773804)
rotation= carla.Rotation(pitch=0, yaw=-95.00008392333984, roll=0)

#location = carla.Location(x=73.083061, y=-11.341327, z=1.699303)
#rotation= carla.Rotation(pitch=-2.299704, yaw=-64.140282, roll=0.000119)

spectator.set_transform(carla.Transform(location,rotation))
