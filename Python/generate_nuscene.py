from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
import numpy as np

# Create a new NuScenes dataset in memory
nusc = NuScenes(version='v1.0-mini', dataroot='C:\\nu')
new_annotation = {
    'instance_token': 'new_instance_token',
    'visibility_token': 'new_visibility_token',
    'attribute_tokens': [],
    'translation': [0, 0, 0],
    'size': [1, 1, 1],
    'rotation': [1, 0, 0, 0],
    'prev': '',
    'next': '',
    'num_lidar_pts': 0,
    'num_radar_pts': 0,
    'category_name': 'car',
    'attribute_name': '',
#    'bbox_corners': Box([0, 0, 0], [1, 1, 1], [1, 0, 0, 0]).corners(),
    'score': 0,
    'md5': '1234567890abcdef',
    'sample_token': 'sample_token',
    'token': 'new_annotation_token'
}


# Create a new scene
scene_token = nusc.scene[0]['token']
scene = nusc.get('scene', scene_token)
sample_token = scene['first_sample_token']

# Create some sample annotations
while sample_token:
    # Load the sample
    sample = nusc.get('sample', sample_token)

    # Create a sample annotation for a car
    car_annotation = {
        'sample_token': sample_token,
        'translation': [0, 0, 0],
        'rotation': [1, 0, 0, 0],
        'size': [1.5, 4, 2],
        'category_name': 'car',
        'attribute_name': '',
        'visibility_token': '',
        'num_lidar_pts': 0,
        'num_radar_pts': 0
    }
    existing_ann_tokens = sample['anns']

    sample_annotations = nusc.get('sample', sample_token)['anns']

    # Add the sample annotation to the NuScenes dataset
    car_annotation_token = scene['anns'].append(car_annotation)

    # Move on to the next sample
    sample_token = sample['next']

# Save the modified NuScenes dataset
nusc.save()
