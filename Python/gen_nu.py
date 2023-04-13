from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box

# Create a new annotation dictionary
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

# Load the NuScenes dataset
nusc = NuScenes(version='v1.0-mini', dataroot='C:\\nu')


# Get the scene you want to add the annotation to
scene = nusc.scene[0]
sample_token = scene['first_sample_token']

sample = nusc.get('sample', sample_token)
existing_ann_tokens = sample['anns']
print('-------------------')
print('before')
print('-------------------')
print(existing_ann_tokens)

# Append the new annotation to the list of annotations for the scene
#existing_ann_tokens.append(new_annotation)

print('-------------------')
print('after')
print('-------------------')
#print(existing_ann_tokens)



sample_token = nusc.sample[0]['token']
sample = nusc.get('sample', sample_token)

# Get the existing annotations
existing_ann_tokens = sample['anns']

# Create a new annotation
new_ann = {
    'sample_token': sample_token,
    'category_name': 'car',
    'bbox': [0.0, 0.0, 0.0, 0.0],  # Set the bbox to zeros for now
    'attributes': {},
}

# Add the new annotation to the list of annotations
new_ann_token = nusc.add_annotation(new_ann)

# Update the list of annotations for the sample
existing_ann_tokens.append(new_ann_token)
nusc.samples[sample_token]['anns'] = existing_ann_tokens

# Update the NuScenes database
nusc.update_annotation_metadata()

# Save the modified scene back to the dataset
nusc.mod_scene(scene)