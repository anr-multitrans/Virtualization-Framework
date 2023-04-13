from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
import numpy as np


# Create a new NuScenes dataset in memory
nusc = NuScenes(version='v1.0-mini', dataroot='c:\\nu')
#nusc = explorer.nusc
nusc.list_scenes()

# Save the dataset to a new directory
new_dir = 'c:\\newdataset'
nusc.write_to_disk(new_dir)

# Add a new scene
scene_rec = {
    'token': 'test_scene',
    'log_token': 'test_log',
    'nbr_samples': 0,
    'first_sample_token': '',
    'last_sample_token': '',
    'name': 'Test Scene',
    'description': 'A test scene',
}
print(nusc.scene)

print('------------\n then \n -----------')

# Append the new scene to the list of scenes in the dataset
scene_token= nusc.scene.append(scene_rec)

# Print out the list of scenes in the dataset
print(nusc.scene)
#scene_token = nusc.add_scene('test_scene')

# Add a new sample to the scene
sample_token = nusc.add_sample('test_sample', scene_token)



# Define the box information
category_name = 'car'
translation = np.array([0, 0, 0])
size = np.array([2, 1, 1])
rotation = Quaternion(axis=[0, 0, 1], angle=0)
velocity = np.array([0, 0, 0])

# Create a new box object
box = Box(translation=translation, size=size, rotation=rotation)

# Create a new annotation object
car_annotation = nusc.create_annotation(
    sample_token=sample_token,
    category_name=category_name,
    translation=translation,
    size=size,
    rotation=rotation,
    velocity=velocity
)

# Add the annotation to the NuScenes dataset
nusc.add_ann_to_recs(car_annotation)