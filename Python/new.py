from nuscenes import NuScenes

# Load the NuScenes dataset
nusc = NuScenes(version='v1.0-mini', dataroot='C:\\nu')


# Load the scene
scene = nusc.scene[0]

# Get the scene description
print(f"Scene name: {scene['name']}")
print(f"Scene description: {scene['description']}")

# Get the scene annotations
sample_annotation_tokens = scene['anns']
for sample_annotation_token in sample_annotation_tokens:
    # Load the sample annotation
    sample_annotation = nusc.get('sample_annotation', sample_annotation_token)

    # Get the object category
    category = sample_annotation['category_name']

    # Get the object position and orientation
    position = sample_annotation['translation']
    orientation = sample_annotation['rotation']

    # Print the object information
    print(f"Object category: {category}")
    print(f"Object position: {position}")
    print(f"Object orientation: {orientation}")


scene = nusc.scene[0]

# Get the scene annotations
for x in scene:
    print(x)
    print(scene[x])
sample_annotation_tokens = scene['anns']
