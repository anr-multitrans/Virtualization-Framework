from nuscenes import NuScenes

# Load the NuScenes dataset
nusc = NuScenes(version='v1.0-mini', dataroot='C:\\nu')

# Get a list of all scene names
scene_names = [s['name'] for s in nusc.scene]

nuscene_to_carla = {
    'vehicle.car': 'vehicle.tesla.model3',
    'vehicle.truck': 'vehicle.carlamotors.carlacola',
    'vehicle.bus': 'vehicle.volkswagen.t2',
    'vehicle.trailer': 'vehicle.carlamotors.carlacola',
    'vehicle.motorcycle': 'vehicle.diamondback.century',
    'vehicle.bicycle': 'vehicle.diamondback.century',
    'human.pedestrian.adult': 'walker.pedestrian.0001',
    'human.pedestrian.child': 'walker.pedestrian.0002',
    'human.pedestrian.construction_worker': 'walker.pedestrian.0003',
    'human.pedestrian.police_officer': 'walker.pedestrian.0004',
    'human.bicyclist': 'walker.bicyclist.0001',
    'human.motorcyclist': 'walker.motorcyclist.0001',
    'construction.cone': 'static.prop.cones',
    'construction.barrier': 'static.prop.cones',
    'construction.road_block': 'static.prop.cones',
    'construction.sign': 'static.prop.cones',
    'construction.misc': 'static.prop.cones',
    'object': 'static.prop.furniture',
    'nature': 'static.prop.tree',
    'vehicle_part': 'static.prop.furniture',
    'flat.surface': 'static.prop.building',
    'sidewalk': 'static.prop.furniture',
    'parking': 'static.prop.furniture',
    'traffic_sign/pole': 'static.prop.furniture'
}

#In this example, we map each NuScenes object category to a Carla blueprint based on their functional roles, size, and shape. 
#For example, we map the "vehicle.car" category in NuScenes to the "vehicle.tesla.model3" blueprint in Carla, and the "human.pedestrian.adult" category to the "walker.pedestrian.0001" blueprint.


# Load the scene
scene = nusc.scene[0]

# Get the scene description
#print(f"Scene name: {scene['name']}")
#print(f"Scene description: {scene['description']}")

# Get the scene annotations
for x in scene:
    print(x)
    print(scene[x])

scene_keys = list(scene.keys())
print(scene_keys)

sample_token = scene['first_sample_token']
sample_annotations = nusc.get('sample', sample_token)['anns']
#sample_annotation_tokens = scene['anns']


if len(sample_annotations) > 0:
    # Iterate over the sample annotation tokens
    for sample_annotation_token in sample_annotations:
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