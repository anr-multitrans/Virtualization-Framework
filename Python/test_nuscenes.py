import matplotlib.pyplot as plt
import tqdm
import numpy as np

from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.bitmap import BitMap

# Load NuScenes map for a given location
nusc_map = NuScenesMap(dataroot='C:/Users/adaoud/Documents/dataset/nuScenes-panoptic-v1.0-mini/panoptic/v1.0-mini', map_name='boston-seaport')

# Load all layers of the map
layers = nusc_map.layers

# Plot each layer of the map
for layer_name, layer_props in layers.items():
    layer_mask = nusc_map.get_map_mask(layer_name)
    plt.imshow(layer_mask, cmap='gray')
    plt.title(layer_name)
    plt.show()
