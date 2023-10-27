import cv2
import numpy as np
import json
import os
from image_tools import build_projection_matrix, get_image_point, CLASS_MAPPING,preprocess_catalog_tree
import carla


def precompute_catalog():
    catalog_json_filename = 'environment_object.json'
    with open(catalog_json_filename, 'r') as catalog_file:
        catalog_data = json.load(catalog_file)
    catalog_tree = catalog_data['Props catalogue']
    catalog_keywords = preprocess_catalog_tree(catalog_tree)
    return catalog_keywords

catalog_keywords=precompute_catalog()

def process_images(rgb, semseg, inst, bbs, width, height, fov, location, rotation, folder_path, tick_id):
    # Step 1: Project bounding boxes
    proj_bbs = {}
    #for label in bbs:
    #    bbs_list= bbs.get(label)
    #    l_label=label.lower()
    #    proj_bbs[l_label] = []
    #    for obj in bbs_list:
    #        bb=obj['bounding_box']
    #        # Project bb to the camera plane and compute center_coordinates
    #        center_coordinates = project_bb_to_camera(bb, width, height, fov, location, rotation)
    #        x, y = center_coordinates
    #        if 0 <= x < width and 0 <= y < height :
    #            proj_bbs[l_label].append((center_coordinates, obj))
    #print(proj_bbs)

    # Step 2: Extract isolated color regions from M = semseg + inst
    isolated_regions = extract_isolated_object_regions(semseg, inst)

    # Step 3: Process each region
    output_data = []
    for region in isolated_regions:
        # 3.a. Assign main label based on the semantic segmentation mapping
        main_label = get_main_label(region, CLASS_MAPPING, semseg)
        if main_label != "unknown":
        #print(main_label)
        # 3.b. Compute min/max_x/y
            min_x, min_y, max_x, max_y = compute_min_max_coordinates(region)
            #print(compute_min_max_coordinates(region))
            # 3.c. Get the best matching bb
            #best_matching_bb = get_best_matching_bb(main_label, proj_bbs, (min_x, min_y, max_x, max_y))

            # 3.d. Compute other labels based on the best matching bb
            #other_labels = compute_other_labels(main_label, best_matching_bb, catalog_keywords)  # Pass catalog_keywords

            # 3.e. Create JSON element
            json_element = {
                "name": f"Object_{tick_id}_{len(output_data)}",
                "id": int(len(output_data) + 1),  # Convert to int
                "min_x": int(min_x),  # Convert to int
                "min_y": int(min_y),  # Convert to int
                "max_x": int(max_x),  # Convert to int
                "max_y": int(max_y),  # Convert to int
                "base_label": main_label,
                "other_labels": []#other_labels
            }
            output_data.append(json_element)

    # Step 4: Save images and JSON
    save_results(rgb, semseg, inst, folder_path, tick_id, output_data, CLASS_MAPPING)

def compute_other_labels(main_label, best_matching_bb, catalog_keywords):
    #print(best_matching_bb)
    other_labels = []
    if best_matching_bb:
        name=best_matching_bb['name']
        # Check if the label contains any keywords from the catalog
        for keyword, labels in catalog_keywords.items():
            if keyword in name.lower() :
                unique_labels = list(set(labels))  # Remove duplicates
                other_labels.extend(unique_labels)

    return other_labels

def project_bb_to_camera(bb, width, height, fov, location, rotation):
    # Implement bb projection to camera plane logic here
    w2c = np.array(carla.Transform(location,rotation).get_inverse_matrix())

    K=build_projection_matrix(width, height, fov)
    bb_location= carla.Location(bb["location"]["x"],bb["location"]["y"],bb["location"]["z"])
    projection_center=get_image_point(bb_location, K, w2c)
    return projection_center

def extract_object_regions(semantic_image, instance_image):
    M= merge_images(semantic_image, instance_image)
    return extract_isolated_regions(M)

def extract_isolated_object_regions(semseg, inst, min_distance=50):
    # Merge 'semseg' and 'inst' images into a single image
    M = merge_images(semseg, inst)

    # Get unique pixel values in the merged image
    unique_values = np.unique(M.reshape(-1, M.shape[2]), axis=0)

    # Create a list to store the isolated regions
    isolated_regions = []

    # Iterate through unique pixel values
    for value in unique_values:
        # Create a mask for pixels with the current value
        mask = np.all(M == value, axis=2)

        # Find connected components within the mask
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)

        # Iterate through the connected components and extract isolated regions
        for label in range(1, num_labels):
            isolated_region = np.zeros_like(M)
            isolated_region[labels == label] = M[labels == label]

            # Check the distance from the centroid to other regions
            isolated = True
            centroid = centroids[label]
            for other_label in range(1, num_labels):
                if other_label != label:
                    other_centroid = centroids[other_label]
                    distance = np.linalg.norm(centroid - other_centroid)

                    if distance < min_distance:
                        isolated = False
                        break

            if isolated:
                isolated_regions.append(isolated_region)

    return isolated_regions

#This code now directly uses the color information in the M image to calculate distances between regions, and it doesn't convert the image to grayscale. This should resolve the error while still allowing you to work with color information.


def extract_isolated_object_color_regions(semseg, inst):
    # Merge 'semseg' and 'inst' images into a single image
    M = merge_images(semseg, inst)

    # Get unique pixel values in the merged image
    unique_values = np.unique(M.reshape(-1, M.shape[2]), axis=0)

    # Create a list to store the isolated regions
    isolated_regions = []

    # Iterate through unique pixel values
    for value in unique_values:
        # Create a mask for pixels with the current value
        mask = np.all(M == value, axis=2)

        # Initialize an empty region with the same shape as the merged image
        isolated_region = np.zeros_like(M)

        # Copy the matching pixels to the isolated region
        isolated_region[mask] = M[mask]

        # Append the isolated region to the list
        isolated_regions.append(isolated_region)

    return isolated_regions


def merge_images(semseg, inst):
    # Split the 'inst' image into individual channels
    b_channel, g_channel, _ = cv2.split(inst)

    # Create a new image by merging information from 'semseg' and 'inst'
    merged_image = cv2.merge([semseg, g_channel, b_channel])

    return merged_image

def extract_isolated_regions(M):
    # Create an empty list to store the isolated regions
    isolated_regions = []

    # Find contours in the merged image 'M'
    contours, _ = cv2.findContours(M, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through the detected contours
    for contour in contours:
        # Create an empty mask
        mask = np.zeros(M.shape[:2], dtype=np.uint8)

        # Draw the current contour on the mask
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Apply the mask to the original merged image
        isolated_region = cv2.bitwise_and(M, M, mask=mask)

        # Append the isolated region to the list
        isolated_regions.append(isolated_region)

    return isolated_regions

def get_main_label(region, class_mapping, semseg):
    # Get all the pixel indices of the region mask
    region_pixel_indices = np.transpose(np.nonzero(region))
    if len(region_pixel_indices)<100:
        return "unknown"

    # Extract the colors of the region pixels from the semantic image 'semseg'
    region_semantic_colors = semseg[region_pixel_indices[:, 0], region_pixel_indices[:, 1]]

    # Find the most common color within the region's semantic colors
    unique_semantic_colors, counts = np.unique(region_semantic_colors, axis=0, return_counts=True)
    main_color = unique_semantic_colors[np.argmax(counts)]
    #print(main_color)
    # Find the label that corresponds to the main color in the class mapping
    bgr_main_color = main_color[::-1] 
    label = next((key for key, value in class_mapping.items() if np.array_equal(value, bgr_main_color.tolist())), "unknown")

    return label

def compute_min_max_coordinates(region):
    # Get all pixel coordinates within the region
    region_pixel_indices = np.transpose(np.nonzero(region))

    # Extract x and y coordinates
    x_coords = region_pixel_indices[:, 1]
    y_coords = region_pixel_indices[:, 0]

    # Compute min and max coordinates
    min_x = np.min(x_coords)
    max_x = np.max(x_coords)
    min_y = np.min(y_coords)
    max_y = np.max(y_coords)

    return min_x, min_y, max_x, max_y


def get_best_matching_bb(main_label, proj_bbs, coordinates):
    min_x, min_y, max_x, max_y = coordinates
    best_matching_bb = None
    best_distance = float('inf')  # Initialize with a large value

    # Iterate through the bounding boxes for the main_label
    for coords, obj in proj_bbs.get(main_label.lower(), []):
        # Check if the center coordinates are within the specified region
        
        bb_center_x,bb_center_y = coords
        #bb_center_x = bb.center.x  # Assuming bb provides center coordinates
        #bb_center_y = bb.center.y
        #print(coordinates)
        #print(coords)
        distance = obj['distance']
        if distance < 20 and min_x <= bb_center_x <= max_x and min_y <= bb_center_y <= max_y:
            # Check if this bounding box is closer than the current best matching one
            if distance < best_distance:
                best_distance = distance
                best_matching_bb = obj
                print("match")
    if best_matching_bb is not None:
        print(best_matching_bb["name"])
    return best_matching_bb


def save_results(rgb, semseg, inst, folder_path, tick_id, output_data, class_mapping):
    # Create subfolders for images and JSON files
    image_folder = os.path.join(folder_path, "images_rgb")
    semseg_image_folder = os.path.join(folder_path, "images_semantic_segmentation")
    instance_image_folder = os.path.join(folder_path, "images_instance_segmentation")
    labeled_image_folder = os.path.join(folder_path, "images_bounding_boxes")

    json_folder = os.path.join(folder_path, "json_bounding_boxes")

    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(semseg_image_folder, exist_ok=True)
    os.makedirs(instance_image_folder, exist_ok=True)
    os.makedirs(labeled_image_folder, exist_ok=True)
    os.makedirs(json_folder, exist_ok=True)

    # Save RGB, semseg, and inst images
    image_id = f"image_{tick_id}"
    image_rgb_path = os.path.join(image_folder, f"{image_id}.png")
    image_semseg_path = os.path.join(semseg_image_folder, f"{image_id}.png")
    image_inst_path = os.path.join(instance_image_folder, f"{image_id}.png")

    cv2.imwrite(image_rgb_path, rgb)
    cv2.imwrite(image_semseg_path, semseg)
    cv2.imwrite(image_inst_path, inst)

    # Save JSON data

    json_data = json.dumps(output_data, indent=4)
    json_file_path = os.path.join(json_folder, f"{image_id}.json")

    with open(json_file_path, 'w') as json_file:
        json_file.write(json_data)

    # Create a copy of the RGB image with bounding boxes and labels
    image_with_boxes = rgb.copy()

    for item in output_data:
        min_x = item["min_x"]
        min_y = item["min_y"]
        max_x = item["max_x"]
        max_y = item["max_y"]
        label = item["base_label"]
        name= item["name"]

        # Get the color for the label from class_mapping
        color = class_mapping.get(label, (0, 0, 255))  # Default to red if not found

        # Draw the bounding box with the corresponding color
        cv2.rectangle(image_with_boxes, (min_x, min_y), (max_x, max_y), color, 2)

        # Write the label next to the bounding box
        cv2.putText(image_with_boxes, label+":"+name, (min_x, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save the image with bounding boxes and labels
    image_with_boxes_path = os.path.join(labeled_image_folder, f"{image_id}.png")
    cv2.imwrite(image_with_boxes_path, image_with_boxes)
# Example usage:
# process_images(rgb, semseg, inst, bbs, width, height, fov, location, rotation, folder_path, tick_id)
def read_json_file(json_filename):
    with open(json_filename, 'r') as json_file:
        data = json.load(json_file)
        camera_parameters = data['camera_parameters']
        camera_transform = data['camera_transform']
        filtered_objects = data['filtered_objects']
        return camera_parameters, camera_transform, filtered_objects

def process_batch(batch_folder, output_path):
    parent_folder = batch_folder

    # Choose one subfolder to get the file names (e.g., 'rgb')
    source_subfolder = 'rgb'

    # Get the file names from the source subfolder
    source_folder_path = os.path.join(parent_folder, source_subfolder)
    file_names = os.listdir(source_folder_path)



    # Iterate through the file names and process corresponding files in other subfolders
    for file_name in file_names:
        tid = file_name.split('_')[1].split('.')[0]

        # Construct the paths for other subfolders based on the id
        rgb_image_path = os.path.join(parent_folder, 'rgb', f'image_{tid}.png')
        semseg_image_path = os.path.join(parent_folder, 'semantic_segmentation', f'image_{tid}.png')
        inst_image_path = os.path.join(parent_folder, 'instance_segmentation', f'image_{tid}.png')
        json_file_path = os.path.join(parent_folder, 'simulation_objects', f'image_{tid}.json')
        
        # Read the images from disk
        rgb = cv2.imread(rgb_image_path)
        semseg = cv2.imread(semseg_image_path)
        inst = cv2.imread(inst_image_path)
        camera_parameters, camera_transform, filtered_objects = read_json_file(json_file_path)
        image_size_x = camera_parameters['image_size_x']
        image_size_y = camera_parameters['image_size_y']
        fov = camera_parameters['fov']
        sensor_tick = camera_parameters['sensor_tick']

        # Extract camera transform
        camera_location = camera_transform['location']
        camera_rotation = camera_transform['rotation']

       
        width = image_size_x  # Replace with your camera width
        height = image_size_y  # Replace with your camera height
        #fov = ...  # Replace with your camera field of view
        location =carla.Location(camera_location['x'], camera_location['y'],camera_location['z']) # Replace with your camera location
        rotation = carla.Rotation(camera_rotation['pitch'] ,camera_rotation['yaw'],camera_rotation['roll']) # Replace with your camera rotation
        process_images(rgb, semseg, inst, filtered_objects, width, height, fov, location, rotation, output_path, tid)
        # Now you can process these files as needed
        # Replace this with your processing logic
        #process_images(rgb_image_path, semseg_image_path, inst_image_path, json_file_path)
def main():
    # Sample input data
    rgb_image_path = 'rgb_image.png'
    semseg_image_path = 'semseg_image.png'
    inst_image_path = 'inst_image.png'

    # Read the images from disk
    rgb = cv2.imread(rgb_image_path)
    semseg = cv2.imread(semseg_image_path)
    inst = cv2.imread(inst_image_path)

    # Replace with the path to your JSON file
    json_filename = 'test_json_file.json'
    bbs={}
    # Read camera parameters and bounding boxes from the JSON file
    camera_parameters, camera_transform, filtered_objects = read_json_file(json_filename)
    for obj in filtered_objects:
        label="static"
        bb= obj['bounding_box']

        if label not in bbs:
            bbs[label] = []
        bbs[label].append(obj)



    # Extract camera parameters
    image_size_x = camera_parameters['image_size_x']
    image_size_y = camera_parameters['image_size_y']
    fov = camera_parameters['fov']
    sensor_tick = camera_parameters['sensor_tick']

    # Extract camera transform
    camera_location = camera_transform['location']
    camera_rotation = camera_transform['rotation']

   
    width = image_size_x  # Replace with your camera width
    height = image_size_y  # Replace with your camera height
    #fov = ...  # Replace with your camera field of view
    location =carla.Location(camera_location['x'], camera_location['y'],camera_location['z']) # Replace with your camera location
    rotation = carla.Rotation(camera_rotation['pitch'] ,camera_rotation['yaw'],camera_rotation['roll']) # Replace with your camera rotation
    folder_path = 'output_folder'  # Replace with the path to your output folder
    tick_id = 1  # Replace with the tick_id

    # Call the process_images function
    process_images(rgb, semseg, inst, bbs, width, height, fov, location, rotation, folder_path, tick_id)

if __name__ == "__main__":
    batch_folder="C:\\Users\\adaoud\\Nextcloud\\Multitrans\\Virtualization-Framework\\Python\\images"
    output_path="output_folder2"
    process_batch(batch_folder, output_path)