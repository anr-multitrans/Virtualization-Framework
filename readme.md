## GUI of the Virtualization Platform 
We built the graphical interface using PyQt5, which allows the user to interact with a simulation environment created with the CARLA Python API. The interface provides various functionalities such as selecting labels for city object detection, recording the simulation, and controlling the simulation's execution (start, pause, stop).

The interface displays three different images, RGB, segmented, and bounding box. The RGB image displays the simulation's environment as seen by a camera. The segmented image shows the environment with different objects separated into classes using different colors. The bounding box image shows the RGB image with bounding boxes around the selected city objects.

Additionally, the interface contains several buttons for adding, selecting, modifying, and removing events, as well as a timeline for displaying the recorded events. The user can select city object labels for detection using checkboxes provided in a group box.

## Requirements

-   Carla simulator (version 0.9.14 or later)
-   Python 3.x (tested on 3.7.9)
-   PyQt5
-   carla
-   keyboard
-   matplotlib
-   numpy
-   opencv-python
-   psutil


## Installation

1.  Install Python 3.x if it is not already installed. You can download the latest version from the official website: [https://www.python.org/downloads/](https://www.python.org/downloads/)
    
2.  Install CARLA simulator by following the instructions on their official website: [https://carla.readthedocs.io/en/latest/start_quickstart/](https://carla.readthedocs.io/en/latest/start_quickstart/)

3.  Clone the repository:

`git clone https://github.com/anr-multitrans/Virtualization-Framework.git` 

4.  Open the terminal and navigate to the directory where you cloned the repository.

5.  Install required packages using pip:
    
``pip install -r requirements.txt`` 

   If you face problems in installing PyQt5 try installing it using apt

   ``sudo apt install python3-pyqt5`` 
  
  for other requirements please refer to their documentations.
  Then to make sure all requirements are installed please repeate the ``pip install -r requirements.txt`` 

6.  Navigate using the terminal to the sub-directory "Python":

`` cd Python ``

7.  Open the file 'config.ini' in any text editor, under [Carla] entry, change the value of path to the path of CARLA installation directory 
    
8.  Run the following command to launch the framework:

`python Editor_UI.py` 

Note: The script will check if CARLA simulator is running, if not it will pass a command to launch it and wait for its availability for connecting clients before launching the framework.

In case of any issues with the installation or running the framework, refer to the documentation or raise an issue on the Github repository.


## Dataset generator (command line script)

The python script `generator.py` serves the purpose  generating datasets by data collection from simulation in  carla without dealing with GUIs, it takes one optional argument `config_file` which specifies the path of a .json file describing the simulatiopn scenario parameters, when no argument is provided the default parametters are loaded from the file `simulation_config.json`
to tun the script just navigate to 'Virtualization-Framework/Python' and in terminal run the following command:

``python generator.py [config_file_path]``

The script will excutes as follows:

### Initializing Carla Connection:
The script attempts to establish a connection to the Carla simulator, ensuring a successful initiation of the connection.

### Directory Creation:
It generates several directories to store diverse data types, including RGB images, semantic segmentation images, instance segmentation images, simulation object data, and labels.

### Loading Simulation Configuration:
The script reads a simulation configuration from the specified file (config_file). This configuration file holds essential parameters for the simulation, such as the number of images to capture and the distribution range.

### Connecting to Carla Simulator:
The script connects to the Carla simulator and loads a specific map, if one is specified within the configuration.

### Setting Carla Simulator Parameters:
It defines various settings for the Carla simulator, including synchronous mode and substep settings to ensure precise simulation control.

### Spawning Vehicles and Sensors:
The script spawns a vehicle and attaches multiple sensors, such as RGB cameras and other relevant sensors. These sensors capture critical data during the simulation.

### Simulating the Scenario:
The script enters a loop to simulate a scenario for a predefined number of iterations (scenario_length). During each iteration, it captures data from the sensors, encompassing RGB, semantic segmentation, and instance segmentation images.

### Data Processing:
The script processes the captured data, which includes object detection and extraction of bounding boxes. This processed data is saved to the specified directories.

### Handling Object Destruction and Simulation Settings:
It manages the destruction of spawned objects and manipulates settings to guarantee the correct execution of the simulation.

### Completing the Simulation:
After the simulation concludes, the ego vehicle is removed, and the Carla simulator's settings are reset to their default values.

## Configuration File (config_file) Structure:

The config_file is a JSON configuration file that outlines the parameters for the simulation. It includes the following elements:

    "map" (optional): Specifies the background world map for the simulation. This field allows you to change the map used in the simulation.

    "objects": Defines the set of objects appearing in every scene. These objects are spawned randomly within and around the visual field of the ego vehicle. The object categories and quantities are specified as follows:
        "vehicle": Number of vehicles randomly selected from a variety of Carla blueprints (excluding motorcycles or bicycles).
        "pedestrian": Number of pedestrians randomly selected from a variety of Carla blueprints.
        "cyclist": Number of cyclists randomly selected from a variety of Carla blueprints.
        "motorcycle": Number of motorcycles randomly selected from a variety of Carla blueprints.
        "specific": Used to specify a set of objects based on a pattern appearing in their blueprint ID. Longer patterns make the objects more specific. This field includes objects like cones, dirt, and barriers, each with a specified quantity.
        "other": Number of other non-specific random objects that may appear in the scene.

    "distribution_range": Specifies the range within which objects are spawned. Objects are spawned within a square located just in front of the vehicle, with a side length equal to the value of distribution_range.

    "nb_images": Indicates the number of iterations. Each iteration constructs a new scene and captures an image per sensor.

