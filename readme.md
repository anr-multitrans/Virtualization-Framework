## GUI of the Virtualization Platform 
We built the graphical interface using PyQt5, which allows the user to interact with a simulation environment created with the CARLA Python API. The interface provides various functionalities such as selecting labels for city object detection, recording the simulation, and controlling the simulation's execution (start, pause, stop).

The interface displays three different images, RGB, segmented, and bounding box. The RGB image displays the simulation's environment as seen by a camera. The segmented image shows the environment with different objects separated into classes using different colors. The bounding box image shows the RGB image with bounding boxes around the selected city objects.

Additionally, the interface contains several buttons for adding, selecting, modifying, and removing events, as well as a timeline for displaying the recorded events. The user can select city object labels for detection using checkboxes provided in a group box.

## Requirements

-   Carla simulator
-   Python 3.x
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

6.  Open the file 'config.ini' in any text editor, under [Carla] entry, change the value of path to the path of CARLA installation directory 
    
7.  Run the following command to launch the framework:

`python Python/Editor_UI.py` 

Note: The script will check if CARLA simulator is running, if not it will pass a command to launch it and wait for its availability for connecting clients before launching the framework.

In case of any issues with the installation or running the framework, refer to the documentation or raise an issue on the Github repository.
