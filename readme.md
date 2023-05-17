
# Installation Guide for CARLA Simulation Platform and the Virtualization Framework Editor

## Before you begin

Before installing CARLA, make sure your system meets the following requirements:

-   System requirements: CARLA is built for Windows and Linux systems.
-   An adequate GPU: CARLA aims for realistic simulations, so the server needs at least a 6 GB GPU, although we would recommend 8 GB. A dedicated GPU is highly recommended for machine learning.
-   Disk space: CARLA will use about 20 GB of space.
-   Python: Python is the main scripting language in CARLA. CARLA supports Python 2.7 and Python 3 on Linux, and Python 3 on Windows.
-   Pip: Some installation methods of the CARLA client library require pip or pip3 (depending on your Python version) version 20.3 or higher. To check your pip version:

  # 
  

    # For Python 3
     pip3 -V

    # For Python 2

    pip -V

 

If you need to upgrade:


    # For Python 3

    pip3 install --upgrade pip

    # For Python 2

    pip install --upgrade pip 

-   Two TCP ports and good internet connection: 2000 and 2001 by default. Make sure that these ports are not blocked by firewalls or any other applications.
-   Other requirements: CARLA requires some Python dependencies. Install the dependencies according to your operating system:

### Linux



`pip install --user pygame numpy &&
pip3 install --user pygame numpy` 

### Debian CARLA installation

The Debian package is available for both Ubuntu 18.04 and Ubuntu 20.04, however, the officially supported platform is Ubuntu 18.04.

1.  Set up the Debian repository in the system:

`sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 1AF1527DE64CB8D9`
`sudo add-apt-repository "deb [arch=amd64] http://dist.carla.org/carla $(lsb_release -sc) main" `

2.  Install CARLA and check for the installation in the /opt/ folder:

`sudo apt-get update # Update the Debian package index`
`sudo apt-get install carla-simulator # Install the latest CARLA version, or update the current installation`
`cd /opt/carla-simulator # Open the folder where CARLA is installed` 

3.  Import additional assets: Each release has its own additional package of extra assets and maps. This additional package includes the maps Town06, Town07, and Town10. These are stored separately to reduce the size of the build, so they can only be imported after the main package has been installed.
    -   Download the appropriate package for your desired version of CARLA.
    -   Extract the package:
        -   On Linux: Move the package to the Import folder and run the following script to extract the contents:


`cd path/to/carla/root
./ImportAssets.sh` 

Once you have followed these steps, you should be able to install and use CARLA on your system. If you encounter any issues, please refer to the official documentation or seek help from the community forums.


## Python API
CARLA Python API is a module that provides a simple interface to interact with the CARLA simulation platform using Python. The API allows users to perform various tasks such as controlling the simulation environment, receiving sensory information from vehicles and pedestrians, and manipulating the simulation objects, such as spawning actors or setting their attributes.

The CARLA Python API is built on top of the Python OpenDRIVE parser, which provides access to the road network data, as well as a set of Python scripts that can be used for various purposes, such as spawning objects or vehicles.

To use the CARLA Python API, you need to first import the carla module into your Python script. Then you can create a client object to connect to the CARLA server, which will allow you to communicate with the simulation environment.

Once connected, you can start interacting with the simulation environment using the client object. You can spawn actors, such as vehicles, pedestrians, or sensors, and control their attributes, such as position, speed, or acceleration. You can also access sensory data from the simulation environment, such as camera or lidar data.

In addition, you can control the simulation environment by setting various parameters such as weather conditions, traffic density, or time of day.

Overall, the CARLA Python API provides a powerful and flexible interface to interact with the CARLA simulation platform using Python. It allows users to build and test various autonomous driving algorithms and explore different scenarios in a simulated environment.
### Run Python scripts
To run a Python script that uses the CARLA Python API, follow these steps:

1.  Make sure you have installed the CARLA simulator and all the required dependencies as explained in the installation guide.
    
2.  Start the CARLA simulator by running the following command in a terminal:
    

     `./CarlaUE4.sh` 
    
   This will launch the CARLA simulator with the default settings.
    
3.   In a new terminal, navigate to the folder where your Python script is located.
    
4.   Make sure you have installed the `carla` Python package by running the following command:
    
        `pip3 install carla` 
    
   This will install the `carla` package that provides the Python API for communicating with the CARLA simulator.
    
5.  Run your Python script using the following command:
    
    `python3 your_script.py` 
    
    Replace `your_script.py` with the actual name of your Python script.
    
-   Your Python script should now connect to the CARLA simulator and start sending commands to control the simulated vehicles and retrieve data from the sensors.
    
    Here is an example script that connects to the simulator and spawns a vehicle:
    
 #   

		import carla
	 
		#Connect to the CARLA server
		client = carla.Client('localhost', 2000)
		client.set_timeout(10.0)

		# Get the world object
		world = client.get_world()

		# Spawn a vehicle
		blueprint_library = world.get_blueprint_library()
		vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
		spawn_point = carla.Transform(carla.Location(x=160.0, y=190.0, z=40.0), carla.Rotation())
		vehicle = world.spawn_actor(vehicle_bp, spawn_point)

		# Control the vehicle
		vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))

		# Wait for a few seconds
		time.sleep(5)

		# Destroy the vehicle
		vehicle.destroy() 

This script connects to the CARLA server, spawns a Tesla Model 3 vehicle at a specific location, and applies a throttle control command to make the vehicle move forward. After 5 seconds, the script destroys the vehicle and terminates the connection to the server.

You can modify this script or create your own script to interact with the CARLA simulator using the Python API.

CARLA provides several examples of how to use its Python API to interact with the simulator. These examples demonstrate various functionalities such as spawning actors, controlling them, and retrieving data from sensors.

To run these examples, you first need to have CARLA installed on your system. Once you have installed CARLA, you can find the Python examples in the `PythonAPI/examples` directory. There are several examples available, such as:

1.  `spawn_npc.py`: This example spawns non-player characters (NPCs) in the simulator.
2.  `manual_control.py`: This example shows how to control a vehicle manually using the keyboard.
3.  `spawn_pedestrians.py`: This example spawns pedestrians in the simulator.
4.  `lidar.py`: This example demonstrates how to use the LIDAR sensor to retrieve point cloud data.
5.  `camera.py`: This example demonstrates how to use the camera sensor to retrieve RGB images and semantic segmentation maps.

To run an example, you can navigate to the `PythonAPI/examples` directory and run the corresponding Python script using the following command:

php

`python3 <example_name>.py` 

Replace `<example_name>` with the name of the example you want to run.

Note that some examples require additional arguments to be passed to the script, such as the IP address and port number of the CARLA simulator. You can find information on the required arguments for each example in the comments at the top of the script.

Once you run an example, it will start the CARLA simulator and perform the actions specified in the script. You can modify the script to customize the actions performed in the simulator or to retrieve different types of data from the sensors.

## GUI of the Virtualization Platform 
We built the graphical interface using PyQt5, which allows the user to interact with a simulation environment created with the CARLA Python API. The interface provides various functionalities such as selecting labels for city object detection, recording the simulation, and controlling the simulation's execution (start, pause, stop).

The interface displays three different images, RGB, segmented, and bounding box. The RGB image displays the simulation's environment as seen by a camera. The segmented image shows the environment with different objects separated into classes using different colors. The bounding box image shows the RGB image with bounding boxes around the selected city objects.

Additionally, the interface contains several buttons for adding, selecting, modifying, and removing events, as well as a timeline for displaying the recorded events. The user can select city object labels for detection using checkboxes provided in a group box.

To run the GUI, follow the steps below:

1.  First, ensure that you have successfully installed Carla and can execute the provided examples.
    
2.  Next, copy the "python" folder to your disk. It is recommended to copy it to the same level as the "examples" and "carla" folders in the "PythonAPI" directory. This is to ensure that the Carla Python modules are accessible from that location.
    
3.  Open a terminal window and navigate to the "python" folder that you just created using the `cd` command. For example:

`cd /path/to/python` 

Replace "/path/to/python" with the actual path to the "Carla/PythonAPI/python" folder on your disk.

4.  Once you are in the "python" folder, run the GUI using the following command:

`python Editor_UI.py` 

This should launch the GUI and you should be able to use it to create scenarios in Carla. If you encounter any errors, ensure that Carla is properly installed and that you have copied the "python" folder to the correct location.



