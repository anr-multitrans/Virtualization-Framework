
# Installation Guide for CARLA Simulation Platform

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
## Conclusion

Once you have followed these steps, you should be able to install and use CARLA on your system. If you encounter any issues, please refer to the official documentation or seek help from the community forums.
