import carla

# Connect to the Carla simulator
#client = carla.Client('localhost', 2000)
#client.set_timeout(10.0)

city_object_labels = [attr for attr in dir(carla.CityObjectLabel)]
for label in city_object_labels:
	print("'"+label+"' :"+ " carla.CityObjectLabel."+label+",")
print(city_object_labels)

# Get the list of available CityObjectLabel types
# Get the list of available CityObjectLabel types
#city_object_labels = [city_object_label.name for city_object_label in list(carla.CityObjectLabel)]

# Print the CityObjectLabel types
#print("Available CityObjectLabel types:")
#for label in city_object_labels:
#    print(label.name)