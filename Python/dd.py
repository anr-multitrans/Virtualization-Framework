import carla

client = carla.Client('localhost', 2000)
client.set_timeout(5.0)
world = client.load_world('Town11')