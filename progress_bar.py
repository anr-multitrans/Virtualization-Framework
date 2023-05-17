from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QProgressBar, QVBoxLayout
from PyQt5.QtCore import QThread, pyqtSignal
import time
import socket
import subprocess
import platform
import threading
import configparser
import os
import carla

class Worker(QThread):
    progress_updated = pyqtSignal(int)

    def run(self):
        for i in range(101):
            self.progress_updated.emit(i)
            time.sleep(0.1)

class Example(QWidget):
    def __init__(self):
        super().__init__()

        # create a progress bar widget
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)

        # create a button to start the progress bar
        self.start_button = QPushButton('Start')
        self.start_button.clicked.connect(self.start_progress)

        # create a layout for the progress bar and button
        layout = QVBoxLayout()
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.start_button)

        # set the main layout of the window
        self.setLayout(layout)

        # set the window properties
        self.setGeometry(100, 100, 300, 100)
        self.setWindowTitle('Progress Bar')
        self.show()

    def start_progress(self):
        # create a worker thread to update the progress bar
        self.worker = Worker()
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.start()

    def update_progress(self, value):
        # update the value of the progress bar
        self.progress_bar.setValue(value)

config = configparser.ConfigParser()
config.read('config.ini')
carla_path = config.get('Carla', 'path')
def check_carla_server():
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)
        client.get_world()
        return True
    except Exception as e:
        print(f"Failed to connect to Carla server: {e}")
        return False


def launch_carla_server():
    # launch the Carla server
    os_name = platform.system()
    print(f"Opening CARLA from path: {os.path.join(carla_path, 'CarlaUE4.exe')}")
    if os_name == 'Windows':
        subprocess.Popen([r"C:\CARLA\latest\CarlaUE4.exe"], cwd=carla_path)
    elif os_name == 'Linux':
        subprocess.Popen(['CarlaUE4.sh', '-opengl'], cwd=carla_path)
    else:
        print('Unsupported operating system')


if __name__ == '__main__':
    if not check_carla_server():
        print('Starting Carla server...')
        t = threading.Thread(target=launch_carla_server)
        t.start()

        while not check_carla_server():
            time.sleep(1)

    print('Launching PyQT UI...')
    app = QApplication([])
    ex = Example()
    app.exec_()