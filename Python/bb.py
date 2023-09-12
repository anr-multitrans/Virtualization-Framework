import sys
from PyQt5.QtWidgets import QApplication, QAction, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, QPushButton, QFileDialog, QLabel, QHBoxLayout, QComboBox, QGridLayout, QCheckBox, QGroupBox, QLineEdit, QProgressBar,QRadioButton,QMessageBox
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices, QIcon
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
#from PyQt5.QtCore import AspectRatioMode
# imprt libraries

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.scenario_tick = 0
        self.progress_updated = pyqtSignal(int)
        self.scenario_name = ''
        self.scenario_folder = ''
        self.global_imabe_error = 0
        self.camera_tick = 0
        self.ThreeD = True
        self.ImageCounter=0
        self.synchro_list = []
        self.selected_labels=selected_labels
        self.bb_labels= bb_labels
        self.is_running= False
        self.is_recording = True
        self.is_step_by_step=False
        self.desired_width = 1280
        self.desired_height = 1024
        self.sensor_width=1280
        self.sesor_height=1024
        self.selected_category = "Scenario Level"
        self.selected_subcategory = "Anomalous Scenario"
        self.selected_example = "Example"
        self.cornercase_form= None
        self.CornerCaseEditor= CornerCaseEditor('CC_terminology.json', self )
        self.is_corner_case=False
        self.spawned_actors=[]
        self.multiple_bbox_tags_colors  = []
        self.client= None
        self.init_ui()
        
        # Initialize the CARLA client and world
        #self.init_carla_client()
        
        # Adjust the graphics and world settings
        #self.adjust_carla_settings()
        
        # Initiate bounding_box_labels
        #self.init_bounding_box_labels()

        # Set the weather parameters
        #self.set_weather_parameters()
        #self.K = self.build_projection_matrix(self.sensor_width, self.sesor_height, 90)
        #self.init_carla_scenario()


    def init_ui(self):
        
        
        self.folder_button = QPushButton('Change output directory')
        self.generate_Scenario_name()
        
        # Set up the main window
        self.setGeometry(100, 100, 800, 600)
        


        ########

        # Set up the layout for the main window
        self.setup_layout(output_layout, main_bb_vbox,main_cc_vbox, progress_layout, recording_widget, palette_widget,group_box, info_layout)

    def setup_layout(self,output_layout, main_bb_vbox,main_cc_vbox, progress_layout, recording_widget, palette_widget,group_box, info_layout):
        central_widget = QWidget()
        #####
        #####
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
    def init_carla_client(self):
        self.client=check_carla_server()
        if self.client== None:
            print("Starting Carla Server")
            t = threading.Thread(target=launch_carla_server)
            t.start()

        while self.client == None:
            self.client = check_carla_server()
            time.sleep(1)

        
    def adjust_carla_settings(self):
        pass
    def init_carla_scenario(self):
        pass

    def setup_bb_radio_buttons(self):
        self.radio2D = QRadioButton('2D')
        self.radio3D = QRadioButton('3D')
        #self.radio3 = QRadioButton('Radio 3')

        # Set radio1 as the default selected button
        self.radio3D.setChecked(True)

        # Create a group box and add the radio buttons to it
        group_Radio_box = QGroupBox("Bounding boxes mode")
        vbox = QVBoxLayout()
        vbox.addWidget(self.radio2D)
        vbox.addWidget(self.radio3D)
        self.radio2D.toggled.connect(self.on_radio2D_toggled)
        self.radio3D.toggled.connect(self.on_radio3D_toggled)
        
        group_Radio_box.setLayout(vbox)
        # Create a layout and add the group box to it
        main_bb_vbox = QVBoxLayout()
        main_bb_vbox.addWidget(group_Radio_box)


        return main_bb_vbox
    def toggle_scenario_selection(self, checked):
        if checked:
            self.info_label.setText("Normal Driving random scenarios")
            self.modify_button.setEnabled(False)
            self.is_corner_case=False
            self.cornercase_form.close()
            self.scenario_length=20000
        else:
            #self.info_label.setText("Normal scenarios")
            self.is_corner_case=True
            self.update_info_label(self.selected_category, self.selected_subcategory, self.selected_example)
            self.modify_button.setEnabled(True)
            self.scenario_length=200

    def update_info_label(self, category, subcategory, example):
        self.selected_category = category
        self.selected_subcategory = subcategory
        selected_example = example
        self.selected_example = selected_example if selected_example else ""
        self.info_label.setText(f"Category: {category}\nSub-Category: {subcategory}\nExample: {example}")


    def setup_scenario_radio_buttons(self):
        ###


        return main_scenario_vbox
    def on_radio2D_toggled(self, checked):
        if checked:
            self.ThreeD=False
            # Perform actions specific to the 2D mode

    def on_radio3D_toggled(self, checked):
        if checked:
            self.ThreeD=True
            # Perform actions specific to the 3D mode
    def setup_menu_bar(self):
        ##
        scenario_menu.addAction(self.cornercase_action)
        return menu_bar

    def setup_palette(self):
        ####
        return output_layout, progress_layout, recording_widget, palette_widget, info_layout

    def select_folder(self):
        # open the file dialog and get the selected folder path
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Folder', os.path.expanduser('~'))
        
        # update the value of the textbox with the selected folder path
        if folder_path:
            self.scenario_folder= folder_path
            self.textbox.setText(self.scenario_folder)
    def generate_Scenario_name(self):
        now = datetime.datetime.now()
        self.scenario_name = "Scenario_" + now.strftime("%Y-%m-%d_%H-%M-%S")
        self.scenario_folder= self.create_directory(self.scenario_name)
        self.setWindowTitle("MultiTrans Virtualization Framework | " +  self.scenario_folder)
        self.folder_button.setEnabled(True)

        print("new scenario folder is generated")
        print(self.scenario_folder)
    def start_scenario(self):
        self.pause_action.setEnabled(True)
        self.stop_action.setEnabled(True)
        self.start_action.setEnabled(False)
        self.folder_button.setEnabled(False)
        self.init_carla_client()
        self.is_running=True
        #settings=self.world.get_settings()
        #settings.synchronous_mode = True
        #self.world.apply_settings(settings)

        #if self.is_corner_case:
        #    self.execute_cornercase_scenario()
        #else:
            # Create a stop event
        #    stop_event = threading.Event()
            # Create a new thread
        #    thread = threading.Thread(target=self.run_script, args=(stop_event,))

            # Start the thread
            #thread.start()
        
        
        #self.progress_updated.connect(self.update_progress)

    def stop_scenario(self):
        self.pause_action.setEnabled(False)
        self.stop_action.setEnabled(False)
        self.start_action.setEnabled(True)
        self.is_running=False
        #settings=self.world.get_settings()
        #settings.synchronous_mode = False
        #self.world.apply_settings(settings)
        self.write_data_to_disk()
        self.generate_Scenario_name()
        self.start_action.setText('Start')
        self.scenario_tick=0
        for _, actor in self.spawned_actors:
            actor.destroy()

    def pause_scenario(self):
        self.pause_action.setEnabled(False)
        self.stop_action.setEnabled(True)
        self.start_action.setEnabled(True)
        self.start_action.setText('Resume')
        self.is_running=False
    def record_scenario(self):
        if self.record_action.text()=='Record':
            self.record_action.setText('Stop recording')
            self.rec_btn.setIcon(QIcon("icons/ON.png"))
        else:
            self.record_action.setText('Record')
            self.rec_btn.setIcon(QIcon("icons/OFF.png"))
        self.is_recording= not self.is_recording
    def create_directory(self, scenario_name):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        exec_path= os.path.join(script_dir,'execution')
        dir_path = os.path.join(exec_path, scenario_name)
        os.makedirs(dir_path, exist_ok=True)
        return os.path.abspath(dir_path)

    def show_cornercase_form(self):
        self.CornerCaseEditor.show_cornercase_form()

    def write_data_to_disk(self):
        # Iterate over the data list and write each item to disk
        print('writing output to disk')

if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)
        main_window = MainWindow()
        main_window.show()
        sys.exit(app.exec_())
    except carla.ClientError as e:
        print("Connection failed:", e)

