import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QComboBox, QPushButton, QGridLayout, QWidget, QAction

class ScenarioEditor(QWidget):
    def __init__(self):
        super().__init__()

        self.select_button = QPushButton("Select")
        self.add_button = QPushButton("Add")
        self.remove_button = QPushButton("Remove")
        self.modify_button = QPushButton("Modify")
        self.timeline_button = QPushButton("Timeline")
        self.new_event_button = QPushButton("New Event")

        layout = QGridLayout()
        layout.addWidget(self.select_button, 0, 0)
        layout.addWidget(self.add_button, 0, 1)
        layout.addWidget(self.remove_button, 0, 2)
        layout.addWidget(self.modify_button, 1, 0)
        layout.addWidget(self.timeline_button, 1, 1)
        layout.addWidget(self.new_event_button, 1, 2)

        self.setLayout(layout)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.sensor_label = QLabel("Sensor Output")
        self.sensor_view = QLabel("Real-time view of sensor output")
        self.scenario_label = QLabel("Scenario")
        self.scenario_combo = QComboBox()
        self.import_action = QAction("Import")
        self.export_action = QAction("Export")
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.record_button = QPushButton("Record")
        self.editor = ScenarioEditor()

        self.menu = self.menuBar().addMenu("Scenario")
        self.menu.addAction(self.import_action)
        self.menu.addAction(self.export_action)

        layout = QGridLayout()
        layout.addWidget(self.sensor_label, 0, 0)
        layout.addWidget(self.sensor_view, 1, 0)
        layout.addWidget(self.scenario_label, 2, 0)
        layout.addWidget(self.scenario_combo, 3, 0)
        layout.addWidget(self.start_button, 4, 0)
        layout.addWidget(self.stop_button, 4, 1)
        layout.addWidget(self.record_button, 4, 2)
        layout.addWidget(self.editor, 5, 0, 1, 3)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
