import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up the main window
        self.setWindowTitle("Example Interface")
        self.setGeometry(100, 100, 400, 300)

        # Create a central widget for the window
        central_widget = QWidget(self)

        # Add a label to the window
        label = QLabel("This is an example interface", central_widget)

        # Add a button to start/stop execution
        button_execute = QPushButton("Start/Stop Execution", central_widget)

        # Add a button to start/stop image sequence recording
        button_record = QPushButton("Start/Stop Recording", central_widget)

        # Add a button to import a scenario
        button_import = QPushButton("Import Scenario", central_widget)

        # Add a button to export the current scenario
        button_export = QPushButton("Export Scenario", central_widget)

        # Add the buttons and label to a vertical layout
        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(button_execute)
        layout.addWidget(button_record)
        layout.addWidget(button_import)
        layout.addWidget(button_export)

        # Set the layout for the central widget
        central_widget.setLayout(layout)

        # Set the central widget for the window
        self.setCentralWidget(central_widget)

# Start the application
app = QApplication(sys.argv)

# Create the main window and show it
window = MainWindow()
window.show()

# Run the application
sys.exit(app.exec_())
