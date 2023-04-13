import sys
from PyQt5.QtWidgets import QApplication, QWidget, QCheckBox, QGroupBox, QVBoxLayout

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Create a QGroupBox widget
        group_box = QGroupBox('City Object Labels')

        self.selected_labels = []
        # Create the dictionary of labels and checkboxes
        self.bb_labels = {
            'Any': False,
            'Bicycle': False,
            'Bridge': False,
            'Buildings': False,
            'Bus': False,
            'Car': False,
            'Dynamic': False,
            'Fences': False,
            'Ground': False,
            'GuardRail': False,
            'Motorcycle': False,
            'NONE': False,
            'Other': False,
            'Pedestrians': False,
            'Poles': False,
            'RailTrack': False,
            'Rider': False,
            'RoadLines': False,
            'Roads': False,
            'Sidewalks': False,
            'Sky': False,
            'Static': False,
            'Terrain': False,
            'traffic_light': False,
            'traffic_sign': False,
            'Train': False,
            'Truck': False,
            'Vegetation': False,
            'Walls': False,
            'Water': False
        }

        # Create the checkboxes and add them to the group box
        checkbox_layout = QVBoxLayout()
        for label, value in self.bb_labels.items():
            checkbox = QCheckBox(label)
            checkbox.setChecked(value)
            checkbox.stateChanged.connect(lambda state, label=label: self.checkbox_state_changed(state, label))
            checkbox_layout.addWidget(checkbox)
        group_box.setLayout(checkbox_layout)

        # Create a layout for the window and add the group box to it
        layout = QVBoxLayout()
        layout.addWidget(group_box)
        self.setLayout(layout)

    def checkbox_state_changed(self, state, label):
        # Update the value of the corresponding label in the dictionary
        if state == 2:
            self.selected_labels.append(label)
        elif state == 0:
            self.selected_labels.remove(label)
        print(self.selected_labels)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())
