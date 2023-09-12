from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QWidget, QVBoxLayout, QGroupBox, QFormLayout, \
    QComboBox, QListWidget, QPushButton, QHBoxLayout, QLabel, QRadioButton, QGridLayout
from PyQt5.QtCore import pyqtSignal
import json

class CornerCaseEditor():
    def __init__(self, INFO_path, parent):
        self.cornercase_form = None
        cornercase_data = self.load_cornercase_data(INFO_path)
        self.parent=parent
        # Get the list of categories from the loaded data
        self.categories = cornercase_data["categories"]

    def load_cornercase_data(self, file_path):
        with open(file_path, "r") as file:
            return json.load(file)

    def show_cornercase_form(self):
        # Create the corner case form if it doesn't exist
        if not self.cornercase_form:
            self.cornercase_form = CornerCaseForm(self.categories)
            self.cornercase_form.selected_category = self.parent.selected_category
            self.cornercase_form.selected_subcategory = self.parent.selected_subcategory
            self.cornercase_form.selected_example = self.parent.selected_example
            self.cornercase_form.selection_changed.connect(self.parent.update_info_label)
            self.parent.corner_case_form=self.cornercase_form

        # Show the corner case form
        self.cornercase_form.show()

class CornerCaseForm(QWidget):
    selection_changed = pyqtSignal(str, str, str)

    def __init__(self, categories):
        super().__init__()
        self.setWindowTitle("Corner Case Selector")
        self.setFixedSize(800, 600)  # Increased window size
        self.selected_category = "Scenario Level"
        self.selected_subcategory = "Anomalous Scenario"
        self.selected_example = "Person walking onto the street"
        self.categories = categories
        self.initUI()

    def initUI(self):
        # Create combo boxes for category, sub-category, and example selection
        self.category_combo = QComboBox()
        self.category_combo.currentIndexChanged.connect(self.show_subcategories)
        self.examples_list = QListWidget()
        self.subcategory_combo = QComboBox()
        self.subcategory_combo.currentIndexChanged.connect(self.show_examples)

        # Populate the category combo box
        for category in self.categories:
            self.category_combo.addItem(category["name"])

        # Set the initial selected category
        self.category_combo.setCurrentText(self.selected_category)

        # Populate the sub-category combo box
        self.show_subcategories(self.category_combo.currentIndex())

        # Set the initial selected sub-category
        self.subcategory_combo.setCurrentText(self.selected_subcategory)

        # Populate the examples list
        self.show_examples(self.subcategory_combo.currentIndex())

        # Create a button to confirm the selection
        self.confirm_button = QPushButton("Confirm")
        self.confirm_button.clicked.connect(self.confirm_selection)

        # Set up a layout for the form
        layout = QVBoxLayout()
        layout.setSpacing(20)  # Added spacing between widgets
        layout.addWidget(QGroupBox("Select Corner Case Category", self))
        layout.addWidget(self.category_combo)
        layout.addWidget(QGroupBox("Select Sub-Category", self))
        layout.addWidget(self.subcategory_combo)
        layout.addWidget(QGroupBox("Select Example", self))
        layout.addWidget(self.examples_list)
        layout.addWidget(self.confirm_button)
        self.setLayout(layout)

    def show_subcategories(self, index):
        # Clear the sub-category combo box
        self.subcategory_combo.clear()

        # Get the selected category
        selected_category = self.category_combo.currentText()

        # Find the selected category in the categories list
        category = next((c for c in self.categories if c["name"] == selected_category), None)

        # Populate the sub-category combo box based on the selected category
        if category:
            for subcategory in category.get("subcategories", []):
                self.subcategory_combo.addItem(subcategory["name"])

    def show_examples(self, index):
        # Clear the examples list
        self.examples_list.clear()

        # Get the selected sub-category
        selected_subcategory = self.subcategory_combo.currentText()
       

        # Find the selected category in the categories list
        selected_category = self.category_combo.currentText()

        # Find the selected category in the categories list
        category = next((c for c in self.categories if c["name"] == selected_category), None)
       
       
        # Find the selected sub-category in the category's subcategories list
        subcategory = next((s for s in category.get("subcategories", []) if s["name"] == selected_subcategory), None)
       
        # Populate the examples list based on the selected sub-category
        if subcategory:
            for example in subcategory.get("examples", []):
                if example.get("implemented", True):
                    self.examples_list.addItem(example["name"])

    def confirm_selection(self):
        # Get the selected category and sub-category
        category = self.category_combo.currentText()
        subcategory = self.subcategory_combo.currentText()

        # Check if an example is selected
        selected_example = self.examples_list.currentItem()
        example = ""
        if selected_example is not None:
            example = selected_example.text()

        # Emit the selection_changed signal with the selected values
        self.selection_changed.emit(category, subcategory, example)

        self.close()

