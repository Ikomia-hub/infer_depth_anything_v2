from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from infer_depth_anything_v2.infer_depth_anything_v2_process import InferDepthAnythingV2Param

# PyQt GUI framework
from PyQt5.QtWidgets import *


# --------------------
# - Class which implements widget associated with the algorithm
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class InferDepthAnythingV2Widget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferDepthAnythingV2Param()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()
        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)

        # Set widget layout
        self.set_layout(layout_ptr)

    def on_apply(self):
        # Apply button clicked slot

        # Get parameters from widget
        # Example : self.parameters.window_size = self.spin_window_size.value()

        # Send signal to launch the algorithm main function
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build algorithm widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class InferDepthAnythingV2WidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the algorithm name attribute -> it must be the same as the one declared in the algorithm factory class
        self.name = "infer_depth_anything_v2"

    def create(self, param):
        # Create widget object
        return InferDepthAnythingV2Widget(param, None)