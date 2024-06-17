from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits PyDataProcess.CPluginProcessInterface from Ikomia API
# --------------------
class IkomiaPlugin(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def get_process_factory(self):
        # Instantiate algorithm object
        from infer_depth_anything_v2.infer_depth_anything_v2_process import InferDepthAnythingV2Factory
        return InferDepthAnythingV2Factory()

    def get_widget_factory(self):
        # Instantiate associated widget object
        from infer_depth_anything_v2.infer_depth_anything_v2_widget import InferDepthAnythingV2WidgetFactory
        return InferDepthAnythingV2WidgetFactory()
