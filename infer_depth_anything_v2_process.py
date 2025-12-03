import copy

import numpy as np
import torch

from ikomia import core, dataprocess, utils
import cv2

from infer_depth_anything_v2.ikutils import load_model


# --------------------
# - Class to handle the algorithm parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferDepthAnythingV2Param(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.model_name = "vits"
        self.input_size = 640
        self.cuda = torch.cuda.is_available()
        self.update = False

    def set_values(self, params):
        # Set parameters values from Ikomia Studio or API
        self.model_name = str(params["model_name"])
        self.input_size = int(params["input_size"])
        self.cuda = utils.strtobool(params["cuda"])
        self.update = True

    def get_values(self):
        # Send parameters values to Ikomia Studio or API
        # Create the specific dict structure (string container)
        params = {
            "model_name": str(self.model_name),
            "input_size": str(self.input_size),
            "cuda": str(self.cuda)
        }
        return params

# --------------------
# - Class which implements the algorithm
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferDepthAnythingV2(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        self.add_output(dataprocess.CImageIO())
        # Create parameters object
        if param is None:
            self.set_param_object(InferDepthAnythingV2Param())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.model = None
        self.device = torch.device("cpu")

    def get_progress_steps(self):
        # Function returning the number of progress steps for this algorithm
        # This is handled by the main progress bar of Ikomia Studio
        return 1

    def _load_model(self):
        param = self.get_param_object()
        self.device = torch.device("cuda" if param.cuda and torch.cuda.is_available() else 'cpu')
        self.model = load_model(name=param.model_name, device=self.device)
        param.update = False

    def init_long_process(self):
        self._load_model()
        super().init_long_process()

    def run(self):
        # Main function of your algorithm
        # Call begin_task_run() for initialization
        self.begin_task_run()

        # Get parameters :
        param = self.get_param_object()

        # Get input :
        img_input = self.get_input(0)

        # Get image from input/output (numpy array):
        src_image = img_input.get_image()

        # Load model
        if param.update:
            self._load_model()

        # Inference
        with torch.no_grad():
            depth = self.model.infer_image(src_image, self.device,  param.input_size)

        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
        depth_color_rgb = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)

        # Get output (image)
        output_inferno = self.get_output(0)
        output_inferno.set_image(depth_color_rgb)

        output_grayscale = self.get_output(1)
        output_grayscale.set_image(depth)

        # Step progress bar (Ikomia Studio):
        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()

# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferDepthAnythingV2Factory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set algorithm information/metadata here
        self.info.name = "infer_depth_anything_v2"
        self.info.short_description = "Depth Anything V2 is a highly practical solution for robust monocular depth estimation"
        # relative path -> as displayed in Ikomia Studio algorithm tree
        self.info.path = "Plugins/Python/Depth"
        self.info.version = "1.1.1"
        self.info.icon_path = "images/depth_map.jpg"
        self.info.authors = "Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, Hengshuang Zhao"
        self.info.article = "Depth Anything V2"
        self.info.journal = "arXiv:2406.09414"
        self.info.year = 2024
        self.info.license = "Apache License 2.0"
        # URL of documentation
        self.info.documentation_link = "https://arxiv.org/abs/2406.09414"
        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_depth_anything_v2"
        self.info.original_repository = "https://github.com/LiheYoung/Depth-Anything"
        # Python version
        self.info.min_python_version = "3.10.0"
        # Ikomia version
        self.info.min_ikomia_version = "0.15.0"
        # Keywords used for search
        self.info.keywords = "Depth Estimation, Pytorch, HuggingFace, map"
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "OTHER"
        # Min hardware config
        self.info.hardware_config.min_cpu = 4
        self.info.hardware_config.min_ram = 16
        self.info.hardware_config.gpu_required = False
        self.info.hardware_config.min_vram = 6

    def create(self, param=None):
        # Create algorithm object
        return InferDepthAnythingV2(self.info.name, param)
