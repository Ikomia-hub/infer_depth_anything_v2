<div align="center">
  <img src="images/depth_map.jpg" alt="Algorithm icon">
  <h1 align="center">infer_depth_anything_v2</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_depth_anything_v2">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_depth_anything_v2">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_depth_anything_v2/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_depth_anything_v2.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

Depth Anything V2, significantly outperforms V1, is a highly practical solution for robust monocular depth estimation.

![illustration depth](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/assets/teaser.png?raw=true)


## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow


```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_depth_anything_v2", auto_connect=True)

# Run directly on your image
wf.run_on(url="https://github.com/Ikomia-dev/notebooks/blob/main/examples/img/img_dog.png?raw=true")

# Display the results
display(algo.get_output(0).get_image()) # Colormap
display(algo.get_output(1).get_image()) # Grayscale
```

## :sunny: Use with Ikomia Studio
Ikomia Studio offers a friendly UI with the same features as the API.
- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).
- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters
- **model_name** (str) - default 'vits': Name of the ViT pre-trained model.
    - 'vits' ; Param: 24.8M
    - 'vitm' ; Param: 97.5M	
    - 'vitl' ; Param: 335.3M
- **input_size** (str) - default '640': Size of the input image.
- **cuda** (bool): If True, CUDA-based inference (GPU). If False, run on CPU.

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_depth_anything_v2", auto_connect=True)

algo.set_parameters({
        'model_name':'vits',
        'input_size': '640',
        'cuda': 'True'})

# Run directly on your image
wf.run_on(url="https://github.com/Ikomia-dev/notebooks/blob/main/examples/img/img_dog.png?raw=true")

# Display the results
display(algo.get_output(0).get_image())
display(algo.get_output(1).get_image())
```

## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_depth_anything", auto_connect=True)

# Run on your image  
wf.run_on(url="https://github.com/Ikomia-dev/notebooks/blob/main/examples/img/img_dog.png?raw=true")

# Iterate over outputs
for output in algo.get_outputs():
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```

