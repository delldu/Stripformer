# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2024(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, Sat 13 Apr 2024 11:37:58 AM CST
# ***
# ************************************************************************************/
#


__version__ = "1.0.0"

import os
from tqdm import tqdm
import torch
import todos

from . import stripformer

import pdb


def get_deblur_model():
    """Create model."""
    device = todos.model.get_device()
    model = stripformer.Stripformer()
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    # make sure model good for C/C++
    model = torch.jit.script(model)
    # https://github.com/pytorch/pytorch/issues/52286
    torch._C._jit_set_profiling_executor(False)
    # C++ Reference
    # torch::jit::getProfilingMode() = false;                                                                                                             
    # torch::jit::setTensorExprFuserEnabled(false);

    todos.data.mkdir("output")
    if not os.path.exists("output/image_deblur.torch"):
        model.save("output/image_deblur.torch")

    return model, device


def deblur_predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_deblur_model()

    # load files
    image_filenames = todos.data.load_files(input_files)

    # start predict
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        # orig input
        input_tensor = todos.data.load_tensor(filename)
        # pytorch recommand clone.detach instead of torch.Tensor(input_tensor)
        orig_tensor = input_tensor.clone().detach()

        predict_tensor = todos.model.forward(model, device, input_tensor)

        output_file = f"{output_dir}/{os.path.basename(filename)}"

        todos.data.save_tensor([orig_tensor, predict_tensor], output_file)
    todos.model.reset_device()
