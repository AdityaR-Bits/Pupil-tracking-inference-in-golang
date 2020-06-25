# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 22:42:44 2020

@author: Aditya
"""

#RUN ON COLAB

import torch
checkpoint = "/content/drive/My Drive/Colab Notebooks/Rootee/final5_checkpoint_pupil_detector_score.pth.tar"
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)

x = torch.randn(1, 3, 224, 224, requires_grad=True).to(device)
torch_out = model(x)

# Export the model
torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "/content/drive/My Drive/Colab Notebooks/Rootee/iris3_pupil_detector.onnx", # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  #opset_version=10,          # the ONNX version to export the model to
                  #do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output']) # the model's output names
                  #dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                #'output' : {0 : 'batch_size'}})
                                
try:
  import onnx
  
except:
  !pip install onnx
  import onnx
onnx_model = onnx.load("/content/drive/My Drive/Colab Notebooks/Rootee/iris3_pupil_detector.onnx")

try:
  import onnxruntime
except:
  !pip install onnxruntime
  import onnxruntime

ort_session = onnxruntime.InferenceSession("/content/drive/My Drive/Colab Notebooks/Rootee/iris2_pupil_detector.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")