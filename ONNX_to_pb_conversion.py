# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 22:32:29 2020

@author: Aditya
"""

#RUN THIS ON GOOGLE COLAB

try:
  import onnx
  
except:
  !pip install onnx
  import onnx
try:
  from onnx_tf.backend import prepare
except: 
  !git clone https://github.com/onnx/onnx.git
  %cd onnx
  !git submodule update --init --recursive
  !pip install -e .
  !git clone https://github.com/onnx/onnx-tensorflow.git
  %cd onnx-tensorflow
  !pip install -e . 
  from onnx_tf.backend import prepare

model_onnx = onnx.load('/content/drive/My Drive/Colab Notebooks/Rootee/iris3_pupil_detector.onnx')
tf_rep = prepare(model_onnx)
# Export model as .pb file
tf_rep.export_graph('/content/drive/My Drive/Colab Notebooks/Rootee/iris_tf.pb')

