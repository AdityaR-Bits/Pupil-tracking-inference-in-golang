# Pupil-tracking-inference-in-golang

PyTorch_to_onnx.py : Convert from PyTorch checkpoints to ONNX file

ONNX_to_pb_conversion.py : Convert from ONNX to .pb tensorflow file

TF_readgraph_inference : Use .pb file to make prediction based on any image input

tfinference.go : Converts .pb file to graph and performs inference on any jpg or png image while resizing to 224x224

tfinference2.go : Converts .pb file to graph and performs inference on any jpg or png image while cropping to 224x224

