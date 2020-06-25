# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 22:38:25 2020

@author: Aditya
"""

import tensorflow as tf
def load_pb(path_to_pb):
    with tf.io.gfile.GFile(path_to_pb, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph
tf_graph = load_pb('/content/drive/My Drive/Colab Notebooks/Rootee/pupil_tf.pb')
sess = tf.compat.v1.Session(graph=tf_graph)

output_tensor = tf_graph.get_tensor_by_name('output:0')
input_tensor = tf_graph.get_tensor_by_name('input:0')

print(output_tensor)
print(input_tensor)

import tensorflow as tf
img = tf.io.read_file('/content/drive/My Drive/Colab Notebooks/Rootee/pictu.png')
img = tf.image.decode_jpeg(img, channels=3)
img = tf.image.convert_image_dtype(img, tf.float32)
img=tf.expand_dims(img,0)
#img= tf.ones([1, 3, 224, 224], tf.float32)
out = tf.transpose(img, [0, 3, 1, 2])
out= tf.constant(out).numpy()
out.shape
print(out)

sample= img
output = sess.run(output_tensor, feed_dict={input_tensor: sample})

print(output)
