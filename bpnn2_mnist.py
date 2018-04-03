"""BPNN Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def bpnn_model_fn(features, labels, mode):

  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 784])

  # Layer #1
  bpnn1 = tf.layers.dense(inputs=input_layer,units=784,activation=tf.nn.sigmoid)

  # Layer #2
  bpnn2 = tf.layers.dense(inputs=bpnn1,units=30,activation=tf.nn.sigmoid)
  
  # Layer #3
  logits = tf.layers.dense(inputs=bpnn2,units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
  
  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.01,momentum=0.9)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  # Load training and eval data
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images[:6000]  # Returns np.array
  train_labels = np.asarray(mnist.train.labels[:6000], dtype=np.int32)
  eval_data = mnist.test.images[:1000]  # Returns np.array
  eval_labels = np.asarray(mnist.test.labels[:1000], dtype=np.int32)

  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
      model_fn=bpnn_model_fn, model_dir="/tmp/mnist_bpnn_model")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=1,
      num_epochs=None,
      shuffle=True)
  mnist_classifier.train(
      input_fn=train_input_fn,
      steps=20000,
      hooks=[logging_hook])

  # Evaluate the model
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)

  # Predict output
  predictions = mnist_classifier.predict(input_fn=eval_input_fn)
  output_labels=[]
  for p in predictions:
    output_labels.append(p['classes'])

   # Print results
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)
  
  # Confusion matrix
  conf=tf.confusion_matrix(
    labels=eval_labels,
    predictions=output_labels,
    num_classes=10,
    dtype=tf.int32,
    name=None,
    weights=None
  )
  print(conf)
  with tf.Session():
    print('Confusion Matrix: \n\n',
         tf.Tensor.eval(conf,feed_dict=None, session=None))

if __name__ == "__main__":
  tf.app.run()
