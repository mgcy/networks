import tensorflow as tf
y2=[2, 2, 4]
d2=tf.convert_to_tensor(y2)

conf=tf.confusion_matrix(
    labels=[1, 2, 4],
    predictions=d2,
    num_classes=5,
    dtype=tf.int32,
)
'''
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

test_confusion = sess.run(conf)
print(test_confusion)



print('Confusion Matrix: \n\n',
        tf.Tensor.eval(conf,
        feed_dict=None,
        session=None))
'''
sess = tf.Session()
with sess.as_default():
    print(conf.eval())
