import tensorflow as tf
from tensorflow.python.client import device_lib
tf.debugging.set_log_device_placement(True)



print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Create some tensors
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)
print(c)

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

print(tf.test.is_built_with_cuda())
tf.config.list_physical_devices('GPU') 
print(device_lib.list_local_devices())
