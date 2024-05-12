import tensorflow as tf
import tensorflow.compat.v1 as tf1

def convert_tf1_to_tf2(checkpoint_path, output_prefix):
    """Converts a TF1 checkpoint to TF2.

    To load the converted checkpoint, you must build a dictionary that maps
    variable names to variable objects.
    ```
    ckpt = tf.train.Checkpoint(vars={name: variable})  
    ckpt.restore(converted_ckpt_path)
    ```

    Args:
      checkpoint_path: Path to the TF1 checkpoint.
      output_prefix: Path prefix to the converted checkpoint.

    Returns:
      Path to the converted checkpoint.
    """
    vars = {}
    reader = tf.train.load_checkpoint(checkpoint_path)
    dtypes = reader.get_variable_to_dtype_map()
    for key in dtypes.keys():
      vars[key] = tf.Variable(reader.get_tensor(key))
    return tf.train.Checkpoint(vars=vars).save(output_prefix)


converted_path = convert_tf1_to_tf2('r50_1x_sk0', 
                                     'tf2')
