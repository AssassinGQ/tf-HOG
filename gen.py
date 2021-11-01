import tensorflow as tf
from hog import HOG
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def load_model():
    inputs = tf.keras.Input(shape = (32,32,3))
    outputs = HOG()(inputs)
    model = tf.keras.Model(inputs, outputs)
    return model

def load_model2():
    # inputs = tf.keras.Input(shape = (16, 256))
    inputs = tf.keras.Input(shape = (16384, 24576))
    # outputs = tf.keras.layers.Dense(256)(inputs)
    d1 = tf.keras.layers.Dense(24576)(inputs)
    d2 = tf.keras.layers.Dense(24576)(d1)
    # d3 = tf.keras.layers.Dense(24576)(d2)
    # d4 = tf.keras.layers.Dense(24576)(d3)v
    outputs = tf.keras.layers.Dense(24576)(d1)
    model = tf.keras.Model(inputs, outputs)
    return model

def main():
    model = load_model2()
    model.save('saved_models')

    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="./frozen_models",
                      name="attention.pb",
                      as_text=False)



    converter = tf.lite.TFLiteConverter.from_saved_model('saved_models')
    tflite_model = converter.convert()

    with open("attention.tflite", 'wb') as f:
        f.write(tflite_model)

    print("Success generate model")

if __name__ == "__main__":
    main()
