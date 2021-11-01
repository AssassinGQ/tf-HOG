import tensorflow as tf
from hog import HOG
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def main():
    model = tf.saved_model.load("./r50x1_1")

    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec({224, 224}, tf.float32))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="./frozen_models",
                      name="resnet.pb",
                      as_text=False)



    # converter = tf.lite.TFLiteConverter.from_saved_model('saved_models')
    # tflite_model = converter.convert()
    #
    # with open("attention.tflite", 'wb') as f:
    #     f.write(tflite_model)

    print("Success generate model")

if __name__ == "__main__":
    main()
