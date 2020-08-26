from keras.models import model_from_json
import numpy as np

def change_model(model, new_input_shape=(None, 40, 40, 3)):
    # replace input shape of first layer
    model._layers[1].batch_input_shape = new_input_shape

    # feel free to modify additional parameters of other layers, for example...


    # rebuild model architecture by exporting and importing via json
    new_model = model_from_json(model.to_json())
    new_model.summary()

    # copy weights from old model to new one
    for layer in new_model.layers:
        try:
            layer.set_weights(model.get_layer(name=layer.name).get_weights())
        except:
            print("Could not transfer weights for layer {}".format(layer.name))

    # test new model on a random input image
    """
    X = np.random.rand(10, 40, 40, 3)
    y_pred = new_model.predict(X)
    print(y_pred)
    """

    return new_model