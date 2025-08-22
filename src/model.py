import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from config import INPUT_SHAPE, NUM_CLASSES, LEARNING_RATE

def build_model():
    """
    Builds the brain tumor classification model using the Functional API.
    """
    inputs = Input(shape = INPUT_SHAPE)

    print("Building model ...........")

    base_model = Xception(
        include_top = False ,
        weights = "imagenet" ,
        input_tensor = inputs ,
        pooling = 'max'
    )

    x = base_model.output
    x = Dropout(rate = 0.3)(x)
    x = Dense(128,activation = 'relu')(x)
    x = Dropout(rate = 0.3)(x)
    outputs = Dense(NUM_CLASSES,activation = 'softmax')(x)

    model = Model(inputs = inputs , outputs = outputs)

    model.compile(
        optimizer = Adam(learning_rate = LEARNING_RATE),
        loss = 'categorical_crossentropy',
        metrics = ['accuracy','precision','recall']
    )

    print("Model built and compiled successfully.....")
    model.summary()

    return model
