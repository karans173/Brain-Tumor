import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input

from config import IMAGE_WIDTH, IMAGE_HEIGHT, BATCH_SIZE

def create_dataframe(path):
    """Creates a pandas DataFrame of image paths and their labels."""
    class_paths = []
    classes = []
    for label in os.listdir(path):
        for file in os.listdir(os.path.join(path, label)):
            class_paths.append(os.path.join(path, label , file))
            classes.append(label)
    return pd.DataFrame({"class path":class_paths , "class":classes})

def data_generator(train_path,test_path):
    """
    Creates and configures the training, validation, and test data generators.
    """
    train_df = create_dataframe(train_path)
    test_df = create_dataframe(test_path)

    valid_df , test_df = train_test_split(test_df , test_size = 0.5 , random_state = 20 , stratify = test_df['class'])

    gen = ImageDataGenerator(
        preprocessing_function = preprocess_input,
        brightness_range = (0.9,1.1)
    )

    ts_gen = ImageDataGenerator(preprocessing_function = preprocess_input)

    target_size = (IMAGE_WIDTH,IMAGE_HEIGHT)
    
    train_gen = gen.flow_from_dataframe(
        train_df,
        x_col = 'class path',
        y_col = 'class',
        target_size = target_size,
        batch_size = BATCH_SIZE
    )
    valid_gen = ts_gen.flow_from_dataframe(
        valid_df ,
        x_col = 'class path',
        y_col = 'class',
        target_size = target_size,
        batch_size = BATCH_SIZE
    )
    test_gen = ts_gen.flow_from_dataframe(
        test_df,
        x_col = 'class path',
        y_col = 'class',
        target_size = target_size,
        batch_size = BATCH_SIZE,
        shuffle = False
    )

    print("Train data : ",len(train_gen))
    print("Test data : ",len(test_gen))
    print("Validation data : ",len(valid_gen))

    return train_gen , test_gen , valid_gen