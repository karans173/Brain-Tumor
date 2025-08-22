import matplotlib.pyplot as plt
import tensorflow as tf

from config import TRAIN_PATH, TEST_PATH, EPOCHS, MODEL_SAVE_PATH
from data_preprocessing import create_data_generators
from model import build_model

def plot_history(history):
    """Plots the training and validation accuracy and loss."""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def main():
    """Main function to run the training pipeline."""
    train_gen, valid_gen, test_gen = create_data_generators(TRAIN_PATH, TEST_PATH)
    
    model = build_model()
    
    print("\nStarting model training...")
    history = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=EPOCHS,
        verbose=1
    )
    print("Model training finished.")
    
    # Step 4: Evaluate the model
    print("\nEvaluating model on the test set...")
    test_loss, test_accuracy = model.evaluate(test_gen, verbose=1)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Step 5: Save the trained model
    print(f"\nSaving model to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH)
    print("Model saved successfully.")
    
    # Step 6: Plot training history
    print("\nPlotting training history...")
    plot_history(history)

if __name__ == '__main__':
    main()