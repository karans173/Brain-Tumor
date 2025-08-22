import os
from datetime import datetime
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import tensorflow as tf
from flask import Flask, request, render_template, url_for, flash, redirect
from werkzeug.utils import secure_filename

app = Flask(__name__)

app.secret_key = "this-is-a-super-secret-key-for-flash-messaging" 
UPLOAD_FOLDER = 'app/static/uploads/'
GRAD_CAM_FOLDER = 'app/static/grad_cam/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GRAD_CAM_FOLDER'] = GRAD_CAM_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRAD_CAM_FOLDER, exist_ok=True)

# --- Load The Model (runs only once at startup) ---
print("Loading Keras model...")
try:
    model = tf.keras.models.load_model('C:\\Brain Tumor\\model\\Model(new).keras')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- Define Global Variables ---
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict(image_path):
    if model is None: raise IOError("Model not loaded or failed to load.")
    image = Image.open(image_path)
    img_resized = image.resize((299, 299))
    img_array = np.array(img_resized)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = tf.keras.applications.xception.preprocess_input(img_batch)
    prediction_probs = model.predict(img_preprocessed)[0]
    predicted_class_index = np.argmax(prediction_probs)
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    probabilities = {CLASS_NAMES[i]: float(prediction_probs[i]) for i in range(len(CLASS_NAMES))}
    return predicted_class_name.capitalize(), probabilities

def get_grad_cam(image_path, last_conv_layer_name='block14_sepconv2_act'):
    if model is None: raise IOError("Model not loaded or failed to load.")
    image = Image.open(image_path)
    img_resized = image.resize((299, 299))
    img_array = np.array(img_resized)
    img_preprocessed = np.expand_dims(img_array.copy(), axis=0)
    img_preprocessed = tf.keras.applications.xception.preprocess_input(img_preprocessed)
    grad_model = tf.keras.models.Model(model.inputs, [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds_as_list = grad_model(img_preprocessed)
        preds = preds_as_list[0]
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (img_resized.width, img_resized.height))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img_array
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    grad_cam_pil = Image.fromarray(superimposed_img)
    filename = secure_filename(os.path.basename(image_path))
    grad_cam_path = os.path.join(app.config['GRAD_CAM_FOLDER'], f'gradcam_{filename}')
    grad_cam_pil.save(grad_cam_path)
    return grad_cam_path

def log_prediction(filename, prediction):
    LOG_FILE = 'log.csv'
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w') as f: f.write('timestamp,filename,prediction\n')
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, 'a') as f: f.write(f'{timestamp},{filename},{prediction}\n')

# --- Main Route ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part in the request.')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading.')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                prediction_result, probabilities = predict(filepath)
                grad_cam_path_full = get_grad_cam(filepath)
                log_prediction(filename, prediction_result)
                
                uploaded_image_path_html = os.path.join('uploads', filename).replace('\\', '/')
                grad_cam_path_html = os.path.join('grad_cam', os.path.basename(grad_cam_path_full)).replace('\\', '/')
                
                history = []
                if os.path.exists('log.csv'):
                    history_df = pd.read_csv('log.csv').tail(5).sort_index(ascending=False)
                    history = history_df.to_dict('records')

                return render_template('index.html',
                                       prediction=prediction_result,
                                       probabilities=probabilities,
                                       uploaded_image_path=uploaded_image_path_html,
                                       grad_cam_path=grad_cam_path_html,
                                       history=history)
            except Exception as e:
                print(f"An error occurred: {e}")
                flash(f'An error occurred during processing. Please try another image or check the logs. Error: {e}')
                return redirect(url_for('index'))

    history = []
    if os.path.exists('log.csv'):
        history_df = pd.read_csv('log.csv').tail(5).sort_index(ascending=False)
        history = history_df.to_dict('records')
    return render_template('index.html', history=history)

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True)