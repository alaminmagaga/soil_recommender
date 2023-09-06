import os
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Define the target size for the images
target_size = (220, 220)  # Adjust this size as needed

# Define the rescaling factor
rescale_factor = 1.0 / 255.0  # Normalize pixel values to the range [0, 1]

# Define a dictionary to map class indices to soil names and recommended crops
soil_info = {
    0: {'name': 'Black Soil', 'crops': ['Cotton', 'Soybean', 'Wheat']},
    1: {'name': 'Cinder Soil', 'crops': ['Tomato', 'Cabbage', 'Pepper']},
    2: {'name': 'Laterite Soil', 'crops': ['Coffee', 'Cashew', 'Rubber']},
    3: {'name': 'Peat Soil', 'crops': ['Cranberries', 'Blueberries', 'Cucumbers']},
    4: {'name': 'Yellow Soil', 'crops': ['Peanuts', 'Sorghum', 'Millets']}
}

# Load your model here (replace 'model.h5' with your actual model file)
from tensorflow.keras.models import load_model
model = load_model('my_model.h5')

@app.route('/', methods=['GET', 'POST'])
def classify_soil():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            # Save the uploaded file
            file_path = os.path.join('uploads', uploaded_file.filename)
            uploaded_file.save(file_path)

            # Load the image and apply rescaling
            img = image.load_img(file_path, target_size=target_size)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x *= rescale_factor  # Apply rescaling
            images = np.vstack([x])

            # Perform prediction with your model
            predictions = model.predict(images)

            # Get the predicted class index
            predicted_class_index = np.argmax(predictions, axis=1)[0]

            # Look up the soil name and recommended crops using the dictionary
            soil_info_dict = soil_info.get(predicted_class_index, {'name': 'Unknown Soil', 'crops': []})

            # Remove the uploaded file
            os.remove(file_path)

            return render_template('result.html', soil_name=soil_info_dict['name'], recommended_crops=soil_info_dict['crops'])

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
