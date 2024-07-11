# way to upload image: endpoint
# way to save the image
# function to make prediction on the image
# show the results
import os
import cv2
from flask import Flask
from flask import request
from flask import render_template
#from tensorflow.keras.models import load_model
from random import randint
from tensorflow.keras.models import load_model

app = Flask(__name__)
UPLOAD_FOLDER = "static/"

model = load_model("model.h5")




    
@app.route("/", methods=["GET", "POST"])
def upload_predict():
    import numpy as np
    class_names = [
                'Basale',
               'Betel',
               
               'Curry', 
               
               'Jasmine',
               
               'Lemon',
               'Mango',
               
               'Neem',
               
               'Peepal',
              ]
    if request.method == "POST":
        from keras.models import load_model  
        from PIL import Image, ImageOps
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(UPLOAD_FOLDER, image_file.filename)
            image_file.save(image_location)
            class_names = open("labels.txt", "r").readlines()


            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

            
            image = Image.open(image_file).convert("RGB")

            size = (224, 224)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

            
            image_array = np.asarray(image)

            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

            data[0] = normalized_image_array

            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]

            print("Class:", class_name[2:], end="")
            print("Confidence Score:", confidence_score)

            tt=''
            
            return render_template("result.html", prediction=class_name, img_path=image_location,treatment=tt)
    return render_template("index.html")
    
if __name__ == "__main__":
    app.run(port=12000, debug=True)
    