import numpy as np
import os
import cv2
import  tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from gevent.pywsgi import WSGIServer
from werkzeug.utils import secure_filename
global graph
graph=tf.Graph()
from flask import Flask ,request ,render_template
app = Flask(__name__)
model = load_model("cnn_handclassification.h5")

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        img = image.load_img(filepath,target_size = (64,64))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis =0)  
        index=['A','B','C','D','E','F','G','H','I']
        preds = np.argmax(model.predict(x),axis=1)
        p=index[preds[0]]
        print("prediction",p)
        text = "the predicted handsign text is : " + str(p)
        
    return text
if __name__ == '__main__':
    app.run(debug = True, threaded = False)
        
        
        
    
    
    