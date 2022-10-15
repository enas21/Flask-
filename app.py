from flask import Flask , render_template , request , url_for , redirect , flash , session
import cv2
from keras.models import load_model
import os
from werkzeug.utils import secure_filename
import numpy as np
import pickle
from collections import deque


UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'enas mostafa'
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
model_path = "assets/7wheat_deseases_Resent50_model.h5"
moodel = load_model(model_path)

@app.route("/")
def hom():
    return render_template("Home.html")

@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/Home")
def Home():
    return render_template("Home.html")

@app.route("/News")
def News():
    return render_template("News.html")

@app.route("/About")
def About():
    return render_template("About.html")

@app.route("/Diseases")
def Diseases():
    return render_template("Diseases.html")

@app.route("/predict" , methods=['POST'])
def predict():
    if 'img' not in request.files:
            flash('No file part')
            return redirect(request.url)
    file = request.files['img']
        # if user does not select file, browser also
        # submit a empty part without filename
    if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
    if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    input = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    session['uploaded_img_file_path'] = filename
    lb = pickle.loads(open("label", "rb").read())
    mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
    Q = deque(maxlen=128)
    vs = cv2.VideoCapture(input)
    (W, H) = (None, None)
    while True:
        (grabbed, frame) = vs.read()
        if not grabbed:
            break
        if W is None or H is None:
            (H, W) = frame.shape[:2]
        output = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224)).astype("float32")
        frame -= mean
        preds = moodel.predict(np.expand_dims(frame, axis=0))[0]
        Q.append(preds)
        results = np.array(Q).mean(axis=0)
        i = np.argmax(results)
        label = lb.classes_[i]
        text = "PREDICTION: {}".format(label.upper())
        cv2.putText(output, text, (4, 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.25, (200, 255, 155), 2)
        # show the output image
        image = cv2.resize(output, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        cv2.imwrite('static/uploads/output'+filename,image)
        #display_output = 'static/uploads/output'+filename
        key = cv2.waitKey(10) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    vs.release()
    return  render_template("index.html" ,prediction_text= text  )


@app.route('/display')
def display():
    filename = session.get('uploaded_img_file_path', None)
    return redirect(url_for('static', filename='uploads/output' + filename))



if __name__ =="__main__":
    app.run(debug=False)