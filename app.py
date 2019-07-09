import os
from flask import Flask, request, redirect, url_for,render_template
from werkzeug.utils import secure_filename
from predictor import prediction
import cv2
import glob
from tensorflow.python.platform import gfile
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['jpg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=os.path.join(app.config['UPLOAD_FOLDER'], filename)))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''




@app.route('/<filename>')
def uploaded_file(filename):
    files = glob.glob('./static/*')
    for f in files:
        os.remove(f)
    ans = prediction(filename)
    if type(ans).__name__ == 'str':
        return '''
        <!doctype html>
        <title>No Face</title>
        <h1>NO FACE FOUND</h1>
        <form>
        <input type="button" value="Go back!" onclick="history.back()">
        </form>
        
        '''
    else:
        filename = glob.glob('./uploads/*.jpg')
        for f in filename:
            os.remove(f)
        result = filename[0].split('\\')[-1].split('.')[0]
        img,score = ans[0],ans[1]
        for i in range(len(img)):
            cv2.imwrite('./static/'+result+'_'+score[i]+'_'+str(i+1)+'.jpg',img[i])
        imgs = glob.glob('./static/*.jpg')
        data = {}
        for i in range(len(imgs)):
            data[result+'_'+score[i]+'_'+str(i+1)+'.jpg'] = score[i]
        return render_template('upload.html',data=data)
        
if __name__ == "__main__":
    app.run(debug = True)
