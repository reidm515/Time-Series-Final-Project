# Importing necessary modules
from flask import Flask, render_template, request, flash, Response
import pandas as pd


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def upload_csv():
    if request.method == 'POST':
        # If POST request is made and Checking if file is in the request or not
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        # this read and displays the file if it selected by user
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            #The function uses pandas read_csv function to read the CSV file into a pandas DataFrame.
            #It skips the first row and sets the delimiter to a comma.
            df = pd.read_csv(file, sep=',', skiprows=1, encoding='latin1')
            # shows the number of lines requsted by user and selects it
            num_lines = request.form['number-of-lines']
            df_display = df.head(int(num_lines))
            # Returning lines as plain text rather than html
            return Response(df_display.to_csv(index=False), mimetype='text/plain')
    return render_template('index.html')

# This initiates the Flask app
if __name__ == '__main__':
    app.run(debug=True)