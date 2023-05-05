import os
from flask import Flask, request, redirect, url_for, flash, send_from_directory, render_template
from werkzeug.utils import secure_filename
from XGBoost import xgboost_forecast
from LSTM import lstm_forecast
import base64
from io import BytesIO
import pandas as pd


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Hourly Datasets (CSV)/'
app.secret_key = 'your_secret_key'


def allowed_file(filename):# This checks if the uploaded file has the correct format
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv'}

# This is for the main page, handling file uploads
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('select_column', filename=filename))
    return render_template('upload.html')



# This Route is for selecting a column and a model to use for forecasting
@app.route('/select_column/<filename>', methods=['GET', 'POST'])
def select_column(filename):
    if request.method == 'POST':
        column_name = request.form['column']
        model = request.form['model']
        if model == "lstm":
            #redirects to corresponding page depending on model selected
            return redirect(url_for('lstm_result', filename=filename, column_name=column_name))
        elif model == "xgboost":
            return redirect(url_for('xgboost_result', filename=filename, column_name=column_name))

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)#read in the uploaded data and create a selection of columns
    df = pd.read_csv(file_path, sep=',', skiprows=1, encoding='latin1')
    columns = list(df.columns)
    return render_template('select_column.html', filename=filename, columns=columns)



#route for the xgboost results page
@app.route('/xgboost_result/<filename>/<column_name>')
def xgboost_result(filename, column_name):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # This runs the imported xgboost model and returns back the results in text form, and the graphs in png image form
    trained_model_xgboost, fig_xgboost, eval_results_xgboost = xgboost_forecast(file_path, column_name)
    final_rmse = eval_results_xgboost['validation_1']['rmse'][-1]

    return render_template('xgboost_result.html', plot_url_xgboost=fig_xgboost, final_rmse=final_rmse)
    #returns the data to our xgboost html


#route for the lsmt results pgae
@app.route('/lstm_result/<filename>/<column_name>')
def lstm_result(filename, column_name):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # This runs the imported lstm model and returns back the results in text form, and the graphs in png image form
    eval_results_lstm, fig_lstm, fig2_lstm = lstm_forecast(file_path, column_name)
    final_rmse = eval_results_lstm['root_mean_squared_error'][-1]


    return render_template('lstm_result.html', plot_url_lstm=fig_lstm, final_rmse=final_rmse, fig2_lstm=fig2_lstm)
    # returns the data to our lstm html


if __name__ == '__main__':
    app.run(debug=True)
