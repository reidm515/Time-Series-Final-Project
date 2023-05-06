import os
from flask import Flask, request, redirect, url_for, flash, send_from_directory, render_template, session

from werkzeug.utils import secure_filename
from XGBoost import xgboost_forecast
from LSTM import lstm_forecast
from arima import arima_forecast
from HWES import hwes_forecast
import base64
from io import BytesIO
import pandas as pd
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from sarimax import sarima_forecast
app = Flask(__name__)


app.config['UPLOAD_FOLDER'] = 'Hourly Datasets (CSV)/'
#this sets the directory for uploading files
app.secret_key = 'your_secret_key'
#this creates a secret key, which ensures secuirty on the clients side
from GapInsertion import annual_maintenance_gaps, random_gaps, weather_outage_gaps
#importing gap insertion methods
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///uploads.db'
db = SQLAlchemy(app)

#creates an instance of the SQLAlchemy database engine to store our user informaiion for later


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

class Upload(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(80), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user = db.relationship('User', backref=db.backref('uploads', lazy=True))
    uploaded_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

def create_database():
    with app.app_context():
        db.create_all()

def allowed_file(filename):# This checks if the uploaded file has the correct format
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv'}

# This is for the main page, handling file uploads

@app.route('/')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('base.html')
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'user_id' not in session:
            return redirect(url_for('login'))

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

            # Save the upload record with the user
            new_upload = Upload(filename=filename, user_id=session['user_id'])
            db.session.add(new_upload)
            db.session.commit()

            return redirect(url_for('select_column', filename=filename))
    return render_template('upload.html')



# This Route is for selecting a column and a model to use for forecasting
@app.route('/select_column/<filename>', methods=['GET', 'POST'])
def select_column(filename):
    if request.method == 'POST':
        column_name = request.form['column']
        column_name2 = request.form['column2']
        model = request.form['model']
        method = request.form['method']

        gap_count = int(request.form['gap_count'])
        gap_size = int(request.form['gap_size'])
        maintenance_duration = int(request.form['maintenance_duration'])
        num_outage_days = int(request.form['num_outage_days'])
        outage_duration = int(request.form['outage_duration'])
        col = request.form['col']

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        df = pd.read_csv(file_path, sep=',', skiprows=1, encoding='latin1')

        # Call the appropriate gap insertion function based on the selected method
        if method == "random_gaps":
            df_with_gaps = random_gaps(df, gap_count, gap_size)
        elif method == "annual_maintenance_gaps":
            df_with_gaps = annual_maintenance_gaps(df, maintenance_duration)
        elif method == "weather_outage_gaps":
            df_with_gaps = weather_outage_gaps(df, num_outage_days, outage_duration, col)

        # Save the gap-filled dataset
        gap_filled_filename = f"{filename[:-4]}_with_gaps.csv"
        df_with_gaps.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], gap_filled_filename), index=False)

        if model == "lstm":
            return redirect(url_for('lstm_result', filename=filename, gap_filled_filename=gap_filled_filename,
                                    column_name=column_name))
        elif model == "xgboost":
            return redirect(url_for('xgboost_result', filename=filename, gap_filled_filename=gap_filled_filename,
                                    column_name=column_name))
        elif model == "arima":
            return redirect(url_for('arima_result', filename=filename, gap_filled_filename=gap_filled_filename,
                                    column_name=column_name))
        elif model == "hwes":
            return redirect(url_for('hwes_result', filename=filename, gap_filled_filename=gap_filled_filename,
                                    column_name=column_name))
        elif model == "sarimax":
            return redirect(url_for('sarimax_result', filename=filename, gap_filled_filename=gap_filled_filename,
                                    column_name=column_name, column_name2=column_name2))

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(file_path, sep=',', skiprows=1, encoding='latin1')
    columns = list(df.columns)

    return render_template('select_column.html', filename=filename, columns=columns)


#route for the xgboost results page
@app.route('/xgboost_result/<filename>/<gap_filled_filename>/<column_name>')
def xgboost_result(filename, gap_filled_filename, column_name):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    gap_filled_file_path = os.path.join(app.config['UPLOAD_FOLDER'], gap_filled_filename)

    trained_model_xgboost, fig_xgboost, eval_results_xgboost = xgboost_forecast(file_path, gap_filled_file_path, column_name)
    final_rmse = eval_results_xgboost['validation_1']['rmse'][-1]

    return render_template('xgboost_result.html', plot_url_xgboost=fig_xgboost, final_rmse=final_rmse)



#route for the lsmt results pgae
@app.route('/lstm_result/<filename>/<gap_filled_filename>/<column_name>')
def lstm_result(filename, gap_filled_filename, column_name):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    gap_filled_file_path = os.path.join(app.config['UPLOAD_FOLDER'], gap_filled_filename)

    results1,results2, plot1, plot2 = lstm_forecast(file_path, gap_filled_file_path, column_name)


    return render_template('lstm_result.html', results1=results1,results2=results2, plot1=plot1, plot2=plot2)

@app.route('/arima_result/<filename>/<gap_filled_filename>/<column_name>')
def arima_result(filename, gap_filled_filename, column_name):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    gap_filled_file_path = os.path.join(app.config['UPLOAD_FOLDER'], gap_filled_filename)

    arima_results1, arima_results2, plot1, plot2, plot3 = arima_forecast(file_path, gap_filled_file_path, column_name)


    return render_template('arima_result.html', arima_results1=arima_results1, arima_results2=arima_results2, plot1=plot1, plot2=plot2, plot3=plot3)

@app.route('/hwes_result/<filename>/<gap_filled_filename>/<column_name>')
def hwes_result(filename, gap_filled_filename, column_name):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    gap_filled_file_path = os.path.join(app.config['UPLOAD_FOLDER'], gap_filled_filename)

    hwes_plot1, hwes_plot2, hwes_plot3, rmse, mae = hwes_forecast(file_path, gap_filled_file_path, column_name)


    return render_template('hwes_result.html', hwes_plot1=hwes_plot1, hwes_plot2=hwes_plot2, hwes_plot3 = hwes_plot3, rmse=rmse, mae=mae)

@app.route('/sarimax_result/<filename>/<gap_filled_filename>/<column_name>/<column_name2>')  # Add column_name2 here
def sarimax_result(filename, gap_filled_filename, column_name, column_name2):  # Add column_name2 as a parameter
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    gap_filled_file_path = os.path.join(app.config['UPLOAD_FOLDER'], gap_filled_filename)

    # Update the function call to include the second column
    plot = sarima_forecast(file_path, gap_filled_file_path, column_name, column_name2)  # Add column_name2 here

    return render_template('sarimax_result.html',plot=plot)




@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists. Please choose a different one.')
            return redirect(url_for('register'))
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        session['user_id'] = new_user.id
        return redirect(url_for('profile'))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            return redirect(url_for('profile'))
        else:
            flash('Invalid username or password.')
    return render_template('login.html')


@app.route('/profile')
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user = User.query.get(session['user_id'])
    return render_template('profile.html', user=user)


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))



if __name__ == '__main__':
    create_database()
    app.run(debug=True)

