from flask import Flask, request, render_template
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        number_of_rows = int(request.form['number_of_rows'])

        df = pd.read_csv(file, sep=',', skiprows=1, encoding='latin1')
        random_row_indices = np.random.randint(0, df.shape[0], size=number_of_rows)
        random_col_indices = np.random.randint(0, df.shape[1], size=number_of_rows)
        df.iloc[random_row_indices, random_col_indices] = np.NaN
        return render_template('result.html', table=df.to_html())

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
