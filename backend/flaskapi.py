from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import pandas as pd
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_csv(file_path, filename, model):
    df = pd.read_csv(file_path)

    # Make predictions using the loaded model
    # Replace this with your actual prediction logic
    predictions = model.predict(df)

    # Add a new column "Predicted Quantity"
    df['Predicted Quantity'] = predictions

    # Check if "Department Country" and "Order Country" are the same
    df['DepCountry=OrderCountry'] = df.apply(lambda row: 'Yes' if row['Department country'] == row['Order Country'] else 'No', axis=1)

    # Save the CSV with the new columns
    processed_file_path = f"{app.config['UPLOAD_FOLDER']}/processed_{filename}"
    df.to_csv(processed_file_path, index=False)

    return processed_file_path

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = f"{app.config['UPLOAD_FOLDER']}/{filename}"
        file.save(file_path)

        # Load the model
        with open('random_forest_model.pkl', 'rb') as model_file:
            loaded_model = pickle.load(model_file)

        # Process CSV, add new columns, and save the file
        processed_file_path = process_csv(file_path, filename, loaded_model)

        # Remove the processed CSV file
        # No need to create a ZIP file, send the CSV directly
        return send_file(processed_file_path, as_attachment=True, download_name=f'processed_{filename}')

    return jsonify({'error': 'Invalid file type'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
