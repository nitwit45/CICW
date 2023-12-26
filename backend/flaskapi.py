from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import pandas as pd
import pickle
from flask_cors import CORS
from sklearn.impute import SimpleImputer
from joblib import Parallel, delayed

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Function to load the quantity prediction model
def load_quantity_prediction_model():
    with open('xgboost_model_final.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    return loaded_model

# Function to process CSV for quantity prediction
def process_quantity_prediction(df, model):
    # Take only the first 10 rows of the DataFrame
    df_subset = df.head(20)

    # Make predictions using the loaded model for quantity prediction
    predictions = model.predict(df_subset)
    predictions = [round(pred) for pred in predictions]

    # Add a new column "Predicted Quantity" to the original DataFrame
    df.loc[df_subset.index, 'Predicted Quantity'] = predictions

    return df

# Function to load the department prediction model
def load_department_prediction_model():
    # Load the pre-trained model from the pickle file
    with open('model_svm.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model

# Function to predict a subset of data in parallel
def predict_subset(model, subset_data):
    return model.predict(subset_data)

def process_department_prediction_multithread(df, model, label_encoders, training_features):
    # Take only the first 10 rows of the DataFrame
    df_subset = df.head(20)

    # Extract the relevant features used during training
    prediction_features = df_subset[training_features].copy()

    # Apply label encoding to categorical columns, handling missing values
    for col, le in label_encoders.items():
        if col in prediction_features.columns:
            prediction_features[col].fillna('Unknown', inplace=True)
            prediction_features.loc[:, col] = le.transform(prediction_features[col])

    # Handle missing values (replace NaN with mean, you can customize the strategy)
    imputer = SimpleImputer(strategy='mean')
    prediction_features = pd.DataFrame(imputer.fit_transform(prediction_features), columns=prediction_features.columns)

    # Use multithreading to parallelize the prediction step
    predictions = model.predict(prediction_features)

    # Add a new column "Predicted_Department" to the original DataFrame
    df.loc[df_subset.index, 'Predicted_Department'] = predictions

    return df

# Function to integrate both predictions and update the original DataFrame
def integrate_predictions(df, quantity_model, department_model, label_encoders, training_features):
    # Process CSV for quantity prediction and obtain the DataFrame
    quantity_df = process_quantity_prediction(df.copy(), quantity_model)

    # Process DataFrame for department prediction using multithreading
    final_df = process_department_prediction_multithread(df.copy(), department_model, label_encoders, training_features)

    # Merge the results of quantity and department predictions with unique suffixes
    merged_df = pd.merge(quantity_df, final_df, how='inner', left_index=True, right_index=True, suffixes=('_quantity', '_department'))

    return merged_df

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        # Save the original file
        original_filename = secure_filename(file.filename)
        original_file_path = f"{app.config['UPLOAD_FOLDER']}/original_{original_filename}"
        file.save(original_file_path)

        # Load the quantity prediction model
        quantity_model = load_quantity_prediction_model()

        # Load the department prediction model
        department_model = load_department_prediction_model()

        # Load the original DataFrame
        original_df = pd.read_csv(original_file_path)

        # List of features used during training for department prediction
        training_features = ['Category Name', 'Customer Zipcode', 'Customer Street', 'Customer Id', 'Customer Segment',
                             'Order City', 'Order Country', 'Order Item Quantity', 'Order Customer Id', 'order date (DateOrders)',
                             'Order Id', 'Order Item Cardprod Id', 'Order Region', 'Order State', 'Product Card Id', 'Product Name',
                             'Product Price']

        # Assuming label_encoders is loaded from your file (label_encoders.pkl)
        with open('label_encoders.pkl', 'rb') as le_file:
            label_encoders = pickle.load(le_file)

        # Integrate both predictions and update the original DataFrame
        final_df = integrate_predictions(original_df, quantity_model, department_model, label_encoders, training_features)

        # Save the final CSV with both predictions
        final_file_path = f"{app.config['UPLOAD_FOLDER']}/final_predictions_original.csv"
        final_df.to_csv(final_file_path, index=False)

        # Send the final CSV back to the front end
        return send_file(final_file_path, as_attachment=True, download_name=f'final_predictions_original.csv')

    return jsonify({'error': 'Invalid file type'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
