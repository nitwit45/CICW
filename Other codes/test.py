import pickle
import pandas as pd
import numpy as np
import sklearn
# Load the model
with open('random_forest_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

df_test = pd.read_csv('test_dataset.csv')

# Make predictions using the loaded model
new_predictions = loaded_model.predict(df_test)

# Make predictions and print each prediction on a new line
for index, row in df_test.iterrows():
    # Assuming your model expects a 2D array, reshape the row
    # prediction = loaded_model.predict(row.values.reshape(1, -1))
    print(f'Prediction for row {index}: {new_predictions[index]}')