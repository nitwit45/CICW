from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import pandas as pd
import pickle
from flask_cors import CORS
from sklearn.impute import SimpleImputer
import csv
import pandas as pd
import random
from deap import base, creator, tools, algorithms
import mpu as mpu
import matplotlib.pyplot as plt
import os

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
    df_subset = df.head(30)

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
    df_subset = df.head(30)

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

# Function to check and split data based on "Department Country" and "Order Country"
def split_data_by_country(df):
    # Extract rows where "Department Country" and "Order Country" are the same
    same_country_df = df[df['Department country'] == df['Order Country']]

    # Extract rows where "Department Country" and "Order Country" are different
    different_country_df = df[df['Department country'] != df['Order Country']]

    return same_country_df, different_country_df

# Function to integrate both predictions and update the original DataFrame
def integrate_predictions_and_split_by_country(df, quantity_model, department_model, label_encoders, training_features):
    # Process CSV for quantity prediction and obtain the DataFrame
    quantity_df = process_quantity_prediction(df.copy(), quantity_model)

    # Process DataFrame for department prediction using multithreading
    final_df = process_department_prediction_multithread(df.copy(), department_model, label_encoders, training_features)

    # Merge the results of quantity and department predictions with unique suffixes
    merged_df = pd.merge(quantity_df, final_df, how='inner', left_index=True, right_index=True)

    # Split data based on "Department Country" and "Order Country"
    same_country_df, different_country_df = split_data_by_country(df)

    # Save the split data into separate CSV files with blank header
    same_country_file_path = f"{app.config['UPLOAD_FOLDER']}/same_country_data.csv"
    different_country_file_path = f"{app.config['UPLOAD_FOLDER']}/different_country_data.csv"

    same_country_df.to_csv(same_country_file_path, index=False, header=False)
    different_country_df.to_csv(different_country_file_path, index=False, header=False)

    # Save the final CSV with both predictions
    final_file_path = f"{app.config['UPLOAD_FOLDER']}/final_predictions_original.csv"
    merged_df.to_csv(final_file_path, index=False)

    # Send the final CSV back to the front end
    return send_file(final_file_path, as_attachment=True, download_name=f'final_predictions_original.csv')

# Function to upload file and process predictions

def perform_multi_objective_optimization(department_coordinates_csv, order_data_csv, output_csv):
    usps_zones = {
        'Zone 1': (1, 50),
        'Zone 2': (51, 150),
        'Zone 3': (151, 300),
        'Zone 4': (301, 600),
        'Zone 5': (601, 1000),
        'Zone 6': (1001, 1400),
        'Zone 7': (1401, 1800),
        'Zone 8': (1801, float('inf')),
        'Zone 9': None  # US territories
    }

    # Define the dictionary mapping weight to the product category
    categories_by_weight = {
        8: ['Crafts', 'DVDs', 'CDs', 'Books', 'Garden', 'Music'],
        12: ['Sporting Goods','Camping & Hiking', 'Tennis & Racquet', 'Lacrosse', 'Water Sports', 'Indoor/Outdoor Games'],
        1: ['Electronics', 'Cameras', 'Computers', 'Health and Beauty', 'Video Games'],
        2: ['Cleats', "Women's Apparel", "Kids' Golf Clubs", 'Baseball & Softball', 'Soccer', 'Accessories',
            "Girls' Apparel", "Women's Clothing", "Men's Clothing", 'Fitness Accessories', 'Golf Balls', 'Golf Gloves'],
        3: ['Cardio Equipment', "Men's Footwear", 'As Seen on TV!', 'Strength Training', 'Baby', 'Fishing', 'Toys'],
        4: ['Basketball', 'Golf Bags & Carts', "Women's Golf Clubs", "Men's Golf Clubs"],
        5: ['Trade-In', 'Hockey'],
        10: ['Golf Shoes'],
        20: ['Boxing & MMA', 'Consumer Electronics', 'Pet Supplies'],
        40: ['Golf Apparel'],
        60: ['Hunting & Shooting', 'Golf Carts'],
        70: [None]
    }

    # Define the shipping rates data
    shipping_rates_data = {
        'Weight (oz)': [8, 12, 1, 2, 3, 4, 5, 10, 20, 40, 60, 70],
        'Zone 1': [5.40, 6.15, 7.60, 8.50, 8.85, 9.55, 10.20, 12.70, 18.20, 37.65, 46.65, 53.25],
        'Zone 2': [5.50, 6.25, 7.75, 9.00, 9.50, 10.00, 10.65, 13.00, 18.40, 37.70, 46.75, 53.35],
        'Zone 3': [5.55, 6.30, 7.85, 9.55, 9.95, 10.70, 11.40, 13.70, 19.60, 46.75, 55.65, 58.95],
        'Zone 4': [5.60, 6.35, 8.00, 10.25, 10.80, 11.65, 12.45, 15.45, 21.90, 59.70, 70.20, 81.45],
        'Zone 5': [5.65, 6.40, 8.15, 11.00, 11.80, 12.85, 13.75, 18.15, 28.15, 70.90, 92.85, 96.65],
        'Zone 6': [5.70, 6.45, 8.25, 11.80, 12.90, 14.30, 21.55, 21.85, 35.25, 85.25, 109.15, 118.55],
        'Zone 7': [5.75, 6.55, 8.40, 12.90, 16.35, 17.65, 26.25, 26.55, 44.40, 99.25, 124.95, 139.95],
        'Zone 8': [5.85, 6.65, 8.55, 14.90, 17.65, 19.00, 29.35, 31.45, 55.50, 113.65, 141.20, 161.75],
        'Zone 9': [5.85, 6.65, 8.55, 14.90, 17.65, 19.00, 29.35, 31.45, 55.50, 113.65, 141.20, 161.75]
    }

    def read_csv(file_path):
        places = []

        with open(file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            
            # Skip the header row
            next(csv_reader, None)

            for row in csv_reader:
                place_name = row[0].strip()

                for i in range(1, len(row)):
                    department_id = f"department id:{i}"
                    coordinates_str = row[i].strip()

                    if coordinates_str:  # Check if coordinates are not empty
                        coordinates = eval(coordinates_str)  # Convert coordinates string to tuple
                        place_info = (department_id, place_name, coordinates)
                        places.append(place_info)

        return places

    def filter_by_department_id(places, department_id):
        filtered_records = []

        for place_info in places:
            if place_info[0] == f"department id:{department_id}":
                location_name = place_info[1]
                coordinates = place_info[2]
                formatted_record = (location_name, coordinates)
                filtered_records.append(formatted_record)

        return filtered_records

    csv_file_path = 'department_coordinates.csv' 
    all_places = read_csv(csv_file_path)

    # Example: Get records for department id 2
    department_id_to_filter = 2
    filtered_records = filter_by_department_id(all_places, department_id_to_filter)
    places=filtered_records
    # for record in filtered_records:
    #     print(record)

    places = filtered_records
    print(places)


    order_data = pd.read_csv(order_data_csv)
    order_data = order_data[['Order Item Id', 'Customer Segment', 'Order Item Quantity',
                             'order_latitude', 'order_longitude', 'Product Price',
                             'Order Item Discount Rate', 'Shipping Mode', 'Department Id',
                             'Category Name', 'Type']]
    order_data = order_data.head(30)

    # ...

    # Constants for weightages
    discount_above_200_weightage = 0.1
    discount_below_200_penalty = 0.1
    DEBIT_payment_weightage = 0.4
    bank_transfer_payment_weightage = 0.6
    home_office_weightage = 0.3
    corporate_weightage = 0.4
    consumer_weightage = 0.3

    # Lead time optimization
    inventory_threshold = 100
    order_volume_increase_percentage = 0.1
    DEBIT_payment_lead_time_weightage = 0.6
    bank_transfer_payment_lead_time_weightage = 0.4

    # Function to shuffle the order of places in an individual
    def shuffle_order():
        places_indices = list(range(len(places)))
        random.shuffle(places_indices)
        PlaceIndex = places_indices[0]
        shipping_mode = random.choice(['Standard Class', 'Second Class', 'Same Day'])
        shipping_days = random.randint(1, 10)
        
        return [PlaceIndex,shipping_mode, shipping_days]

    # ...
    # Define DEAP types and fitness function
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0,1.0))  # Minimize both distance and cost
    creator.create("Individual", list, fitness=creator.FitnessMulti)
    
    # Create a DEAP toolbox
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, shuffle_order)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # ... 
    def distance_coor(coord1, coord2):
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        return mpu.haversine_distance((lat1, lon1), (lat2, lon2))

    # def determine_usps_zone(best_distance, usps_zones):
    #     for zone, (lower_bound, upper_bound) in usps_zones.items():
    #         if upper_bound is None or lower_bound <= best_distance <= upper_bound:
    #             return zone

    # ... 

    def calculate_shipping_mode_cost(shipping_mode):
        cost_map = {'Standard Class': 1, 'Second Class': 0.8, 'Same Day': 0.5}
        return cost_map.get(shipping_mode, 1)

    # ... 

    def evaluate(individual):
        total_cost, total_profit, total_lead_time = 0, 0, 0
        preparation_time = 1

        # ... (Continue with the rest of the code)

        def calculate_shipping_cost(product_category, best_distance,order_volume,shipping_mode):
            weight_category = next(
                key for key, value in categories_by_weight.items() if product_category in value
            )
            # print(weight_category)
            # weight_oz = shipping_rates_data['Weight (oz)'][weight_category]
            # print(weight_oz)

            zone = determine_usps_zone(best_distance, usps_zones)
            # print(zone)
            # shipping_rate = shipping_rates_data[weight_category][zone]
            shipping_rate=shipping_rates_data[zone][shipping_rates_data['Weight (oz)'].index(weight_category)]
            shipping_weightage_mode = calculate_shipping_mode_cost(shipping_mode)
            shipping_cost = order_volume * weight_category * shipping_rate * shipping_weightage_mode
            return shipping_cost

        def determine_usps_zone(best_distance, usps_zones):
            for zone, (lower_bound, upper_bound) in usps_zones.items():
                if upper_bound is None or lower_bound <= best_distance <= upper_bound:
                            return zone
        # Iterate over each order
        for i, order in enumerate(order_info_list):
            # print(f"Processing order {i + 1}")
            # Extract relevant information from the individual array
            place_indices = individual[0]
            shipping_mode = individual[1]
            shipping_days = individual[2]

            place_coords = places[place_indices][1]
            order_coords = (float(order['order_latitude']), float(order['order_longitude']))
            best_distance = round(distance_coor(order_coords, place_coords), 2)
            # print(best_distance)
            order_volume = order['Order Item Quantity']
            product_category=order['Category Name']
            # print(order_volume,product_category)

            shipping_cost = calculate_shipping_cost(str(product_category), best_distance,order_volume,shipping_mode)
            # print(f'shipping:{shipping_cost}')
            # Calculate shipping cost
            # shipping_cost = calculate_shipping_cost(order['Category Name'], best_distance)
            # total_cost += shipping_cost

            # Consider order volume for lead time optimization
            if order_volume > inventory_threshold:
                preparation_time *= (1 + order_volume_increase_percentage)
            

            # Consider payment method for lead time optimization
            if order['Type'] == 'DEBIT':
                total_lead_time += preparation_time * DEBIT_payment_lead_time_weightage
            else:
                total_lead_time += preparation_time * bank_transfer_payment_lead_time_weightage

            # Discount weightage based on quantity
            if order_volume > 200 and order['Order Item Discount Rate'] > 0:
                discount_weightage = discount_above_200_weightage
            else:
                discount_weightage = -discount_below_200_penalty

            # Payment method weightage
            if order['Type'] == 'DEBIT':
                payment_method_weightage = DEBIT_payment_weightage
            else:
                payment_method_weightage = bank_transfer_payment_weightage

            # Customer segment weightage
            if order['Customer Segment'] == 'Home Office':
                customer_segment_weightage = home_office_weightage
            elif order['Customer Segment'] == 'Corporate':
                customer_segment_weightage = corporate_weightage
            else:
                customer_segment_weightage = consumer_weightage

            # Calculate profit per order
            profit_per_order = (
                (order['Product Price'] * order_volume)
                - (shipping_cost + discount_weightage + payment_method_weightage)
            )  

            # Apply customer segment weightage
            total_profit += round((profit_per_order * customer_segment_weightage), 2)
        

            # print(f"Total Distance: {best_distance}, Total Profit: {total_profit}")

        # Return the total distance and total profit for all orders
            return best_distance, total_profit

    # ... (Continue with the rest of the code)

    # Define the order information as a list of dictionaries
    order_info_list = order_data.to_dict(orient='records')

    toolbox.register("evaluate", evaluate)
    # Number of orders
    num_orders = len(order_info_list)

    # Number of individuals in the population
    population_size = 50

    # Number of generations
    generations = 100

    # Loop through each order, create a population, and perform genetic algorithm
    best_fitness_values = []
    gen_best_fitness_values=[]
    pareto_fronts = []
    whole_output = []
    # best_fitness_values = []
    # Loop through each order, create a population, and perform genetic algorithm
    csv_filename = "output_results.csv"
    csv_header = ["Order", "Department ID", "Best Coordinates", "Best Shipping Mode", "Shipping Days"]

    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write the header to the CSV file
        csv_writer.writerow(csv_header)

        for order_index in range(num_orders):

            department_id_to_filter=order_data['Department Id'][order_index]
            places=filter_by_department_id(all_places, department_id_to_filter)

            # Create a population for the current order
            population = toolbox.population(n=population_size)

            # Evaluate the entire population for the current order
            fitness_values = list(map(toolbox.evaluate, population))

            # Assign fitness values to individuals
            for ind, fit in zip(population, fitness_values):
                ind.fitness.values = fit
            
            # Perform genetic algorithm for the current order
            algorithms.eaMuPlusLambda(population, toolbox, mu=population_size, lambda_=2*population_size,
                                    cxpb=0.7, mutpb=0.0, ngen=generations, stats=None, halloffame=None,
                                    verbose=False)
            
            # Extract Pareto front for the current order
            pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

            # Find the best individual in the final population
            best_individual = tools.selBest(population, k=1)[0]
            place_index, shipping_mode, shipping_days = best_individual
            
            # Extract fitness values of individuals in the Pareto front
            pareto_front_fitness_values = [ind.fitness.values for ind in pareto_front]
            
            # Store the Pareto front for the current order
            pareto_fronts.append(pareto_front_fitness_values)

            place_index, shipping_mode, shipping_days = best_individual

            coordinates = [(places[int(place_index)][1][0], places[int(place_index)][1][1])]
        
            print(best_individual.fitness.values[0])
            output_message = (
            f"Order {order_data['Order Item Id'][order_index]} -\n"
            f"Department ID {department_id_to_filter} -\n"
            f"  Best Coordinates: {coordinates}\n"
            f"  Best Shipping Mode: {shipping_mode}\n"
            f"  Shipping Days: {shipping_days}\n"
        )      
            # Writing to CSV file
            csv_row = [{order_data['Order Item Id'][order_index]}, department_id_to_filter, coordinates, shipping_mode, shipping_days]
            csv_writer.writerow(csv_row)
            output_tuple = (order_index + 1, output_message)

            # Append the tuple to the list
            whole_output.append(output_tuple)

            # Print the formatted output
            # print(whole_output)
        
            gen_best_fitness_values.append(best_individual.fitness.values)

            # Store the best fitness values for the current order
            best_fitness_values.append(gen_best_fitness_values)

            # print(best_individual.fitness.values)
    print(f"Results written to {csv_filename}")
    min_length = min(len(fit) for fit in best_fitness_values)
    # Initialize a string to store the formatted output
    formatted_output = ""

    # Iterate through the whole_output list
    for iteration, (order_index, output_message) in enumerate(whole_output, start=1):
        formatted_output += f"Iteration {iteration}\n"
        
        # Split the output_message to extract relevant information
        lines = output_message.split('\n')
        
        for line in lines:
            # Extract relevant information from each line
            if "Order" in line:
                order_info = line.split('-')[0].strip()
                formatted_output += f"    {order_info}\n"
            elif "Department ID" in line:
                department_info = line.split('-')[0].strip()
                formatted_output += f"    {department_info}\n"
            elif "Best Coordinates" in line:
                coordinates_info = line.split(':')[-1].strip()
                formatted_output += f"    {coordinates_info}\n"
            elif "Best Shipping Mode" in line:
                shipping_mode_info = line.split(':')[-1].strip()
                formatted_output += f"    {shipping_mode_info}\n"
            elif "Shipping Days" in line:
                shipping_days_info = line.split(':')[-1].strip()
                formatted_output += f"    {shipping_days_info}\n"
        
        # Add a newline between iterations
        formatted_output += "\n"

    # Print the formatted output
    print(formatted_output)
    plt.figure(figsize=(12, 8))  # Adjust the figure size as needed
    for order_index, best_fitness_values_order in enumerate(best_fitness_values):
        trimmed_fitness_values = gen_best_fitness_values[:min_length]
        plt.plot(range(1, len(best_fitness_values_order) + 1), best_fitness_values_order, label=f"Order {order_index + 1}")

    plt.xlabel('Generation')
    plt.ylabel('Best Fitness Value')
    plt.title('Convergence Plot')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -1.15, 1, 1), ncol=5, mode='expand', fancybox=True, shadow=True, title='Legend Title', columnspacing=1.5)
    plt.tight_layout(pad=3.0)  # Increase the padding for better readability    
    plt.savefig('convergence_plot.png')
    # plt.show()
    # Plotting the Convergence Plot
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    
    # Plotting the Pareto front for each order
    for order_index, pareto_front_fitness_values in enumerate(pareto_fronts):
        # Extract fitness values for each objective
        objective1_values = [fitness[0] for fitness in pareto_front_fitness_values]
        objective2_values = [fitness[1] for fitness in pareto_front_fitness_values]

        # Plot the Pareto front for the current order
        plt.scatter(objective1_values, objective2_values, label=f'Order {order_index + 1}')

    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.title('Pareto Front for Multi-Objective Optimization')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -1.15, 1, 1), ncol=5, mode='expand', fancybox=True, shadow=True, title='Legend Title', columnspacing=1.5)
    plt.tight_layout(pad=3.0)  # Increase the padding for better readability    
    plt.savefig('pareto_front_plot.png')
    # plt.show()

    return formatted_output

    # Example usage:


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

        # Integrate both predictions, update the original DataFrame, and split data by country
        final_response = integrate_predictions_and_split_by_country(
            original_df, quantity_model, department_model, label_encoders, training_features)

        return final_response
    
    return jsonify({'error': 'Invalid file type'})

@app.route('/button', methods=['POST'])
def button_manage():

    another_response = perform_multi_objective_optimization('department_coordinates.csv', 'cleaned_data.csv', 'output_results.csv')    
    final_file_path = "output_results.csv"

    # Send the final CSV back to the front end
    return send_file(final_file_path, as_attachment=True, download_name=f'output_results.csv')


@app.route('/button2', methods=['POST'])
def get_file1():
    # Assuming you have file1.png in the project's root directory
    file_path = 'convergence_plot.png'

    if os.path.exists(file_path):
        return send_file(file_path, mimetype='image/png', as_attachment=True)
    else:
        return "File not found", 404

@app.route('/button3', methods=['POST'])
def get_file2():
    # Assuming you have file2.png in the project's root directory
    file_path = 'pareto_front_plot.png'

    if os.path.exists(file_path):
        return send_file(file_path, mimetype='image/png', as_attachment=True)
    else:
        return "File not found", 404




if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)