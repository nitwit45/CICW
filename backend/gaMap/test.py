import math
import random
import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
from geopy.geocoders import Nominatim
from datetime import datetime, timedelta
from itertools import permutations
from geopy.distance import geodesic
from math import radians, sin, cos, sqrt, atan2
from flask import Flask, jsonify
from flask_cors import CORS

df = pd.read_csv('data4.csv')

department_id = 3
date = '2016-04-05' 

df['Date'] = pd.to_datetime(df['shipping date (DateOrders)']).dt.date

filtered_data = df[(df['Department Id'] == department_id) & (df['Date'] == pd.to_datetime(date).date())]


category_weight = {
    8: ['Crafts', 'DVDs', 'CDs', 'Books', 'Garden', 'Music'],
    12: ['Camping & Hiking', 'Tennis & Racquet', 'Lacrosse', 'Water Sports', 'Indoor/Outdoor Games'],
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
    70: ['Oversized (This category may include items that exceed the weight limits of the other categories)']
}

def calculate_capacity(row):
    for weight, categories in category_weight.items():
        if row['Category Name'] in categories:
            return row['Order Item Quantity'] * weight
    return row['Order Item Quantity'] * 70 

filtered_data['Capacity'] = filtered_data.apply(calculate_capacity, axis=1)


def evaluate_allocation(individual):
    truck1_capacity = 0
    truck2_capacity = 0
    
    for i, allocate in enumerate(individual):
        if allocate:
            if truck1_capacity + sorted_orders.iloc[i]['Capacity'] <= 9000:
                truck1_capacity += sorted_orders.iloc[i]['Capacity']
            else:
                return 10000, 
        else:
            if truck2_capacity + sorted_orders.iloc[i]['Capacity'] <= 9000:
                truck2_capacity += sorted_orders.iloc[i]['Capacity']
            else:
                return 10000,
    
    return abs(truck1_capacity - truck2_capacity),

sorted_orders = filtered_data.sort_values(by='Capacity', ascending=False)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, len(sorted_orders))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate_allocation)

population = toolbox.population(n=100)
cxpb, mutpb, ngen = 0.5, 0.2, 50

for gen in range(ngen):
    offspring = algorithms.varAnd(population, toolbox, cxpb, mutpb)
    fitnesses = toolbox.map(toolbox.evaluate, offspring)
    for ind, fit in zip(offspring, fitnesses):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))

best_solution = tools.selBest(population, k=1)[0]
best_fitness = best_solution.fitness.values[0]

truck1_solution = []
truck2_solution = []
for i, allocate in enumerate(best_solution):
    if allocate:
        truck1_solution.append(sorted_orders.iloc[i])
    else:
        truck2_solution.append(sorted_orders.iloc[i])

capacity_1 = 0
capacity_2 = 0

print("Truck 1:")
for order in truck1_solution:
    capacity_1 += order['Capacity']
    print(f"Order Id: {order['Order Id']}, Capacity: {order['Capacity']}")
    
print('Total capacity of truck 1: ',capacity_1)

print("\nTruck 2:")
for order in truck2_solution:
    capacity_2 += order['Capacity']
    print(f"Order Id: {order['Order Id']}, Capacity: {order['Capacity']}")

print('Total capacity of truck 2: ',capacity_2)
print(f"\nBest Fitness: {best_fitness}")

def filter_orders_by_date_only(department_id, date):
    data['shipping date (DateOrders)'] = pd.to_datetime(data['shipping date (DateOrders)'])

    data['date_only'] = data['shipping date (DateOrders)'].dt.date

    date_to_filter = pd.to_datetime(date).date()

    filtered_data = data[(data['Department Id'] == department_id) & (data['date_only'] == date_to_filter)]
    
    if filtered_data.empty:
        print("No orders found for the given Department ID and date.")
    else:
        print("Orders matching the criteria:")
        print(filtered_data[['Order Id', 'order_longitude', 'order_latitude']])
    return filtered_data

def unique_coordinates_for_department(department_id):
    department_data = data[data['Department Id'] == department_id]

    unique_coordinates = department_data[['Latitude', 'Longitude']].drop_duplicates().values.tolist()
   
    return unique_coordinates

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371.0 

    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = sin(dlat / 2)*2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)*2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

file_path = 'data4.csv' 

data = pd.read_csv(file_path, encoding='ISO-8859-1')

order_coordinates = filter_orders_by_date_only(department_id, date)
branch_coordinates = unique_coordinates_for_department(department_id)

min_distance = float('inf')
nearest_branch = None

branch_distances = {}

for branch_coord in branch_coordinates:
    total_distance = 0

    for idx, order_row in order_coordinates.iterrows():
        order_longitude = order_row['order_longitude']
        order_latitude = order_row['order_latitude']
        distance = calculate_distance(branch_coord[0], branch_coord[1], order_latitude, order_longitude)
        total_distance += distance

    branch_distances[str(branch_coord)] = total_distance

    if total_distance < min_distance:
        min_distance = total_distance
        nearest_branch = branch_coord

print(f"\nThe branch with coordinates {nearest_branch} has the shortest total distance: {min_distance} km")

user_location = nearest_branch

def calculate_distance(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    R = 6371.0

    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = sin(dlat / 2)*2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)*2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

def evaluate(individual):
    total_distance = 0
    total_capacity_truck1 = 0
    total_capacity_truck2 = 0

    for idx in individual:
        if idx < len(truck1_solution):
            order = truck1_solution[idx]
            total_capacity_truck1 += order['Capacity']
            total_distance += calculate_distance(user_location, [order['order_latitude'], order['order_longitude']])
        else:
            order = truck2_solution[idx - len(truck1_solution)]
            total_capacity_truck2 += order['Capacity']
            total_distance += calculate_distance(user_location, [order['order_latitude'], order['order_longitude']])

    max_capacity_truck1 = 9000
    max_capacity_truck2 = 9000

    capacity_penalty_truck1 = max(0, total_capacity_truck1 - max_capacity_truck1)
    capacity_penalty_truck2 = max(0, total_capacity_truck2 - max_capacity_truck2)
    fitness = total_distance + capacity_penalty_truck1 + capacity_penalty_truck2

    return fitness,

toolbox = base.Toolbox()

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox.register("indices", random.sample, range(len(truck1_solution) + len(truck2_solution)), len(truck1_solution) + len(truck2_solution))
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

population_size = 50
num_generations = 100
cxpb, mutpb = 0.7, 0.2

pop = toolbox.population(n=population_size)


result, _ = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=num_generations, verbose=False)


best_individual = tools.selBest(result, k=1)[0]
best_path = best_individual 

best_path_truck1 = best_individual[:len(truck1_solution)]
best_path_truck2 = best_individual[len(truck1_solution):]

print("Best Path for Truck 1:", best_path_truck1)
print("Best Path for Truck 2:", best_path_truck2)

truck1_longitude = []
truck1_latitude = []
truck2_longitude = []
truck2_latitude = []

for value in best_path_truck1:
    if 0 <= value < len(order_coordinates):
        longitude_value = order_coordinates.iloc[value]['order_longitude']
        latitude_value = order_coordinates.iloc[value]['order_latitude']
        truck1_longitude.append(longitude_value)
        truck1_latitude.append(latitude_value)

print("Truck 1 - Longitude:", truck1_longitude)
print("Truck 1 - Latitude:", truck1_latitude)

for value in best_path_truck2:
    if 0 <= value < len(order_coordinates):
        longitude_value = order_coordinates.iloc[value]['order_longitude']
        latitude_value = order_coordinates.iloc[value]['order_latitude']
        truck2_longitude.append(longitude_value)
        truck2_latitude.append(latitude_value)

print("Truck 2 - Longitude:", truck2_longitude)
print("Truck 2 - Latitude:", truck2_latitude)

unique_longitude1 = {}
unique_latitude1 = {}


for lon in truck1_longitude:
    unique_longitude1[lon] = None

for lat in truck1_latitude:
    unique_latitude1[lat] = None

truck1_longitude = list(unique_longitude1.keys())
truck1_latitude = list(unique_latitude1.keys())

print("Truck 1 - Longitude after removing duplicates:", truck1_longitude)
print("Truck 1 - Latitude after removing duplicates:", truck1_latitude)

unique_longitude2 = {}
unique_latitude2 = {}

for lon in truck2_longitude:
    unique_longitude2[lon] = None

for lat in truck2_latitude:
    unique_latitude2[lat] = None

truck2_longitude = list(unique_longitude2.keys())
truck2_latitude = list(unique_latitude2.keys())

print("Truck 2 - Longitude after removing duplicates:", truck2_longitude)
print("Truck 2 - Latitude after removing duplicates:", truck2_latitude)

truck1_coordinates = [[lat, lon] for lat, lon in zip(truck1_latitude, truck1_longitude)]

print("Truck 1 - Coordinates after removing duplicates:")
print(truck1_coordinates)

truck2_coordinates = [[lat, lon] for lat, lon in zip(truck2_latitude, truck2_longitude)]

print("\nTruck 2 - Coordinates after removing duplicates:")
print(truck2_coordinates)