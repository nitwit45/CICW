# Global Supply Chain Optimization

This project optimizes global manufacturing supply chains using influential machine learning models and specialized Genetic Algorithms (GAs). Three key ML models – Product Quantity Prediction, Categorizations of Products by Departments, and Orders by Modes of Shipping – form the foundation of our comprehensive solution.

## Project Overview

The project fine-tunes logistics, optimizes inventory, and enhances profitability within the intricate landscape of supply chain management. It employs dual GAs, each designed to address specific complexities based on whether the orders are local or international.

- For local orders, the GA optimizes product distribution, inventory of delivery vehicles, and maps out the optimal route for efficient delivery.
- For international orders, a second GA is designed to augment profit per order, optimize the Leap Time of production and delivery, Shipping Days, and Mode of Shipping.

Leveraging the Data Co Global Supply Chain dataset, consisting of structured ('DataCoSupplyChainDataset.csv') and unstructured data, empowers advanced analytics and machine learning.

## Getting Started

### Frontend Installation

```bash
# Navigate to the frontend directory
cd frontend

# Install dependencies
npm install

# Start Project
npm start
```

### Backend Installation

```bash
# Navigate to the backend directory
cd backend

# Create a virtual environment (optional but recommended)
python3.11 -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python flaskapi.py
```
