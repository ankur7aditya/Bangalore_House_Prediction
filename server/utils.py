import json
import pickle
import numpy as np

__locations = None
__data_columns = None
__model = None

def get_estimated_price(location, sqft, bhk, bath):
    global __data_columns
    try:
        loc_index = __data_columns.index(location.lower())
    except ValueError:
        loc_index = -1
    
    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    if loc_index >= 0:
        x[loc_index] = 1
    
    return round(__model.predict([x])[0], 2)

def get_location_names():
    return __locations

def load_saved_artifacts():
    print("Loading saved artifacts...start")
    global __data_columns
    global __locations
    global __model

    # Load columns.json and extract data_column
    try:
        with open("./artifacts/columns.json") as f:
            __data_columns = json.load(f)['data_column']  # Corrected key: 'data_column'
            __locations = __data_columns[3:]  # Assuming the first 3 columns are sqft, bath, and bhk
            print(f"Loaded data columns: {__data_columns}")  # Debugging print
    except Exception as e:
        print(f"Error loading columns.json: {e}")
    
    # Load the trained model from pickle file
    try:
        with open('./artifacts/banglore_home_prices_model.pickle', 'rb') as f:
            __model = pickle.load(f)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")

    print("Loading saved artifacts...done")

if __name__ == "__main__":
    load_saved_artifacts()
    print(get_location_names())  
    print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))
