from flask import Flask, render_template, request
import pickle
import numpy as np
import json


# Load the model
with open('artifacts/model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the StandardScaler
with open('artifacts/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


# Load column names and options
with open('artifacts/columns.json') as file:
    columns = json.load(file)['columns']

property_types = columns[3:7]

locations = columns[9:]

print(len(columns))

app = Flask(__name__)


def predict(features):

    input_array = np.zeros(len(columns))
    
    # standarizing numerical values and inputing
    unscaled = features[0:3]
    scaled = scaler.transform([unscaled])[0] # numerical values scaled

    baths = scaled[0]
    bedrooms = scaled[1]
    area = scaled[2]

    input_array[columns.index('baths')] = baths
    input_array[columns.index('bedrooms')] = bedrooms
    input_array[columns.index('area_sqft')] = area


    #inputing categorical values
    prop_index = columns.index(features[3])
    purp_index = columns.index(features[4])
    loc_index = columns.index(features[5])

    input_array[prop_index] = 1
    input_array[purp_index] = 1
    input_array[loc_index] = 1
    
    

    input_array = input_array.reshape(1, 200)
    y= model.predict(input_array)[0]

    return y


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    inlakh = True

    if request.method == 'POST':
        # Gather input data from the form
        features = [
            float(request.form['baths']),  #0
            float(request.form['bedrooms']), #1
            float(request.form['area']), #2
            request.form['property_type'], #3
            request.form['purpose'], #4
            request.form['location'] #5
        ]

        prediction = predict(features)

        if prediction < 1:
            prediction = prediction*1e5
            inlakh = False
        
    return render_template('index.html', prediction=prediction, property_types=property_types, locations=locations, inlakh= inlakh)


if __name__ == '__main__':
    app.run(debug=True)
