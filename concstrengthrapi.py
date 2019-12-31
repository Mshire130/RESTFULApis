from flask import Flask, request, jsonify 
import uuid
import os
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__) 

expected = {
    "cement": {"min":102,"max":540},
    "slag": {"min":0,"max":359},
    "flyash": {"min":0,"max":200},
    "water": {"min":122,"max":247},
    "superplasticizer": {"min":0,"max":32},
    "coarseaggregate": {"min":801,"max":1145},
    "fineaggregate": {"min":594,"max":992},
    "age": {"min":1,"max":365},
}

model = load_model("testmodel.h5")

@app.route('/api/calcstrength', methods= ['POST'])
def calc_strength():
    content = request.json
    error = []

    for name in content:
        if name in expected:
            expected_min = expected[name]['min']
            expected_max = expected[name]['max'] 

            value = content[name]

            if value < expected_min or value > expected_max:
                error.append(f'Out of range, {name} should be within {expected_min}-{expected_max} however value is {value}')
        else:
            error.append(f'Unexpected field: {name}')


    for name in expected:
        if name not in content:
            error.append(f'Missing {name} field')


    if len(error) < 1:
        #if no errors, start predicting 
        input_array = np.zeros((1,8))

        input_array[0,0] = content['cement']
        input_array[0,1] = content['slag']
        input_array[0,2] = content['flyash']
        input_array[0,3] = content['water']
        input_array[0,4] = content['superplasticizer']
        input_array[0,5] = content['coarseaggregate']
        input_array[0,6] = content['fineaggregate']
        input_array[0,7] = content['age']

        pred = model.predict(input_array)
        conc_strength = float(pred[0])
        response = {"id": str(uuid.uuid4()), 'Concrete strength': conc_strength}

    else:

        response = {"id": str(uuid.uuid4()), 'errors': error}
        

    return jsonify(response)

#need to normalize the content that comes in
#get normalizing from load concreg.py




    

if __name__ == '__main__':
    app.debug = True
    app.run(host = "0.0.0.0", port = 5000)