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

#creating stats class, containing value, mean, std
class stats:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std 
    def normalize(self, value, mean, std):
        self.normalized = ((value - mean) / std)  

#Intialising objects containing std and mean
cement = stats(281.346602,104.505455)
slag = stats(71.850902,85.812598)    
flyash = stats(55.238419,63.853538)
water = stats(182.108044,21.215630)
superplasticizer = stats(6.091262,5.998052)
coarseaggregate = stats(974.500000,76.926675)
fineaggregate = stats(771.734535,80.189816) 
age = stats(45.313454,62.168311)

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

        cement.value = content['cement']
        slag.value = content['slag']
        flyash.value = content['flyash']
        water.value = content['water']
        superplasticizer.value = content['superplasticizer']
        coarseaggregate.value = content['coarseaggregate']
        fineaggregate.value = content['fineaggregate']
        age.value = content['age']

        #Normalizing input value of variables
        cement.normalize(cement.value,cement.mean,cement.std)
        slag.normalize(slag.value,slag.mean,slag.std)
        flyash.normalize(flyash.value,flyash.mean,flyash.std)
        water.normalize(water.value,water.mean,water.std)
        superplasticizer.normalize(superplasticizer.value,superplasticizer.mean, superplasticizer.std)
        coarseaggregate.normalize(coarseaggregate.value,coarseaggregate.mean, coarseaggregate.std)
        fineaggregate.normalize(fineaggregate.value,fineaggregate.mean,fineaggregate.std)
        age.normalize(age.value,age.mean,age.std) 

        #creating array 
        input_array = np.zeros((1,8))

        #assigning normalized variables to corresponding array position
        input_array[0,0] = cement.normalized
        input_array[0,1] = slag.normalized
        input_array[0,2] = flyash.normalized
        input_array[0,3] = water.normalized
        input_array[0,4] = superplasticizer.normalized
        input_array[0,5] = coarseaggregate.normalized 
        input_array[0,6] = fineaggregate.normalized
        input_array[0,7] = age.normalized 


        #predicting model 
        pred = model.predict(input_array)
        conc_strength = float(pred[0])
        response = {"id": str(uuid.uuid4()), 'Concrete strength': conc_strength}

    else:

        response = {"id": str(uuid.uuid4()), 'errors': error}
        

    return jsonify(response)


if __name__ == '__main__':
    app.debug = True
    app.run(host = "0.0.0.0", port = 5000)