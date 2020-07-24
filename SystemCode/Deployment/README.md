# Jupyter
## 1. Export the model

```bash
#### Export the scaler and model ####

import pickle
# Export the scaler
with open('./diabetes-scaler.pkl', 'wb') as scaler_pkl:
  pickle.dump(scaler, scaler_pkl)

# Export the model
with open('./diabetes-knn-model.pkl', 'wb') as model_pkl:
  pickle.dump(knn, model_pkl)
```

## 2. Import the model
```bash
#### Import the scaler and model ####

import pickle

# Import all the packages you need for your model below
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Load the scaler 
with open('./diabetes-scaler.pkl', 'rb') as scaler_pkl:
    scaler_load = pickle.load(scaler_pkl)
    
# Load the model
with open('./diabetes-knn-model.pkl', 'rb') as model_pkl:
    knn_load = pickle.load(model_pkl)
```

## 3. Make prediction
```bash
X_unseen = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])

# Apply the scale
X_unseen_scale = scaler_load.transform(X_unseen)

# Get the result
result = knn_load.predict(X_unseen_scale)

# Print result to the console
print('Predicted result for observation ' + str(X_unseen) + ' is: ' + str(result))
```

# Create API endpoint with Flask
```bash
$ pwd
/Users/kenly/Documents/Work/ISS-IS02PT/PRS-PM-ISY5002-GROUP5/SystemCode/Deployment/deploy

$ python3 -m venv venv

$ source venv/bin/activate 

$ pip install -r env_requirements.txt

$ python flask_code.py

$ curl -X POST 0.0.0.0:80/predict -H 'Content-Type: application/json' -d '[1, 85, 66, 29, 0, 26.6, 0.351, 31]'
```
# Package into Docker container
```code
$ cat Dockerfile

$ docker build -t ldkhang/diabetes-knn-1.0 .

$ docker run -p 8000:80 ldkhang/diabetes-knn-1.0

```




# Deploy in Cloud Run

# CI/CD