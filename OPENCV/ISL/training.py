import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load data using pickle
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Flatten data and convert to numpy arrays
data_flat = []
for sample in data:
    flattened_sample = np.ravel(sample).astype(np.float32)
    data_flat.append(flattened_sample)

# Convert data_flat to numpy array
data_flat = np.array(data_flat)

labels = np.ravel(labels)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data_flat, labels, test_size=0.2, shuffle=True, stratify=labels)

# Convert data types for KNN
x_train = np.array(x_train).astype(np.float32)
y_train = np.array(y_train).astype(np.int32)
x_test = np.array(x_test).astype(np.float32)
y_test = np.array(y_test).astype(np.int32)

# Initialize KNN classifier
model = KNeighborsClassifier(n_neighbors=5)

# Train the KNN model
model.fit(x_train, y_train)

# Make predictions
y_predict = model.predict(x_test)

# Evaluate accuracy
score = accuracy_score(y_test, y_predict)
print('{}% of samples were classified correctly using KNN!'.format(score * 100))

# Save the trained model
with open('knn_model.pkl', 'wb') as f:
    pickle.dump(model, f)
