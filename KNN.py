#!/usr/bin/env python
# coding: utf-8

# In[338]:


from sklearn import datasets
from math import sqrt
from sklearn.model_selection import train_test_split

def get_distance(row1, row2):
    distance = 0.0 # if row1 and row2 are the same, return 0.0
    features = len(row1) - 1 # minus 1 because the last of data is a label.
    for feature in range(features): 
        distance += (row1[feature] - row2[feature]) ** 2
    return sqrt(distance)

def get_neighbors(starting_point, other_points, n_neighbors):
    
    distances = [] # (label, distance)
    
    for other_point in other_points:
        distance = get_distance(starting_point, other_point)
        distances.append((other_point[-1], distance))
    
    distances = [(key, value) for key, value in sorted(distances, key=lambda item: item[1])]# sort distances with value
    neighbors = []
    for i in range(n_neighbors):
        neighbors.append(distances[i][0])
    return neighbors

def label_counts(neighbors):
    counts = {}
    for neighbor in neighbors:
        if neighbor not in counts:
            counts[neighbor] = 0
        counts[neighbor] += 1
    counts = {key: value for key, value in sorted(counts.items(), key=lambda item: item[1], reverse=True)}
    return counts

def prediction(starting_point, other_points, n_neighbors):
    neighbors = get_neighbors(starting_point, other_points, n_neighbors)
    counts = label_counts(neighbors)
    prediction = list(counts.keys())[0]
    if prediction == starting_point[-1]:
        return prediction, True
    else:
        return prediction, False

def connect_data_with_target(datas, targets):
    data_with_target = []
    for data, target in zip(datas, targets):
        data_with_target.append(list(data)+[target.tolist()])
    return data_with_target

if __name__ == "__main__":

    iris = datasets.load_iris()

    train_datasets = connect_data_with_target(X_train, y_train)
    test_datasets = connect_data_with_target(X_test, y_test)

    true_or_false = []
    for test_data in test_datasets:
        label, answer = prediction(test_data, train_datasets, 9)
        true_or_false.append(answer)

    answers = []
    for answer in true_or_false:
        if answer == True:
            answers.append(answer)

    print("Accuracy: ", len(answers) / len(test_datasets))
    # neighbor: 1 -> 1.0
    # neighbor: 2 -> 1.0
    # neighbor: 3 -> 1.0
    # ...
    # neighbor: 7 -> 0.9736842105263158
    # neighbor: 8 -> 1.0
    # neighbor: 9 -> 0.9736842105263158


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




