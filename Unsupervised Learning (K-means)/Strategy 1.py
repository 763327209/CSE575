import numpy as np
import scipy.io
import math
import matplotlib.pyplot as plt

# Importing the data from the "AllSamples" file for the learning algorithm
raw_dataset = scipy.io.loadmat("AllSamples.mat")
data = raw_dataset['AllSamples']

# Extracting the x and y coordinates from the given dataset
x_cordinate = data[:,0]
y_cordinate = data[:,1]

# Initializing the centroid randomly
def centroid_init(k, data_sets):
    randomly = np.random.choice(data.shape[0], k, replace=False)
    centroids = data_sets[randomly]
    return centroids

# Objective Function for the K-means Algorithm
def k_means_obj(data_sets, centroids):
    objectiveval = []
    for r in data_sets:
        objectiveval.append(((np.linalg.norm((r - centroids), axis=0) ** 2)))
    return np.sum(objectiveval)

# Calculating the Euclidean Distance from the given Coordinates
def cal_euclidean_distance(x_cord, y_cord, x_cent, y_cent):
    x_new = (x_cent - x_cord) ** 2
    y_new = (y_cent - y_cord) ** 2
    disteuc = math.sqrt(x_new + y_new)
    return disteuc

# Initializing Color Map
color_map = ['orange','red', 'green', 'blue', 'yellow', 'grey', 'purple', 'maroon', 'yellowgreen', 'skyblue', 'wheat',
             'pink', 'cyan']

# initializing the k values from k = 2-10 and objective function
obj_plot = []
k_value = [2, 3, 4, 5, 6, 7, 8, 9, 10]

for k in k_value:
    centroids = centroid_init(k, data)
    y_centroid = centroids.take(1, axis=1).tolist()
    x_centroid = centroids.take(0, axis=1).tolist()
    # plot scatter graph for given data set with initial k
    figure = plt.figure(figsize=(5, 5))
    print("Graph for k = ", k)
    plt.scatter(x_cordinate, y_cordinate,color='blue')
    plt.scatter(x_centroid, y_centroid, marker='+', s=100, color='black')
    condition = True
    plt.ylim(-2, 10)
    plt.xlim(-2, 10)
    plt.show()
    temp1 = np.zeros(len(data))
    condition = True
    all = 0

    while (condition):
        Distt = []
        # For every data point in the given data
        for point in data:
            EDistt = []
            for cent in centroids:
                EDistt.append(cal_euclidean_distance(point[0], point[1], cent[0], cent[1]))
            Distt.append(EDistt)
        Distt = np.asarray(Distt)
        condition = True
        y_centroid = centroids.take(1, axis=1).tolist()
        x_centroid = centroids.take(0, axis=1).tolist()
        # Updating the centroid
        centroid_old = np.copy(centroids)
        # plot scatter graph
        temp1 = np.argmin(Distt, axis=1)
        figures = plt.figure(figsize=(5, 5))

        for i in range(k):
            points = np.asarray([data[j] for j in range(len(data)) if temp1[j] == i])
            centroids[i] = np.mean(points,axis=0)
            # printing map
            y_c = points.take(1, axis=1).tolist()
            x_c = points.take(0, axis=1).tolist()
            plt.scatter(x_c, y_c, c=color_map[i])
        condition_temp = (np.array_equal(centroid_old, centroids))
        if condition_temp:
            condition = False
        else:
            condition = True
        plt.ylim(-2, 10)
        plt.xlim(-2, 10)
        plt.scatter(x_centroid, y_centroid, marker='+', s=100, color="black")
        plt.show()

    # Calculating Objective Function and its Graph
    for i in range(k):
        tempp = np.asarray([data[j] for j in range(len(data)) if temp1[j] == i])
        val = k_means_obj(tempp, centroids[i])
        all = all + val
    obj_plot.append(all)
    print(obj_plot)
    print("Clusters is ", all)

# Plot for Objective Function
plt.plot(k_value, obj_plot, 'bx-')
plt.title('Plot of Objective Function ( Strategy 1 )')
plt.ylabel('Objective Function')
plt.xlabel('K Value')
plt.show()