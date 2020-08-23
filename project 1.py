import numpy as np
import scipy.io

dataset = scipy.io.loadmat("mnist_data.mat")

# Extracting the 4 individual Matrix from Dataset
training_set = dataset["trX"]
training_set_label = dataset["trY"]
testing_set = dataset["tsX"]
testing_set_label = dataset["tsY"]

# splitting the train dataset into 7 and 8
training_set_7 = training_set[0:6265]
training_set_8 = training_set[6265:]

# Calculating the Mean and Standard Deviation of each image 
mean_7 = np.mean(training_set_7, axis=1)
mean_8 = np.mean(training_set_8, axis=1)
std_7 = np.std(training_set_7, axis=1)
std_8 = np.std(training_set_8, axis=1)

# Mean and Standard Daviation 'array' for the training Dataset and testing Dataset
training_set_mean = np.mean(training_set, axis=1)
testing_set_mean = np.mean(testing_set, axis=1)
training_set_sd = np.std(training_set, axis=1)
testing_set_sd = np.std(testing_set, axis=1)

# calculating the probability that an image belongs to 7 and 8 
prob_7 = mean_7.size / training_set_mean.size
prob_8 = mean_8.size / training_set_mean.size

# Naive Bayes
# Function to form PDF
def p_x_given_y(x, mean, variance):
    p_x_give_y = (1 / (np.sqrt(2 * np.pi * variance))) * np.exp(-(x - mean) ** 2 / (2 * variance))
    return p_x_give_y

# # calculate post prob for image with 7
post_prob_7 = prob_7 \
              * p_x_given_y(testing_set_mean, mean_7.mean(), mean_7.var()) \
              * p_x_given_y(testing_set_sd, std_7.mean(), std_7.var())

# # calculate post prob for image with 8
post_prob_8 = prob_8 \
              * p_x_given_y(testing_set_mean, mean_8.mean(), mean_8.var()) \
              * p_x_given_y(testing_set_sd, std_8.mean(), std_8.var())

compare = np.greater(post_prob_8, post_prob_7)

# Converting the True Values to 1 & False Values to 0
compare_numeric = compare.astype(np.int)

# calculating the accuracy for 7
accuracy_7 = ((np.count_nonzero(np.equal(compare_numeric[0:1028], np.squeeze(testing_set_label)[0:1028]))
              / np.squeeze(testing_set_label)[0:1028].size)
              * 100)
print('The Accuracy of the Naive Bayes for predicting "7" is ', accuracy_7, "%")

# calculating the accuracy for class - 8
accuracy_8 = ((np.count_nonzero(np.equal(compare_numeric[1028:], np.squeeze(testing_set_label)[1028:]))
              / np.squeeze(testing_set_label)[1028:].size)
              * 100)
print('The Accuracy of the Naive Bayes for predicting "8" is ', accuracy_8, "%")

# Comparing and finding tot number of correctly matched and unmatched using testing dataset label

compare_numeric_and_label = np.equal(compare_numeric, testing_set_label)

#Total number of correct prediction
tot_correct_prediction = np.count_nonzero(compare_numeric_and_label)

# The overall accuracy (Total Number of Images Predicted Correctly of the Test Dataset/Total Number of Images in the Test Dataset)

accuracy_naive_bayes = ((tot_correct_prediction / testing_set_label.size) * 100)
print("The Accuracy of the Naive Bayes is ", accuracy_naive_bayes, "%")


#   "" LOGISTIC REGRESSION CLASSIFIER ""
# Defining function for Logistic Regression
def sig_func(x):
    return 1 / (1 + np.exp(-x))

def predict(X, weights):
    
    linear_comb = np.dot(X, weights)
    # doing prediction
    y_predicted = sig_func(linear_comb)
    # identify 7 or 8
    y_predicted_temp = np.greater(y_predicted,0.5)
    # change the boolean expression to 1 & 0
    y_predicted = np.multiply(y_predicted_temp,1)
    
    return y_predicted


def training_gradient_ascent(X, y, learning_rate, iterations):
    
    # initialize weights
    total_sample, total_features = X.shape
    weights = np.zeros(total_features)
    # gradient ascent
    for iterations in range(iterations):
        linear_comb = np.dot(X, weights)
        # apply sigmoid_func function
        y_predicted = sig_func(linear_comb)
        # calculate gradient
        gradient = (1 / total_sample) * np.dot(X.T, (y - y_predicted))
        # update weights
        weights += learning_rate * gradient
        
    return weights

def calculate_accuracy (compared, true_labels, test_for):
    
    if test_for == 0:
        # testing for hand written 7 and squeeze the input array
        compared_array = np.squeeze(compared)[0:1028]
        true_labels_array = np.squeeze(true_labels)[0:1028]
    
    elif test_for == 1:
        compared_array = np.squeeze(compared)[1028:]
        true_labels_array = np.squeeze(true_labels)[1028:]       
    else:
        compared_array = np.squeeze(compared)
        true_labels_array = np.squeeze(true_labels)

    total_test_case = true_labels_array.shape[0]
    
    test_result = np.equal(compared_array, true_labels_array)
    # count the positive cases
    true_cases = np.count_nonzero(test_result)
    
    accuracy = true_cases/total_test_case
    accuracy_percent = accuracy * 100
    
    return accuracy, accuracy_percent
#setup learning rate and max iteration for LR
learning_rate = 0.001
max_iteration = 10000

#  calculate weights
weights = training_gradient_ascent(training_set, np.squeeze(training_set_label), learning_rate, max_iteration)
# feed weights to calculate predict value
predicted = predict(testing_set,weights)

# calculate accuracy for 7 and 8
accuracy_7, accuracy_7_precent = calculate_accuracy(predicted, testing_set_label,0)
print('The Accuracy of the Logistic Regression for predicting "7" is ', accuracy_7_precent, "%")

accuracy_8, accuracy_8_precent = calculate_accuracy(predicted, testing_set_label,1)
print('The Accuracy of the Logistic Regression for predicting "8" is ', accuracy_8_precent, "%")

# calculate overall accuracy
accuracy, accuracy_precent = calculate_accuracy(predicted, testing_set_label,2)
print('The Accuracy of the Logistic Regression is ', accuracy_precent, "%")

print('Learning Rate has set to ', learning_rate)
print('Max iteration has set to ', max_iteration)