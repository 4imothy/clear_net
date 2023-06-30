import numpy as np
from sklearn.datasets import load_iris

np.random.seed(10381)

iris = load_iris()
x = iris.data
target = iris.target.reshape(-1, 1)

# remove class 0
while target[0] == 0:
    x = x[1:]
    target = target[1:]


# 4
input_size = len(x[0])
output_size = 1
learning_rate = 0.5
# weights = np.random.uniform(low=100, high=100, size=params['input_size'])

bias = np.random.uniform(low=-0.3, high=0.3)
params = {
    "input_size": input_size,
    "output_size": output_size,
    "learning_rate": 0.1,
    "bias": bias,
}

params['weights'] = np.random.uniform(low=-0.5, high=0.5, size=params['input_size'])


def predict(x, weights, bias):
    y = np.dot(x, weights) + bias
    sig = sigmoid(y)

    return sig


def get_data(x, params, target):
    # get the predictions as y
    predictions = predict(x, params['weights'], params['bias'])
    mse = 0
    classes = []
    for i in range(2):
        classes.append([])
    for p, t in zip(predictions, target):
        mse += (p + 1 - t)**2

    mse /= len(x)

    for i in range(len(predictions)):
        if predictions[i] > 0.5:
            classes[1].append(x[i])
        else:
            classes[0].append(x[i])

    return mse, classes, predictions


def sigmoid(x):
    return (1 / (1 + np.exp(-x)))


def main():
    mse, _classes, predictions = get_data(x, params, target)

    good_weights = [0.2, -0.50, 0.03, 0.3]
    bias = -0.7
    params['bias'] = bias
    params['weights'] = good_weights
    mse, classes, predictions = get_data(x, params, target)

    bad_weights = [0.05, -0.05, 1.0, 0.7]
    params['weights'] = bad_weights
    mse, classes, predictions = get_data(x, params, target)

    all_weights, all_classes, all_biases, good_weights, bias, err = train(x, params, target)
    

def train(x, params, target):
    all_weights = []
    all_classes = []
    all_biases = []
    weights = params['weights']
    bias = params['bias']
    itr = 0
    while True:
        mse, classes, predictions = get_data(x, params, target)
        delta_w, delta_b = getChangers(params['learning_rate'], weights, predictions)
        for i in range(len(weights)):
            weights[i] -= delta_w[i][0]
        bias -= delta_b
        all_weights.append(weights)
        all_classes.append(classes)
        all_biases.append(bias)

        print(weights, bias, mse)
        itr += 1

        if mse < 0.04:
            break

    print("Num iter: ", itr)
    return all_weights, all_classes, all_biases, weights, bias, mse


def getChangers(learning_rate, weights, predictions):
    alter = (-2 * learning_rate) / len(x)

    delta_w = []
    for i in range(len(x[0])):
        sum = 0
        for j in range(len(x)):
            sum += (target[j] - 1 - predictions[j]) * (1 - predictions[j]) * predictions[j] * x[j][i]
        sum *= alter
        delta_w.append(sum)

    sum = 0
    for i in range(len(x)):
        sum += (target[i] - 1 - predictions[i]) * predictions[i] * (1 - predictions[i])

    return delta_w, alter * sum


if __name__ == "__main__":
    main()
