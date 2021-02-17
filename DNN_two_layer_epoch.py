import gzip
# import tensorflow as tf
# from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os

# fashion_mnist = keras.datasets.fashion_mnist
# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
def load_data():
    dirname = os.path.join('datasets', 'fashion-mnist')
    base = 'D:/Dpan/python学习资料跳槽加油/MachineLearning/DNN/data/'
    files = ['train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
      't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz']

    paths = [base + f_name for f_name in files]

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
        imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
        imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)
    return (x_train, y_train), (x_test, y_test)

# 本地数据读取
(train_images, train_labels), (test_images, test_labels) = load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# plt.figure()
# plt.imshow(train_images[3])
# plt.colorbar()
# plt.grid(False)
# plt.show()
#
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()
def convert_to_onehot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y
train_images_flatten = train_images.reshape(train_images.shape[0], -1).T
test_images_flatten = test_images.reshape(test_images.shape[0], -1).T
train_labels_onehot = convert_to_onehot(train_labels, 10)
test_labels_onehot = convert_to_onehot(test_labels, 10)
# 对数据进行标准化
train_set_x = train_images_flatten/255.0
test_set_x = test_images_flatten/255.0

# 编写模型相关部分
def relu(Z):
    A = np.maximum(0, Z)
    assert(A.shape == Z.shape)
    cache = Z
    return A, cache
def softmax(Z):
    Z_shift = Z - np.max(Z, axis=0)
    A = np.exp(Z_shift)/np.sum(np.exp(Z_shift), axis=0)
    cache = Z_shift
    return A, cache
def initialize_parameters(nx, nh, ny):
    np.random.seed(1)
    W1 = np.random.randn(nh, nx) * 0.01
    b1 = np.zeros((nh, 1))
    W2 = np.random.randn(ny, nh) * 0.01
    b2 = np.zeros((ny, 1))

    assert(W1.shape == (nh, nx))
    assert(b1.shape == (nh, 1))
    assert(W2.shape == (ny, nh))
    assert(b2.shape == (ny, 1))

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    return parameters
#--------forward---------
def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    if activation == "softmax":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = softmax(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert(A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    return A, cache
def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -(np.sum(Y * np.log(AL))) / float(m)
    assert(cost.shape == ())
    return cost
#----backward-----
def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / float(m)
    db = np.sum(dZ, axis = 1, keepdims=True) / float(m)
    dA_prev = np.dot(W.T, dZ)

    assert(dA_prev.shape == A_prev.shape)
    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)

    return dA_prev, dW, db

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z<=0] = 0
    assert(dZ.shape == Z.shape)
    return dZ

def softmax_backward(Y, cache):
    Z = cache
    s = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    dZ = s - Y
    assert (dZ.shape == Z.shape)
    return dZ

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "softmax":
        dZ = softmax_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

def update_parameters(parameters, grads, learning_rate, i):
    L = len(parameters) // 2
    for l in range(1, L+1):
        parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]

    return parameters

#---模型整合及有效性检测---
def two_layer_model(X, Y, parameters, learning_rate = 0.1):

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    A1, cache1 = linear_activation_forward(X, W1, b1, activation="relu")
    A2, cache2 = linear_activation_forward(A1, W2, b2, activation="softmax")

    cost = compute_cost(A2, Y)

    dA1, dW2, db2 = linear_activation_backward(Y, cache2, activation="softmax")
    dA0, dW1, db1 = linear_activation_backward(dA1, cache1, activation="relu")

    grads["dW1"] = dW1
    grads["db1"] = db1
    grads["dW2"] = dW2
    grads["db2"] = db2

    parameters = update_parameters(parameters, grads, learning_rate)

    return cost, parameters

#随机打散X和Y
def _shuffle(X, Y):
    randomize = np.arange(len(X[0]))#生成从0到len(X)的list
    np.random.shuffle(randomize)
    return (X[:, randomize], Y[:, randomize])

max_iter = 16
batch_size = 32
train_x = train_set_x
train_y = train_labels_onehot
m = train_x.shape[1]
np.random.seed(1)
(nx, nh, ny) = (784, 128, 10)
parameters = initialize_parameters(nx, nh, ny)
costs = []
i = 0
for epoch in range(max_iter):
    grads = {}
    X_train, Y_train = _shuffle(train_x, train_y)
    batch_num = int(np.floor(m) / batch_size)
    for idx in range(batch_num):
        X = X_train[:, idx*batch_size:(idx+1)*batch_size]
        Y = Y_train[:, idx*batch_size:(idx+1)*batch_size]
        i += 1
        cost, parameters = two_layer_model(X, Y, parameters, learning_rate=0.09)

        if i % 1000 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 1000 == 0:
            costs.append(cost)

plt.plot(np.squeeze(costs))
plt.ylabel("cost")
plt.xlabel("iterations(per tens)")
plt.title("Learning rate=" + str(0.1))
plt.show()


def predict_labels(X, y, parameters):
    m = X.shape[1]
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    # Forward propagation
    A1, _ = linear_activation_forward(X, W1, b1, activation='relu')
    probs, _ = linear_activation_forward(A1, W2, b2, activation='softmax')
    # convert probas to 0-9 predictions
    predict_label = np.argmax(probs, axis=0)
    print("Accuracy:" + str(np.sum(predict_label == y) / float(m)))
    return predict_label

predictions = predict_labels(test_set_x, test_labels, parameters)
print(predictions)

