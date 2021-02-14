import pathlib
import sys

import numpy as np
import pandas as pd
import sklearn
import torch

np.seterr(divide='ignore', invalid='ignore')


def unpickle_data(path_name):
    read_data = []
    for table in pathlib.Path(path_name).glob("*.pkl"):
        read_data.append(pd.read_pickle(table))
    return read_data


def train_model_for_each_movie(list_arr):
    for i in range(len(list_arr)):
        x = 1
        train_x_i, train_y_i, test_x_i, test_y_i = extract_input_output(list_arr[i])
        input_layer = len(train_x_i)
        output_layer = len(train_y_i)
        newtwork_layers = (input_layer, 10, 7, 7, 6, 5, output_layer)
        num_iterations = 50
        learning_rate = 0.005
        parameters = train_model_DNN(train_x_i, train_y_i, newtwork_layers, learning_rate, num_iterations)
        pred_train, pred_test = predict_accuracy(parameters, train_x_i, train_y_i, test_x_i, test_y_i)
        print('Training Accuracy for' + str(i) + ' = {}'.format(pred_train))
        print('Test Accuracy for' + str(i) + ' = {}'.format(pred_test))
        print('Done for movie' + str(x) + '{}'.format(x))
        x += 1
    done = 'Done!'
    return done


def predict(X, y, parameters):
    m = X.shape[0]
    print('m = {}'.format(X.shape))
    print('y = {}'.format(y.shape))
    p = np.zeros((1, m))
    probas, caches = forward_prop(X, parameters)
    for i in range(0, probas.shape[0]):
        if probas[i, 0] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0
    acc = str(np.sum(p == y) / m)
    # print("Accuracy: " + str(np.sum((p == y) / m)))
    return acc


def predict_accuracy(parameters, train_x, train_y, test_x, test_y):
    print('Predicting values!')
    print('train_X = {}'.format(train_x.shape))
    print('train_y = {}'.format(train_y.shape))
    print('test_X = {}'.format(test_x.shape))
    print('test_y = {}'.format(test_x.shape))
    pred_train = predict(train_x, train_y, parameters)
    pred_test = predict(test_x, test_y, parameters)
    return pred_train, pred_test


def linear_forward(activation_prev, W, b):
    Z = np.dot(W, activation_prev) + b
    cache = (activation_prev, W, b)
    assert (Z.shape == (W.shape[0], activation_prev.shape[1]))
    return Z, cache


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = 1 / m * np.dot(dZ, A_prev.T)
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db


def forward_prop(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)],
                                             activation="relu")
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation="sigmoid")
    caches.append(cache)
    # assert (AL.shape == (1, X.shape[0]))
    return AL, caches


def backward_prop(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL,
                                                                                                      current_cache,
                                                                                                      activation="sigmoid")
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache,
                                                                    activation="relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads


def compute_cost(AL, Y):
    m = Y.shape[0]
    cost = (-1 / m) * np.sum((Y * np.log(AL)) + ((1 - Y) * np.log(1 - AL)))
    if (cost < 0 or cost == 'nan'):
        return -1
    print('in_cost = {}'.format(cost))
    cost = np.squeeze(cost)
    return cost


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache


def sigmoid_backward(dA, cache):
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    assert (dZ.shape == Z.shape)
    return dZ


def relu(X):
    A = np.maximum(0, X)
    assert (A.shape == X.shape)
    cache = A
    return A, cache


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    return dZ


def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    cache = (linear_cache, activation_cache)
    return A, cache


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, cache[1])
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, cache[1])
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters['W' + str(l + 1)] - learning_rate * grads['dW' + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * grads['db' + str(l + 1)]
    return parameters


def train_model_DNN(X, Y, layers_dims, learning_rate=0.05, num_iterations=1000):
    parameters = initialize_parameters(layers_dims)
    cost = 0
    for i in range(0, num_iterations):
        if (cost < 0):
            break
        AL, caches = forward_prop(X, parameters)
        cost = compute_cost(AL, Y)
        if (cost == -1):
            break
        grads = backward_prop(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
    print('cost = {}'.format(cost))
    return parameters


def extract_input_output(input_dict):
    dim = (input_dict['place'].shape[0])
    print('input = {}'.format(dim))
    Y = input_dict['scene_transition_boundary_ground_truth'].reshape(dim - 1, 1)
    print('Y' + str(Y.shape))
    Y = np.append(Y, [['False']], axis=0)
    print('Y after append' + str(Y.shape))
    place_arr = (input_dict['place'])
    list_place = np.split(place_arr, 4, axis=1)
    print('sizes' + str(list_place[0].shape) + str(list_place[1].shape) + str(list_place[2].shape) + str(
        list_place[3].shape))
    cast_arr = input_dict['cast'].reshape(dim, -1)
    print('cast = {}'.format(cast_arr.shape))
    action_arr = input_dict['action'].reshape(dim, -1)
    print('action = {}'.format(action_arr.shape))
    audio_arr = input_dict['audio'].reshape(dim, -1)
    print('audio = {}'.format(audio_arr.shape))
    stacked_X = np.column_stack(
        (list_place[0], list_place[1], list_place[2], list_place[3], cast_arr, action_arr, audio_arr))
    print('stacked_col_size = {}'.format(stacked_X.shape))
    print('y = {}'.format(Y.shape))
    assert (stacked_X.shape[0] == Y.shape[0])
    train_x, test_x, train_y, test_y = sklearn.model_selection.train_test_split(stacked_X, Y, test_size=0.3,
                                                                                train_size=0.7, shuffle=True,
                                                                                stratify=None)
    print('after split:' + str(train_x.shape) + str(train_y.shape) + str(test_x.shape) + str(test_y.shape))
    return train_x, train_y, test_x, test_y


def initialize_parameters(layer_dimensions):
    parameters = {}
    L = len(layer_dimensions)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dimensions[l], layer_dimensions[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dimensions[l], 1))
        assert (parameters['W' + str(l)].shape == (layer_dimensions[l], layer_dimensions[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dimensions[l], 1))
    print('Random Initialization of weights and bias successful!')
    return parameters


def convert_tensor_to_numpy(unpickled_data):
    list_of_movies = []
    for dict_elem in unpickled_data:
        place_arr = to_numpy(dict_elem['place'])
        cast_arr = to_numpy(dict_elem['cast'])
        action_arr = to_numpy(dict_elem['action'])
        audio_arr = to_numpy(dict_elem['audio'])
        scene_transition_boundary_ground_truth_arr = to_numpy(dict_elem['scene_transition_boundary_ground_truth'])
        shot_end_frame_arr = to_numpy(dict_elem['shot_end_frame'])
        scene_transition_boundary_prediction_arr = to_numpy(dict_elem['scene_transition_boundary_prediction'])
        imdb_id = dict_elem['imdb_id']
        dict_movie = {'place': place_arr, 'cast': cast_arr, 'action': action_arr, 'audio': audio_arr,
                      'scene_transition_boundary_ground_truth': scene_transition_boundary_ground_truth_arr,
                      'shot_end_frame': shot_end_frame_arr,
                      'scene_transition_boundary_prediction': scene_transition_boundary_prediction_arr,
                      'imdb_id': imdb_id}
        list_of_movies.append(dict_movie)
    return list_of_movies


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


if __name__ == '__main__':
    print('Screen Segmentation')
    data_dir = sys.argv[1]
    read_data_unpickled = unpickle_data(data_dir)
    print('Successfully unpickled {} files!'.format(len(read_data_unpickled)))
    list_of_movie_arrays = convert_tensor_to_numpy(read_data_unpickled)
    status = train_model_for_each_movie(list_of_movie_arrays)
    print(status)
