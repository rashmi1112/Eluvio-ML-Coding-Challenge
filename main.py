import itertools
import pathlib
import sys

import numpy as np
import pandas as pd
import torch


def unpickle_data(path_name):
    read_data = []
    for table in pathlib.Path(path_name).glob("*.pkl"):
        read_data.append(pd.read_pickle(table))
    return read_data


def partition_data(data):
    per = 0.7
    length_data = len(data)
    idx = np.random.permutation(data)
    train_data = idx[0:int(np.round(per * length_data))]
    test_data = idx[int(np.round(per * length_data)):length_data + 1]
    if len(train_data) != int(np.round(length_data * per)):
        print('Error partitioning data!')
        return False
    else:
        print('Successfully partitioned data into train and test.')
        return train_data, test_data


def train_model_for_each_movie(list_arr):
    for i in range(len(list_arr)):
        x_i, y_i = extract_input_output(list_arr[i])
        train_x, train_y = partition_data(x_i)
        test_x, test_y = partition_data(y_i)
        # dict = train_model_DNN()


def initialize_parameters_deep(layer_dimensions):
    parameters = {}
    L = len(layer_dimensions)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dimensions[l], layer_dimensions[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dimensions[l], 1))
        assert (parameters['W' + str(l)].shape == (layer_dimensions[l], layer_dimensions[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dimensions[l], 1))

    return parameters


def linear_forward(activation_prev, W, b):
    Z = np.dot(W, activation_prev) + b
    cache = (activation_prev, W, b)
    return Z, cache


def forward_prop(X, parameters):
    caches = []
    A = X
    L = len(parameters)
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)],
                                             activation="relu")
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation="sigmoid")
    caches.append(cache)
    return AL, caches


def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = (-1 / m) * np.sum((Y * np.log(AL)) + ((1 - Y) * np.log(1 - AL)))
    cost = np.squeeze(cost)
    return cost


def sigmoid(x):
    sig = 1 / (1 + np.exp(-x))
    return sig


def relu(x):
    rel = max(0, x)
    return rel


def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    cache = (linear_cache, activation_cache)
    return A, cache


def train_model_DNN(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    costs = []
    parameters = initialize_parameters_deep(layers_dims)
    for i in range(0, num_iterations):
        AL, caches = forward_prop(X, parameters)
        cost = compute_cost(AL, Y)


def extract_input_output(input_dict):
    dim = len(input_dict['place'])
    Y = input_dict['scene_transition_boundary_ground_truth'].reshape(-1, 1)
    # print('Y_shape={}', format(len(Y)))
    # print('input_1 = {}'.format((input_dict['place'].shape[0])))
    # print('place = {}'.format(input_dict['place'].shape))
    # print('cast = {}'.format(input_dict['cast'].shape))
    # print('act = {}'.format(input_dict['action'].shape))
    # print('aud = {}'.format(input_dict['audio'].shape))
    place_list = input_dict['place'].reshape(dim, 1)
    cast_list = input_dict['cast'].reshape(dim, 1)
    action_list = input_dict['action'].reshape(dim, 1)
    audio_list = input_dict['audio'].reshape(dim, 1)
    X = list(itertools.chain(place_list, cast_list, action_list, audio_list))
    return X, Y


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
    # (train_data, test_data) = partition_data(list_of_movie_arrays)
    # print('Train data size = {}'.format(len(train_data)))
    # print('Test data size = {}'.format(len(test_data)))

    X_m, Y_m = extract_input_output(list_of_movie_arrays[0])

    # network_layers = [7, 5, 6, 4, 4]
    # parameters = initialize_parameters(network_layers)
