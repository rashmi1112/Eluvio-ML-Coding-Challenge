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
    round = np.round(64 * 0.7)
    print('value {}'.format(round))
    (train_data, test_data) = partition_data(list_of_movie_arrays)
    print('Train data size = {}'.format(len(train_data)))
    print('Test data size = {}'.format(len(test_data)))

    network_layers = [7, 5, 6, 4, 4]
    parameters = initialize_parameters(network_layers)
    print('Parameters: {}'.format(parameters))
