import pathlib
import sys

import pandas as pd
import torch


def unpickle_data(path_name):
    read_data = []
    for table in pathlib.Path(path_name).glob("*.pkl"):
        read_data.append(pd.read_pickle(table))
    return read_data


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
        print('Converted Tensors to Numpy Arrays')
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
