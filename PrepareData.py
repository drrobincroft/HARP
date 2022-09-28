#'''Prepare data from file

#Before preparing data, you must preprocess it
#(1)Use FakeMan2 extra mode to transfer data in wav.npy into feature sequences with MFCC;
#(2)Use Matlab PreprocessBodyDataAfterMFCC.m funtion to tranfer data in body.npy into refined sequences with spline interpolation;
#'''

import numpy as np
import time
import torch

def ReadTrainAndValidData():    
    start=time.time()
    train_wav_data_array = torch.from_numpy(np.load('ellen/train_noduplication/mfcc_normalised.npy'))
    print("train_wav_data_array: {}".format(train_wav_data_array.shape))
    train_body_data_array = torch.from_numpy(np.load('ellen/train_noduplication/body_matched.npy'))
    print("train_body_data_array: {}".format(train_body_data_array.shape))
    validate_wav_data_array = torch.from_numpy(np.load('ellen/valid/mfcc_normalised.npy'))
    print("validate_wav_data_array: {}".format(validate_wav_data_array.shape))
    validate_body_data_array = torch.from_numpy(np.load('ellen/valid/body_matched.npy'))
    print("validate_body_data_array: {}".format(validate_body_data_array.shape))

    end=time.time()
    print(f"\nTime to read: {round(end-start,5)} seconds.")

    return torch.cat((train_wav_data_array,validate_wav_data_array),0), torch.cat((train_body_data_array,validate_body_data_array),0),validate_wav_data_array,  validate_body_data_array

def ReadTestData():
    start=time.time()
    test_wav_data_array = np.load('ellen/test/mfcc_normalised.npy')
    print("test_wav_data_array: {}".format(test_wav_data_array.shape))
    test_body_data_array = np.load('ellen/test/body_matched.npy')
    print("test_body_data_array: {}".format(test_body_data_array.shape))
    end=time.time()
    print(f"\nTime to read: {round(end-start,5)} seconds.")

    return torch.from_numpy(test_wav_data_array), torch.from_numpy(test_body_data_array)

def ReadTrainDataAndValidData_Harp():
    start=time.time()
    train_wav_data_array = torch.from_numpy(np.load('ellen/train_noduplication/mfcc_normalised_gentle.npy'))
    print("train_wav_data_array: {}".format(train_wav_data_array.shape))
    train_body_data_array = torch.from_numpy(np.load('ellen/train_noduplication/body_matched_gentle_angle.npy'))
    print("train_body_data_array: {}".format(train_body_data_array.shape))
    validate_wav_data_array = torch.from_numpy(np.load('ellen/valid/mfcc_normalised.npy'))
    print("validate_wav_data_array: {}".format(validate_wav_data_array.shape))
    validate_body_data_array = torch.from_numpy(np.load('ellen/valid/body_matched_angle.npy'))
    print("validate_body_data_array: {}".format(validate_body_data_array.shape))

    end=time.time()
    print(f"\nTime to read: {round(end-start,5)} seconds.")

    return train_wav_data_array, train_body_data_array, validate_wav_data_array,  validate_body_data_array

def ReadTestData_Harp():
    start=time.time()
    test_wav_data_array = np.load('ellen/test/mfcc_normalised.npy')
    print("test_wav_data_array: {}".format(test_wav_data_array.shape))
    test_body_data_array = np.load('ellen/test/body_matched_angle.npy')
    print("test_body_data_array: {}".format(test_body_data_array.shape))
    end=time.time()
    print(f"\nTime to read: {round(end-start,5)} seconds.")

    return torch.from_numpy(test_wav_data_array), torch.from_numpy(test_body_data_array)