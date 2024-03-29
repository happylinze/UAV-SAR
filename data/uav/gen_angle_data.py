import os
import numpy as np
from numpy.lib.format import open_memmap
from ThirdOrderRep import ThirdOrderRep
import multiprocessing as mp
from multiprocessing import Array
# bone
from tqdm import tqdm


def gen_angle_data_one_num_worker(path):
    data = np.load(path)
    train_x = data[f'x_train']
    train_y = data[f'y_train']
    test_x = data[f'x_test']
    test_y = data[f'y_test']

    N_train, T_train, _ = train_x.shape
    train_x = train_x.reshape((N_train, T_train, 2, 17, 3)).transpose(0, 4, 1, 3, 2)
    new_train_x = np.zeros((N_train, 9, T_train, 17, 2))
    Tor = ThirdOrderRep()
    for i in tqdm(range(N_train)):
        new_train_x[i] = Tor.getThridOrderRep(train_x[i])

    N_test, T_test, _ = test_x.shape
    test_x = test_x.reshape((N_test, T_test, 2, 17, 3)).transpose(0, 4, 1, 3, 2)
    new_test_x = np.zeros((N_test, 9, T_test, 17, 2))
    for i in tqdm(range(N_test)):
        new_test_x[i] = Tor.getThridOrderRep(test_x[i])

    new_train_x = new_train_x.transpose(0, 2, 4, 3, 1).reshape(N_train, T_train, -1)
    new_test_x = new_test_x.transpose(0, 2, 4, 3, 1).reshape(N_test, T_test, -1)

    np.savez(f'{path[:-4]}_angle.npz', x_train=new_train_x, y_train=train_y, x_test=new_test_x, y_test=test_y)

if __name__ == '__main__':
    gen_angle_data_one_num_worker('MMVRAC_CSv1.npz')
    gen_angle_data_one_num_worker('MMVRAC_CSv2.npz')