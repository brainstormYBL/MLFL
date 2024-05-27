"""
Author   : Bao-lin Yin
Data     : 10.23 2023
Version  : V1.0
Function : Processing the data to train the model
"""
import numpy as np
import pandas as pd
import torch


class ProcessData:
    def __init__(self, address, radio_train, num_client):
        self.address = address
        self.radio_train = radio_train
        self.num_client = num_client
        self.data_pd = pd.read_csv(self.address)
        self.data_np = self.data_pd.to_numpy()
        self.num_data = self.data_np.shape[0]
        self.dim_label = 1
        self.dim_feature = self.data_np.shape[1] - 1
        self.data_each_client = self.split_data_to_all_client()
        self.train_label, self.train_feature, self.test_label, self.test_feature = \
            self.split_data_each_client_train_and_test()
        # 深入理解这一段代码
        self.train_label_tensor = {key: torch.tensor(value) for key, value in self.train_label.items()}
        self.train_feature_tensor = {key: torch.tensor(value) for key, value in self.train_feature.items()}
        self.test_label_tensor = {key: torch.tensor(value) for key, value in self.test_label.items()}
        self.test_feature_tensor = {key: torch.tensor(value) for key, value in self.test_feature.items()}
        self.all_label_tensor = torch.Tensor(np.reshape(self.data_np[:, self.dim_feature], (-1, self.dim_label)))
        self.all_feature_tensor = torch.Tensor(self.data_np[:, 0:self.dim_feature])

    def split_data_to_all_client(self):
        num_data_each_client = self.num_data // self.num_client
        data_each_client = dict()
        for index_client in range(self.num_client):
            data_each_client["client" + str(index_client)] = \
                self.data_np[index_client * num_data_each_client:(index_client + 1) * num_data_each_client]
        return data_each_client

    def split_data_each_client_train_and_test(self):
        train_label = dict()
        train_feature = dict()
        test_label = dict()
        test_feature = dict()
        for index_client in range(self.num_client):
            num_data_train = int(self.data_each_client["client" + str(index_client)].shape[0] * self.radio_train)
            train_feature["client" + str(index_client)] = self.data_each_client["client" + str(index_client)][
                                                          0:num_data_train, 0:self.dim_feature]
            train_label["client" + str(index_client)] = np.reshape(self.data_each_client["client" + str(index_client)][
                                                        0:num_data_train, self.dim_feature], (-1, self.dim_label))
            test_feature["client" + str(index_client)] = self.data_each_client["client" + str(index_client)][
                                                         num_data_train:
                                                         self.data_each_client["client" + str(index_client)].shape[0],
                                                         0:self.dim_feature]
            test_label["client" + str(index_client)] = np.reshape(self.data_each_client["client" + str(index_client)][
                                                       num_data_train:
                                                       self.data_each_client["client" + str(index_client)].shape[0],
                                                                  self.dim_feature], (-1, self.dim_label))
        return train_label, train_feature, test_label, test_feature
