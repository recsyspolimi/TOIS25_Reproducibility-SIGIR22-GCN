#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 29/01/2023

@author: Maurizio Ferrari Dacrema
"""

from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_user_wise
from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix
from SIGIR2022.KGAT_data.LastFM_KGAT_DataReader import _get_ICM_from_df

import os, zipfile, shutil
from Recommenders.DataIO import DataIO
import numpy as np
import pandas as pd


class KGAT_DataReader(object):
    """
    The knowledge base contains first the items, then other entities. The IDs of the items match, if an entity has an ID which
    is higher than the number of items it means it is another type of entity (location for example).
    The users are not part of the knowledge base
    """
    URM_DICT = {}
    ICM_DICT = {}
    UCM_DICT = {}

    def __init__(self, dataset_name, pre_splitted_path, train_validation_test = [0.70, 0.10, 0.20], freeze_split = True):
        super(KGAT_DataReader, self).__init__()

        pre_splitted_path += "data_split/"
        pre_splitted_filename = "splitted_data"
        
        dataset_dir = "SIGIR2022/KGAT_data/Data/{}/".format(dataset_name)
        
        # If directory does not exist, create
        if not os.path.exists(pre_splitted_path):
            os.makedirs(pre_splitted_path)

        dataIO = DataIO(pre_splitted_path)

        try:
            print("{}: Attempting to load saved data from {}".format(os.path.dirname(__file__), pre_splitted_path + pre_splitted_filename))
            for attrib_name, attrib_object in dataIO.load_data(pre_splitted_filename).items():
                self.__setattr__(attrib_name, attrib_object)
                
        except FileNotFoundError:
            
            if freeze_split:
                raise Exception("Splitted data not found!")
            
            print("{}: Pre-splitted data not found, building new one".format(os.path.dirname(__file__)))
            print("{}: loading data".format(os.path.dirname(__file__)))

            URM_train_validation = self._load_data_file(os.path.join(dataset_dir, 'train.txt'))
            URM_test = self._load_data_file(os.path.join(dataset_dir, 'test.txt'))

            URM_all = URM_train_validation + URM_test
            URM_all.data = np.ones_like(URM_all.data)

            # Split user-wise
            train_validation_percentage = train_validation_test[0] + train_validation_test[1]
            URM_train_validation, URM_test = split_train_in_two_percentage_user_wise(URM_all, train_percentage=train_validation_percentage)

            URM_train, URM_validation = split_train_in_two_percentage_user_wise(URM_train_validation, train_percentage=train_validation_test[0]/train_validation_percentage)


            dataFile = zipfile.ZipFile(os.path.join(dataset_dir, 'kg_final.txt.zip'))
            kg_path = dataFile.extract('kg_final.txt', path=pre_splitted_path + "decompressed/")

            self.knowledge_base_df = pd.read_csv(kg_path, header=None, sep=" ")
            self.knowledge_base_df.columns = ["head", "relation", "tail"]
            self.knowledge_base_df.drop_duplicates()

            shutil.rmtree(pre_splitted_path + "decompressed", ignore_errors=True)

            _, n_items = URM_train.shape

            self.ICM_DICT = {
                "ICM_entities": _get_ICM_from_df(self.knowledge_base_df.copy(), n_items)
            }

            self.UCM_DICT = {}

            self.URM_DICT = {
                "URM_train": URM_train,
                "URM_test": URM_test,
                "URM_validation": URM_validation,
            }

            # You likely will not need to modify this part
            data_dict_to_save = {
                "ICM_DICT": self.ICM_DICT,
                "UCM_DICT": self.UCM_DICT,
                "URM_DICT": self.URM_DICT,
                "knowledge_base_df": self.knowledge_base_df,
            }

            dataIO.save_data(pre_splitted_filename, data_dict_to_save=data_dict_to_save)

            print("{}: loading complete".format(os.path.dirname(__file__)))


    def _load_data_file(self, filePath, separator = " "):

        URM_builder = IncrementalSparseMatrix(auto_create_row_mapper=False, auto_create_col_mapper=False)

        fileHandle = open(filePath, "r", encoding = "utf-8")

        for line in fileHandle:
            if (len(line)) > 1:
                line = line.replace("\n", "")
                line = line.split(separator)
                
                # Avoid parsing users with no interactions
                if len(line[1]) > 0:
                    line = [int(line[i]) for i in range(len(line))]
                    URM_builder.add_single_row(line[0], line[1:], data=1.0)

        fileHandle.close()

        return  URM_builder.get_SparseMatrix()
