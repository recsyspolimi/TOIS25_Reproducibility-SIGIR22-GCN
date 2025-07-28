#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 17/12/2022

@author: Maurizio Ferrari Dacrema
"""

from SIGIR2022.GDE_our_interface.GDE_RecommenderWrapper import GDE_RecommenderWrapper
from SIGIR2022.GTN_our_interface.GTN_RecommenderWrapper import GTN_RecommenderWrapper
from SIGIR2022.HAKG_our_interface.HAKG_RecommenderWrapper import HAKG_RecommenderWrapper
from SIGIR2022.HCCF_our_interface.HCCF_RecommenderWrapper import HCCF_RecommenderWrapper
from SIGIR2022.INMO_our_interface.INMO_RecommenderWrapper import INMO_RecommenderWrapper
from SIGIR2022.KGCL_our_interface.KGCL_RecommenderWrapper import KGCL_RecommenderWrapper
from SIGIR2022.RGCF_our_interface.RGCF_RecommenderWrapper import RGCF_RecommenderWrapper
from SIGIR2022.SimGCL_our_interface.SimGCL_RecommenderWrapper import SimGCL_RecommenderWrapper
from SIGIR2022.LightGCN_our_interface.LightGCNRecommender import LightGCNRecommender

from SIGIR2022.KGAT_data.KGAT_DataReader import KGAT_DataReader

from HyperparameterTuning.SearchSingleCase import SearchSingleCase
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs

from optimize_all_baselines import _baseline_tune, _run_algorithm_hyperopt, ExperimentConfiguration, copy_reproduced_metadata_in_baseline_folder
from Utils.plot_popularity import plot_popularity_bias, save_popularity_statistics
from data_statistics import save_data_statistics
from Utils.ResultFolderLoader import ResultFolderLoader

import numpy as np
import os, traceback, argparse
import pandas as pd

from Evaluation.Evaluator import EvaluatorHoldout
from Utils.assertions_on_data_for_experiments import assert_implicit_data, assert_disjoint_matrices

from skopt.space import Real, Integer, Categorical

from HyperparameterTuning.SearchSingleCase import SearchSingleCase
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs


def _GNN_algorithm_tune(experiment_configuration,
                        knowledge_base_df,
                        output_folder_path,
                        use_gpu):


    recommender_class_list = [
        GDE_RecommenderWrapper,
        GTN_RecommenderWrapper,
        HAKG_RecommenderWrapper,
        RGCF_RecommenderWrapper,
        HCCF_RecommenderWrapper,
        INMO_RecommenderWrapper,
        KGCL_RecommenderWrapper,
        SimGCL_RecommenderWrapper,
        LightGCNRecommender,
    ]


    earlystopping_hyperparameters = {"validation_every_n": 5,
                                     "stop_on_validation": True,
                                     "lower_validations_allowed": 5,
                                     "evaluator_object": experiment_configuration.evaluator_validation_earlystopping,
                                     "validation_metric": experiment_configuration.metric_to_optimize,
                                     "epochs_min": 0
                                     }

    for recommender_class in recommender_class_list:

        hyperparameterSearch = SearchBayesianSkopt(recommender_class,
                                                   evaluator_validation=experiment_configuration.evaluator_validation,
                                                   evaluator_test=experiment_configuration.evaluator_test)

        if recommender_class in [HAKG_RecommenderWrapper, KGCL_RecommenderWrapper]:

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS=[experiment_configuration.URM_train, knowledge_base_df],
                CONSTRUCTOR_KEYWORD_ARGS={"use_gpu": use_gpu, "verbose": True},
                FIT_POSITIONAL_ARGS=[],
                FIT_KEYWORD_ARGS={},
                EARLYSTOPPING_KEYWORD_ARGS=earlystopping_hyperparameters,
            )

        else:

            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS=[experiment_configuration.URM_train],
                CONSTRUCTOR_KEYWORD_ARGS={"use_gpu": use_gpu, "verbose": False},
                FIT_POSITIONAL_ARGS=[],
                FIT_KEYWORD_ARGS={},
                EARLYSTOPPING_KEYWORD_ARGS=earlystopping_hyperparameters,
            )

        # GDE does not need the Cython sampler
        if recommender_class not in [GDE_RecommenderWrapper]:
            recommender_input_args.CONSTRUCTOR_KEYWORD_ARGS["use_cython_sampler"] = True


        n_users, n_items = experiment_configuration.URM_train.shape


        global_hyperparameters_range_dictionary = {
            "epochs": Categorical([1000]),
            "batch_size": Categorical([256, 512, 1024, 2048, 4096]),
            "learning_rate": Real(low=1e-6, high=1e-1, prior="log-uniform"),

            "embedding_size": Integer(2, 350),
            # "sgd_mode": Categorical(["sgd", "adagrad", "adam", "rmsprop"]),
            "sgd_mode": Categorical(["adam"]),
        }


        dropout_range = Real(low=0.1, high=0.9, prior="uniform")
        regularization_range = Real(low=1e-6, high=1e-1, prior="log-uniform")


        if recommender_class is GDE_RecommenderWrapper:

            #   ratio must be <= 0.33 due to the lobpcg function
            ratio_high = 0.10

            hyperparameters_range_dictionary = {
                "beta": Real(low=1e-1, high=1e2, prior="log-uniform"),
                "feature_type": Categorical(["smoothed", "both"]),
                "drop_out": dropout_range,
                "reg": regularization_range,

                # For both ratios:
                #   floor(ratio * user_size) and floor(smooth_ratio * item_size) must be >=1
                #   ratio must be <= 0.33 due to the lobpcg function
                "smooth_ratio": Real(low=1/(min(n_users, n_items) -1), high=ratio_high, prior="log-uniform"),
                "rough_ratio": Real(low=1/(min(n_users, n_items) -1), high=ratio_high, prior="log-uniform"),
                "loss_type": Categorical(["adaptive", "bpr"]),
            }


        elif recommender_class is GTN_RecommenderWrapper:

            hyperparameters_range_dictionary = {
                "GNN_layers_K": Integer(1, 6),
                "embedding_smoothness_weight": Integer(1, 15),
                "l2_reg": regularization_range,
                "dropout_rate_lightgcn": dropout_range,
                "dropout_rate_gtn": dropout_range,
            }


        elif recommender_class is HAKG_RecommenderWrapper:

            hyperparameters_range_dictionary = {
                "GNN_layers_K": Integer(1, 6),
                "angle_loss_weight": Real(low=1e-6, high=1e-1, prior="log-uniform"),
                "l2_reg": regularization_range,
                "add_inverse_relation": Categorical([True, False]),
                "node_dropout_rate": dropout_range,
                "mess_dropout_rate": dropout_range,
                "angle_dropout_rate": dropout_range,

                # No criteria was given to select these two
                "n_negative_samples_M": Integer(100, 500),
                "contrastive_loss_margin": Real(low=0.1, high=0.9, prior="uniform"),
            }


        elif recommender_class is HCCF_RecommenderWrapper:

            hyperparameters_range_dictionary = {
                "GNN_layers_K": Integer(1, 6),
                "HYP_layers_C": Integer(1, 4),
                "hyperedge_size": Integer(2, 350),

                "dropout": dropout_range,
                "contrastive_loss_weight": Real(low=1e-7, high=1e-1, prior="log-uniform"),
                "l2_reg": regularization_range,
                "contrastive_loss_temperature_tau": Real(low=1e-2, high=1e0, prior="log-uniform"),

                "leaky_relu_slope": Categorical([0.01]), # Categorical([0.5]), # default is 0.01
                # "learning_rate_decay": Categorical([None]), # Categorical([0.96]),
            }


        elif recommender_class is INMO_RecommenderWrapper:

            hyperparameters_range_dictionary = {

                "K": Integer(1, 6),
                "l2_reg": regularization_range,
                "template_loss_weight": Real(low=1e-4, high=1e-1, prior="log-uniform"),
                # Using the options implemented in the original source code
                "template_node_ranking_metric": Categorical(["page_rank"]), #Categorical(["degree", "sort", "page_rank"]),
                "dropout": dropout_range,
                "template_ratio": Real(low=0.1, high=1.0, prior="uniform"),

                "normalization_decay": Categorical([0.99]),
            }


        elif recommender_class is KGCL_RecommenderWrapper:

            hyperparameters_range_dictionary = {
                "GNN_layers_K": Integer(1, 6),
                "contrastive_loss_temperature_tau": Real(low=1e-2, high=1e0, prior="log-uniform"),

                "GNN_dropout_rate": dropout_range,
                "knowledge_graph_dropout_rate": dropout_range,
                "user_interaction_dropout_rate": dropout_range,

                "mix_ratio": Real(low=0.1, high=0.9, prior="uniform"),
                "uicontrast": Categorical(["WEIGHTED", "WEIGHTED-MIX"]),
                "entities_per_head": Integer(1, 20),
                "l2_reg": regularization_range,
                "self_supervised_loss_weight": Real(low=1e-4, high=1e-1, prior="log-uniform"),
            }


        elif recommender_class is RGCF_RecommenderWrapper:

            hyperparameters_range_dictionary = {
                "GNN_layers_K": Integer(1, 6),
                "prune_threshold_beta": Real(low=1e-3, high=1e0, prior="log-uniform"),
                "contrastive_loss_temperature_tau": Real(low=1e-2, high=1e0, prior="log-uniform"),
                "contrastive_loss_weight": Real(low=1e-7, high=1e-1, prior="log-uniform"),
                "augmentation_ratio": Real(low=0.01, high=0.3, prior="log-uniform"),

                "l2_reg": regularization_range,
            }


        elif recommender_class is SimGCL_RecommenderWrapper:

            hyperparameters_range_dictionary = {
                "GNN_layers_K": Integer(1, 6),
                "noise_magnitude_epsilon": Real(low=1e-2, high=1e0, prior="log-uniform"),
                "contrastive_loss_temperature_tau": Real(low=1e-2, high=1e0, prior="log-uniform"),
                "contrastive_loss_weight": Real(low=1e-7, high=1e-1, prior="log-uniform"),
                "l2_reg": regularization_range,
            }


        elif recommender_class is LightGCNRecommender:

            hyperparameters_range_dictionary = {
                "GNN_layers_K": Integer(1, 6),      # The original paper limits it to 4
                "l2_reg": regularization_range,
                "dropout_rate": dropout_range,
            }



        assert len(hyperparameters_range_dictionary.keys() & global_hyperparameters_range_dictionary.keys()) == 0

        all_hyperparameters = global_hyperparameters_range_dictionary.copy()
        all_hyperparameters.update(hyperparameters_range_dictionary.copy())


        recommender_input_args_last_test = recommender_input_args.copy()
        recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = experiment_configuration.URM_train_last_test

        ## Final step, after the hyperparameter range has been defined for each type of algorithm
        hyperparameterSearch.search(recommender_input_args,
                                    hyperparameter_search_space=all_hyperparameters,
                                    n_cases=experiment_configuration.n_cases,
                                    n_random_starts=experiment_configuration.n_random_starts,
                                    resume_from_saved=experiment_configuration.resume_from_saved,
                                    save_model=experiment_configuration.save_model,
                                    evaluate_on_test=experiment_configuration.evaluate_on_test,
                                    max_total_time=experiment_configuration.max_total_time,
                                    output_folder_path=output_folder_path,
                                    output_file_name_root=recommender_class.RECOMMENDER_NAME,
                                    metric_to_optimize=experiment_configuration.metric_to_optimize,
                                    cutoff_to_optimize=experiment_configuration.cutoff_to_optimize,
                                    recommender_input_args_last_test=recommender_input_args_last_test)






def run_this_algorithm_experiment(dataset_name,
                                  flag_baselines_tune = False,
                                  flag_article_default = False,
                                  flag_article_tune = False,
                                  flag_print_results = False):

    result_folder_path = "result_experiments/{}/{}/".format(ALGORITHM_NAME, dataset_name)
    data_folder_path = result_folder_path + "data/"
    baseline_folder_path = result_folder_path + "baselines/"
    GNN_model_folder_path = result_folder_path + "GNN_models/"

    dataset = KGAT_DataReader(dataset_name, data_folder_path, [0.70, 0.10, 0.20])

    print('Current dataset is: {}'.format(dataset_name))

    URM_train = dataset.URM_DICT["URM_train"].copy()
    URM_validation = dataset.URM_DICT["URM_validation"].copy()
    URM_test = dataset.URM_DICT["URM_test"].copy()
    knowledge_base_df = dataset.knowledge_base_df.copy()

    URM_train_last_test = URM_train + URM_validation

    # Ensure IMPLICIT data and disjoint test-train split
    assert_implicit_data([URM_train, URM_validation, URM_test])
    assert_disjoint_matrices([URM_train, URM_validation, URM_test])

    # If directory does not exist, create
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)

    save_data_statistics(URM_train + URM_validation + URM_test,
                         dataset_name,
                         data_folder_path + "data_statistics")

    plot_popularity_bias([URM_train + URM_validation, URM_test],
                         ["Training data", "Test data"],
                         data_folder_path + "popularity_plot")

    save_popularity_statistics([URM_train + URM_validation, URM_test],
                               ["Training data", "Test data"],
                               data_folder_path + "popularity_statistics")


    metric_to_optimize = 'NDCG'
    cutoff_to_optimize = 20
    cutoff_list = [5, 10, 20, 30, 40, 50, 100]
    max_total_time = 14*24*60*60  # 14 days
    n_cases = 50
    n_processes = 4

    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=cutoff_list)
    evaluator_validation_earlystopping = EvaluatorHoldout(URM_validation, cutoff_list=[cutoff_to_optimize])
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)


    experiment_configuration = ExperimentConfiguration(
             URM_train = URM_train,
             URM_train_last_test = URM_train_last_test,
             ICM_DICT = dataset.ICM_DICT,
             UCM_DICT = dataset.UCM_DICT,
             n_cases = n_cases,
             n_random_starts = int(n_cases/3),
             resume_from_saved = True,
             save_model = "best",
             evaluate_on_test = "best",
             evaluator_validation = evaluator_validation,
             KNN_similarity_to_report_list = KNN_similarity_to_report_list,
             evaluator_test = evaluator_test,
             max_total_time = max_total_time,
             evaluator_validation_earlystopping = evaluator_validation_earlystopping,
             metric_to_optimize = metric_to_optimize,
             cutoff_to_optimize = cutoff_to_optimize,
             n_processes = n_processes,
             )

    ################################################################################################
    ######
    ######      REPRODUCED ALGORITHM
    ######
    
    use_gpu = True

    if flag_article_tune:
        _GNN_algorithm_tune(experiment_configuration, knowledge_base_df, GNN_model_folder_path, use_gpu)

    ################################################################################################
    ######
    ######      BASELINE ALGORITHMS
    ######

    if flag_baselines_tune:
        _baseline_tune(experiment_configuration, baseline_folder_path)

    ################################################################################################
    ######
    ######      PRINT RESULTS
    ######

    if flag_print_results:

        other_algorithm_list = [
            GDE_RecommenderWrapper,
            GTN_RecommenderWrapper,
            HAKG_RecommenderWrapper,
            RGCF_RecommenderWrapper,
            HCCF_RecommenderWrapper,
            INMO_RecommenderWrapper,
            KGCL_RecommenderWrapper,
            SimGCL_RecommenderWrapper,
            LightGCNRecommender,
        ]

        result_loader = ResultFolderLoader(baseline_folder_path,
                                           base_algorithm_list = None,
                                           other_algorithm_list = other_algorithm_list,
                                           KNN_similarity_list=KNN_similarity_to_report_list,
                                           ICM_names_list=dataset.ICM_DICT.keys(),
                                           UCM_names_list=dataset.UCM_DICT.keys(),
                                           )

        result_loader.generate_latex_results(result_folder_path + "{}_{}_{}_latex_results.txt".format(ALGORITHM_NAME, dataset_name, "article_metrics"),
                                             metrics_list=['RECALL', 'NDCG'],
                                             cutoffs_list=[cutoff_to_optimize],
                                             table_title=None,
                                             highlight_best=True)

        result_loader.generate_latex_results(
            result_folder_path + "{}_{}_{}_latex_results.txt".format(ALGORITHM_NAME, dataset_name, "beyond_accuracy_metrics"),
            metrics_list=["NOVELTY", "DIVERSITY_MEAN_INTER_LIST", "COVERAGE_ITEM", "DIVERSITY_GINI", "SHANNON_ENTROPY"],
            cutoffs_list=cutoff_list,
            table_title=None,
            highlight_best=True)

        result_loader.generate_latex_time_statistics(result_folder_path + "{}_{}_{}_latex_results.txt".format(ALGORITHM_NAME, dataset_name, "time"),
                                                     n_evaluation_users = np.sum(np.ediff1d(URM_test.indptr) >= 1),
                                                     table_title=None)


if __name__ == '__main__':

    ALGORITHM_NAME = "ALL"
    CONFERENCE_NAME = "SIGIR22"

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--baseline_tune',        help="Baseline hyperparameter search", type=bool, default=True)
    parser.add_argument('-t', '--article_tune',         help="Reproduced model hyperparameters search", type=bool, default=False)
    parser.add_argument('-p', '--print_results',        help="Print results", type=bool, default=True)

    input_flags = parser.parse_args()
    print(input_flags)

    # Reporting only the cosine similarity is enough
    KNN_similarity_to_report_list = ["cosine"]

    dataset_list = ["yelp2018", "amazon-book"]
    # dataset_list = ["amazon-book"]

    for dataset_name in dataset_list:
        print ("Running dataset: {}".format(dataset_name))
        run_this_algorithm_experiment(dataset_name,
                                   flag_baselines_tune = input_flags.baseline_tune,
                                   flag_article_default = input_flags.article_default,
                                   flag_article_tune = input_flags.article_tune,
                                   flag_print_results = input_flags.print_results,
                                   )
