from absl import logging
import numpy as np
import os
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from transformers import AdamW
from torch.utils.data import DataLoader
from os import path
from PIL import Image
from torchvision.transforms import ToTensor
import time
import networkx as nx
import textdistance

import base_navigator
import evaluation


navigator = base_navigator.BaseNavigator()

MAIN_DIR = './data/images'
ACTION_DICT = {'forward': 0, 'left': 1, 'right': 2, 'stop': 3}
ACTION_DICT_IDX = {0:'forward', 1:'left', 2:'right', 3:'stop'}



class Trainer:
    def __init__(
            self,
            model,
            device,
            optimizer,
            train_loader,
            dev_loader,
            num_epochs = 10
    ):

        self.model = model

        self.device = device

        self.optimizer = optimizer
        self.train_loader = train_loader
        self.dev_loader = dev_loader

        self.loss = nn.CrossEntropyLoss()
        self.num_epochs = num_epochs

    def evaluate_TC(self, panoid_pred, panoid_gt):
        '''
        Task completion (TC) measures the accuracy of completing the task correctly.
        We consider an execution correct if the agent reaches the exact goal position
        or one of its neighboring nodes in the environment graph.
        :return:
        0 if  accuracy.
        '''
        if panoid_gt == panoid_pred:
            return 1
        neighbors = [x for x in navigator.nx_graph.neighbors(panoid_pred)]
        if panoid_gt in neighbors:
            return 1
        return 0

    def evaluate_SPD(self, panoid_pred, panoid_gt):
        '''
         Shortest-path distance (SPD) measures the mean distance in the graph between
         the agent’s final panorama and the goal.
         SPD ignores turning actions and the agent heading.
        :return:
        SPD measurement.
        '''
        return nx.shortest_path_length(navigator.nx_graph, panoid_pred, panoid_gt)

    def evaluate_SED(self, list_panoid_pred, list_panoid_gt):
        '''
         Success weighted by edit distance (SED).
         SED is related to success weighted by path length (SPL),
         but is designed for instruction following in graph based environments,
         where a specific correct path exists.
        :return:
        SED measurement.
        '''
        lev_dist = textdistance.levenshtein.distance(list_panoid_gt,list_panoid_pred)
        max_len = max(len(list_panoid_gt), len(list_panoid_pred))
        return 1-(lev_dist/max_len)

    # def get_dtw_matrix(observed_panos, golden_panos, distance_fn):
    #     """Dynamic Time Warping (DTW).
    #     Muller, Meinard. "Dynamic time warping."
    #     Information retrieval for music and motion (2007): 69-84.
    #     Dynamic Programming implementation, O(NM) time and memory complexity.
    #     Args:
    #       observed_panos: List of observed pano ids or names.
    #       golden_panos: List of golden pano ids or names.
    #       distance_fn: Method for getting the distance between two panos.
    #     Returns:
    #       A 2-D matrix with DTW scores.
    #     """
    #     num_obs_panos = len(observed_panos)
    #     num_golden_panos = len(golden_panos)
    #
    #     dtw_matrix = np.inf * np.ones((num_obs_panos + 1, num_golden_panos + 1))
    #     dtw_matrix[0][0] = 0
    #     for i in range(num_obs_panos):
    #         for j in range(num_golden_panos):
    #             best_prev_cost = min(
    #                 dtw_matrix[i][j],  # Move both
    #                 dtw_matrix[i + 1][j],  # Move query
    #                 dtw_matrix[i][j + 1]  # Move reference
    #             )
    #             cost = distance_fn(observed_panos[i], golden_panos[j])
    #             dtw_matrix[i + 1][j + 1] = cost + best_prev_cost
    #
    #     return dtw_matrix
    #
    # def evaluate_ndtw(self, panoid_pred_list, golden_path):
    #     '''
    #     Normalized Dynamic Time Warping (nDTW): a minimized cumulative distance
    #     between the agent path and true path, normalized by path length.
    #     :return:
    #     '''
    #     distance_fn = navigator.
    #     dtw_matrix = self.get_dtw_matrix(agent_path, golden_path,
    #                                                  distance_fn)
    #     dtw = dtw_matrix[len(agent_path)][len(golden_path)]
    #     pln_dtw = dtw / len(golden_path)
    #     ndtw = tf.math.exp(-1. * dtw / (_SUCCESS_THRESHOLD * len(golden_path)))


    def evaluate(self, panoid_pred_list, panoid_gt_list):
        panoid_pred = panoid_pred_list[-1]
        panoid_gt = panoid_gt_list[-1]
        tc = self.evaluate_TC(panoid_pred, panoid_gt)
        spd = self.evaluate_SPD(panoid_pred,panoid_gt)
        sed = self.evaluate_SED(panoid_pred_list, panoid_gt_list)
        return tc, spd, sed


    def train_model(self):

        '''Main function for training model.'''
        # Initialize running values.
        global_step = 0
        len_dev = len(self.dev_loader)
        # Training loop.
        self.model.train()

        for epoch in range(self.num_epochs):
            running_loss = 0.0
            logging.info("Epoch number: {}".format(epoch))
            print ("epoch ", epoch)
            self.model.train()
            for batch_idx, batch in enumerate(self.train_loader):
                print (batch_idx)
                text = {key: val.to(self.device) for key, val in batch['text'].items()}
                images_features = batch['images_features'].to(self.device).squeeze(0)
                actions_target = batch['actions'].to(self.device).squeeze(0)

                actions_binary = batch['actions_binary'].to(self.device).squeeze(0)
                actions_output = self.model(text, images_features, actions_binary)

                loss = self.loss(actions_output, actions_target)

                self.optimizer.zero_grad()
                loss.backward()
                running_loss+=loss
                self.optimizer.step()

            print ("loss:", running_loss)

            ##############
            self.model.eval()
            evaluator = evaluation.Evaluation(len_dev)
            with torch.no_grad():
                for batch_idx, batch in enumerate(self.dev_loader):
                    print (batch_idx)
                    text = {key: val.to(self.device) for key, val in batch['text'].items()}
                    images_feature = batch['images_features'][:,0].to(self.device)
                    route_panoids = batch['route_panoids']
                    route_panoids = [x[0] for x in route_panoids]
                    first_panoid = route_panoids[0]

                    panormid_pred = []
                    action_list=[]
                    panormid_pred.append(first_panoid)
                    curr_state = (first_panoid,0)
                    action = -1
                    action_binary = torch.zeros(4)
                    while action!= ACTION_DICT['stop']:
                        actions_output = self.model(text, images_feature, action_binary.unsqueeze(0).to(self.device))
                        topv, topk = actions_output.data.topk(2)
                        action = topk.squeeze().tolist()[0]
                        new_state = navigator._get_next_graph_state(curr_state, ACTION_DICT_IDX[action])
                        if action==ACTION_DICT['forward']:
                            if new_state[0] == curr_state[0]: # Couldn't move forward.
                                action = topk.squeeze().tolist()[1]
                                new_state = navigator._get_next_graph_state(curr_state, ACTION_DICT_IDX[action])
                            else:
                                panormid_pred.append(new_state[0])
                        curr_state = new_state
                        panoid = curr_state[0]
                        feature_path = path.abspath(path.join(MAIN_DIR, 'features', panoid + '.pt'))
                        images_feature = torch.load(feature_path).to(self.device)
                        action_list.append(action)
                        action_binary = torch.zeros(4)
                        action_binary[action] = 1
                        if len(action_list)>36:
                            action=3
                            print ("!!!!!")
                    evaluator.evaluate_single_sample(panormid_pred, route_panoids)

                evaluator.performance_info()

        print("END")

