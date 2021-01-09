import textdistance
import networkx as nx

import base_navigator

navigator = base_navigator.BaseNavigator()

class Evaluation:
    def __init__(self, n_examples):
        self.tc = 0
        self.spd = 0
        self.sed =0
        self.n_examples = n_examples


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

    def evaluate_single_sample(self, panoid_pred_list, panoid_gt_list):
        panoid_pred = panoid_pred_list[-1]
        panoid_gt = panoid_gt_list[-1]
        self.tc += self.evaluate_TC(panoid_pred, panoid_gt)
        self.spd += self.evaluate_SPD(panoid_pred,panoid_gt)
        self.sed += self.evaluate_SED(panoid_pred_list, panoid_gt_list)

    def performance_info(self):
        print(f"DEV performance -\n" +
              f"TC: {self.tc / self.n_examples}\n" +
              f"SPD: {self.spd / self.n_examples}\n" +
              f"SED: {self.sed / self.n_examples}")

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