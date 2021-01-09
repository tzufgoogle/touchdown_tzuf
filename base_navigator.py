import os
from graph_loader import GraphLoader
import  numpy as np

class BaseNavigator:
    def __init__(self):
        self.graph, self.nx_graph = GraphLoader().construct_graph()

        self.graph_state = None
        self.prev_graph_state = None

    def navigate(self):
        raise NotImplementedError

    def step(self, go_towards):
        '''
        Execute one step and update the state. 
        go_towards: ['forward', 'left', 'right']
        '''
        next_panoid, next_heading = self._get_next_graph_state(self.graph_state, go_towards)

        if len(self.graph.nodes[next_panoid].neighbors) < 2:
            # stay still when running into the boundary of the graph
            print(f'At the border (number of neighbors < 2). Did not go "{go_towards}".')
            return
        self.prev_graph_state = self.graph_state
        self.graph_state = (next_panoid, next_heading)
        
    def _get_next_graph_state(self, curr_state, go_towards):
        '''Get next state without changing the current state.'''
        curr_panoid, curr_heading = curr_state

        if go_towards == 'forward':
            neighbors = self.graph.nodes[curr_panoid].neighbors
            if curr_heading in neighbors:
                # use current heading to point to the next node
                next_node = neighbors[curr_heading]
            else:
                # weird node, stay put
                next_node = self.graph.nodes[curr_panoid]
        elif go_towards == 'left' or go_towards == 'right':
            # if turn left or right, stay at the same node 
            next_node = self.graph.nodes[curr_panoid]
        elif go_towards == 'stop':
            return curr_panoid, curr_heading
        else:
            raise ValueError('Invalid action.')

        next_panoid = next_node.panoid
        next_heading = self._get_nearest_heading(curr_state, next_node, go_towards)
        return next_panoid, next_heading

    def _get_nearest_heading(self, curr_state, next_node, go_towards):
        _, curr_heading = curr_state
        next_heading = None

        diff = float('inf')
        if go_towards == 'forward':
            diff_func = lambda next_heading, curr_heading: 180 - abs(abs(next_heading - curr_heading) - 180)
        elif go_towards == 'left':
            diff_func = lambda next_heading, curr_heading: (curr_heading - next_heading) % 360
        elif go_towards == 'right':
            diff_func = lambda next_heading, curr_heading: (next_heading - curr_heading) % 360
        else:
            return curr_heading

        for heading in next_node.neighbors.keys():
            if heading == curr_heading and go_towards != 'forward':
                # don't match to the current heading when turning
                continue
            diff_ = diff_func(int(heading), int(curr_heading))
            if diff_ < diff:
                diff = diff_
                next_heading = heading

        if next_heading is None:
            next_heading = curr_heading
        return next_heading

    def get_available_next_moves(self, graph_state):
        '''Given current node, get available next actions and states.'''
        next_actions = ['forward', 'left', 'right']
        next_graph_states = [
            self._get_next_graph_state(graph_state, 'forward'),
            self._get_next_graph_state(graph_state, 'left'),
            self._get_next_graph_state(graph_state, 'right')
        ]
        return next_actions, next_graph_states

    def show_state_info(self, graph_state):
        '''Given a graph state, show current state information and available next moves.'''
        print('Current graph state: {}'.format(graph_state))
        available_actions, next_graph_states = self.get_available_next_moves(graph_state)

        print('Available next actions and graph states:')
        for action, next_graph_state in zip(available_actions, next_graph_states):
            print('Action: {}, to graph state: {}'.format(action, next_graph_state))
        print('==============================')

    def turn_to_node(self, next_graph_states, curr_heading, next_panoid_gt):
        # left
        curr_panoid, left_heading = next_graph_states[1]
        next_panoid, next_heading = self._get_next_graph_state((curr_panoid, left_heading), "forward")

        if next_panoid == next_panoid_gt:
            return "left", left_heading

        else:
            curr_panoid, right_heading = next_graph_states[2]
            next_panoid, next_heading = self._get_next_graph_state((curr_panoid, right_heading), "forward")
            return "right", right_heading


    def get_actions(self, seq_route_panoids, start_heading):
        next_actions_dict = {'forward': 0, 'left': 1, 'right': 2}
        list_actions = []
        list_action_numeric = []
        list_binary_actions = []
        panoids_list = []
        heading_list = []
        execution_list = []
        heading = start_heading
        panoids_list.append(seq_route_panoids[0])
        list_binary_actions.append([np.zeros(4)])
        for idx in range(len(seq_route_panoids)-1):
            curr_img = seq_route_panoids[idx]
            curr_state = (curr_img, heading)
            next_panoid_ground_tf = seq_route_panoids[idx+1]
            while True: # not forward
                next_actions, next_graph_states = self.get_available_next_moves(curr_state)
                forward_panoid, forward_heading = next_graph_states[0]
                if forward_panoid == next_panoid_ground_tf: # forward is available.
                    heading = forward_heading
                    list_actions.append("forward")
                    binary_action = np.zeros(4)
                    binary_action[0] = 1
                    list_action_numeric.append(0)
                    list_binary_actions.append([binary_action])
                    panoids_list.append(forward_panoid)
                    heading_list.append(heading)
                    execution_list.append((curr_state, 0))
                    break
                else:
                    # One turn is needed.
                    action, next_heading = self.turn_to_node(next_graph_states, heading, next_panoid_ground_tf)
                    heading = next_heading
                    list_actions.append(action)
                    action_idx = next_actions_dict[action]
                    binary_action = np.zeros(4)
                    binary_action[action_idx] = 1
                    list_binary_actions.append([binary_action])
                    list_action_numeric.append(action_idx)
                    panoids_list.append(curr_img)
                    heading_list.append(heading)
                    execution_list.append((curr_state, action_idx))

                curr_state = (curr_img, heading)

        assert len(list_actions) >= len(seq_route_panoids)
        assert len(list_binary_actions) == len(panoids_list)
        assert len(list_action_numeric)+1 == len(list_binary_actions)
        assert len(execution_list) == len(list_actions)
        list_actions.append("stop")
        list_action_numeric.append(3)
        panoids_list.append(panoids_list[-1])
        list_binary_actions = np.concatenate(list_binary_actions)
        return list_actions, list_action_numeric, panoids_list, execution_list, list_binary_actions






