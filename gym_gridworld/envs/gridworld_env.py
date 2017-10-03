import gym
import sys
import os
import time
import copy
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from PIL import Image as Image
import matplotlib.pyplot as plt

# define colors
# 0: black; 1 : gray; 2 : blue; 3 : green; 4 : red
COLORS = {0:[0.0,0.0,0.0], 1:[0.5,0.5,0.5], \
          2:[0.0,0.0,1.0], 3:[0.0,1.0,0.0], \
          4:[1.0,0.0,0.0], 6:[1.0,0.0,1.0], \
          7:[1.0,1.0,0.0]}

class GridworldEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    num_env = 0 
    def __init__(self):
        self.actions = [0, 1, 2, 3, 4]
        self.inv_actions = [0, 2, 1, 4, 3]
        self.action_space = spaces.Discrete(5)
        self.action_pos_dict = {0: [0,0], 1:[-1, 0], 2:[1,0], 3:[0,-1], 4:[0,1]}
 
        ''' set observation space '''
        self.obs_shape = [128, 128, 3]  # observation space shape
        self.observation_space = spaces.Box(low=0, high=1, shape=self.obs_shape)
    
        ''' initialize system state ''' 
        this_file_path = os.path.dirname(os.path.realpath(__file__))
        self.grid_map_path = os.path.join(this_file_path, 'plan4.txt')        
        self.start_grid_map = self._read_grid_map(self.grid_map_path) # initial grid map
        self.current_grid_map = copy.deepcopy(self.start_grid_map)  # current grid map
        self.observation = self._gridmap_to_observation(self.start_grid_map)
        self.grid_map_shape = self.start_grid_map.shape

        ''' agent state: start, target, current state '''
        self.agent_start_state, _ = self._get_agent_start_target_state(
                                    self.start_grid_map)
        _, self.agent_target_state = self._get_agent_start_target_state(
                                    self.start_grid_map)
        self.agent_state = copy.deepcopy(self.agent_start_state)

        ''' set other parameters '''
        self.restart_once_done = False  # restart or not once done
        self.verbose = False # to show the environment or not

        GridworldEnv.num_env += 1
        self.this_fig_num = GridworldEnv.num_env 
        if self.verbose == True:
            self.fig = plt.figure(self.this_fig_num)
            plt.show(block=False)
            plt.axis('off')
            self._render()

    def _step(self, action):
        ''' return next observation, reward, finished, success '''
        action = int(action)
        nxt_agent_state = (self.agent_state[0] + self.action_pos_dict[action][0],
                            self.agent_state[1] + self.action_pos_dict[action][1])
        if action == 0: # stay in place
            return (self.observation, 0, False, True) 
        if nxt_agent_state[0] < 0 or nxt_agent_state[1] >= self.grid_map_shape[0]:
            return (self.observation, -0.1, False, False)
        if nxt_agent_state[1] < 0 or nxt_agent_state[1] >= self.grid_map_shape[1]:
            return (self.observation, -0.1, False, False)
        # successful behavior
        org_color = self.current_grid_map[self.agent_state[0], self.agent_state[1]]
        new_color = self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]]
        if new_color == 0:
            if org_color == 4:
                self.current_grid_map[self.agent_state[0], self.agent_state[1]] = 0
                self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]] = 4
            elif org_color == 6 or org_color == 7:
                self.current_grid_map[self.agent_state[0], self.agent_state[1]] = org_color - 4
                self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]] = 4
            self.agent_state = copy.deepcopy(nxt_agent_state)
        elif new_color == 1: # gray
            return (self.observation, -0.1, False, False)
        elif new_color == 2 or new_color == 3:
            self.current_grid_map[self.agent_state[0], self.agent_state[1]] = 0
            self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]] = new_color + 4
            self.agent_state = copy.deepcopy(nxt_agent_state)
        self.observation = self._gridmap_to_observation(self.current_grid_map)
        self._render()
        if nxt_agent_state[0] == self.agent_target_state[0] and nxt_agent_state[1] == self.agent_target_state[1] :
            target_observation = copy.deepcopy(self.observation)
            if self.restart_once_done:
                self.observation = self._reset()
                return (self.observation, 10, True, True)
            else:
                return (target_observation, 10, True, True)
        else:
            return (self.observation, 0, False, True)
 
    def _reset(self):
        self.agent_state = copy.deepcopy(self.agent_start_state)
        self.current_grid_map = copy.deepcopy(self.start_grid_map)
        self.observation = self._gridmap_to_observation(self.start_grid_map)
        self._render()
        return self.observation

    def _read_grid_map(self, grid_map_path):
        grid_map = open(grid_map_path, 'r').readlines()
        grid_map_array = []
        for k1 in grid_map:
            k1s = k1.split(' ')
            tmp_arr = []
            for k2 in k1s:
                try:
                    tmp_arr.append(int(k2))
                except:
                    pass
            grid_map_array.append(tmp_arr)
        grid_map_array = np.array(grid_map_array)
        return grid_map_array

    def _get_agent_start_target_state(self, start_grid_map):
        start_state = None
        target_state = None
        for i in range(start_grid_map.shape[0]):
            for j in range(start_grid_map.shape[1]):
                this_value = start_grid_map[i,j]
                if this_value == 4:
                    start_state = [i,j]
                if this_value == 3:
                    target_state = [i,j]
        if start_state is None or target_state is None:
            sys.exit('Start or target state not specified')
        return start_state, target_state

    def _gridmap_to_observation(self, grid_map, obs_shape=None):
        if obs_shape is None:
            obs_shape = self.obs_shape
        observation = np.random.randn(*obs_shape)*0.0
        gs0 = int(observation.shape[0]/grid_map.shape[0])
        gs1 = int(observation.shape[1]/grid_map.shape[1])
        for i in range(grid_map.shape[0]):
            for j in range(grid_map.shape[1]):
                for k in range(3):
                    this_value = COLORS[grid_map[i,j]][k]
                    observation[i*gs0:(i+1)*gs0, j*gs1:(j+1)*gs1, k] = this_value
        return observation
  
    def _render(self, mode='human', close=False):
        if self.verbose == False:
            return
        img = self.observation
        fig = plt.figure(self.this_fig_num)
        plt.clf()
        plt.imshow(img)
        fig.canvas.draw()
        plt.pause(0.00001)
        return 
 
    def change_start_state(self, sp):
        ''' change agent start state '''
        ''' Input: sp: new start state '''
        if self.agent_start_state[0] == sp[0] and self.agent_start_state[1] == sp[1]:
            _ = self._reset()
            return True
        elif self.start_grid_map[sp[0], sp[1]] != 0:
            return False
        else:
            s_pos = copy.deepcopy(self.agent_start_state)
            self.start_grid_map[s_pos[0], s_pos[1]] = 0
            self.start_grid_map[sp[0], sp[1]] = 4
            self.current_grid_map = copy.deepcopy(self.start_grid_map)
            self.agent_start_state = [sp[0], sp[1]]
            self.observation = self._gridmap_to_observation(self.current_grid_map)
            self.agent_state = copy.deepcopy(self.agent_start_state)
            self._reset()
            self._render()
        return True
        
    
    def change_target_state(self, tg):
        if self.agent_target_state[0] == tg[0] and self.agent_target_state[1] == tg[1]:
            _ = self._reset()
            return True
        elif self.start_grid_map[tg[0], tg[1]] != 0:
            return False
        else:
            t_pos = copy.deepcopy(self.agent_target_state)
            self.start_grid_map[t_pos[0], t_pos[1]] = 0
            self.start_grid_map[tg[0], tg[1]] = 3
            self.current_grid_map = copy.deepcopy(self.start_grid_map)
            self.agent_target_state = [tg[0], tg[1]]
            self.observation = self._gridmap_to_observation(self.current_grid_map)
            self.agent_state = copy.deepcopy(self.agent_start_state)
            self._reset()
            self._render()
        return True
    
    def get_agent_state(self):
        ''' get current agent state '''
        return self.agent_state

    def get_start_state(self):
        ''' get current start state '''
        return self.agent_start_state

    def get_target_state(self):
        ''' get current target state '''
        return self.agent_target_state

    def _jump_to_state(self, to_state):
        ''' move agent to another state '''
        if self.current_grid_map[to_state[0], to_state[1]] == 0:
            if self.current_grid_map[self.agent_state[0], self.agent_state[1]] == 4:
                self.current_grid_map[self.agent_state[0], self.agent_state[1]] = 0
                self.current_grid_map[to_state[0], to_state[1]] = 4
                self.observation = self._gridmap_to_observation(self.current_grid_map)
                self.agent_state = [to_state[0], to_state[1]]
                self._render()
                return (self.observation, 0, False, True)
            if self.current_grid_map[self.agent_state[0], self.agent_state[1]] == 6:
                self.current_grid_map[self.agent_state[0], self.agent_state[1]] = 2
                self.current_grid_map[to_state[0], to_state[1]] = 4
                self.observation = self._gridmap_to_observation(self.current_grid_map)
                self.agent_state = [to_state[0], to_state[1]]
                self._render()
                return (self.observation, 0, False, True)
            if self.current_grid_map[self.agent_state[0], self.agent_state[1]] == 7:  
                self.current_grid_map[self.agent_state[0], self.agent_state[1]] = 3
                self.current_grid_map[to_state[0], to_state[1]] = 4
                self.observation = self._gridmap_to_observation(self.current_grid_map)
                self.agent_state = [to_state[0], to_state[1]]
                self._render()
                return (self.observation, 0, False, True)
        elif self.current_grid_map[to_state[0], to_state[1]] == 4:
            return (self.observation, 0, False, True)
        elif self.current_grid_map[to_state[0], to_state[1]] == 1:
            return (self.observation, -1, False, False)
        elif self.current_grid_map[to_state[0], to_state[1]] == 3:
            self.current_grid_map[self.agent_state[0], self.agent_state[1]] = 0
            self.current_grid_map[to_state[0], to_state[1]] = 7
            self.agent_state = [to_state[0], to_state[1]]
            self.observation = self._gridmap_to_observation(self.current_grid_map)
            self._render()
            if self.restart_once_done:
                self.observation = self._reset()
                return (self.observation, 1, True, True)
            return (self.observation, 1, True, True)
        else:
            return (self.observation, -1, False, False)

    def _close_env(self):
        plt.close(1)
        return
    
    def jump_to_state(self, to_state):
        a, b, c, d = self._jump_to_state(to_state)
        return (a, b, c, d) 
