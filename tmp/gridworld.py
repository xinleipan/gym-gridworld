"""
Implementation of gridworld MDP. 
Copyright @ xinleipan
    
Xinlei Pan, 2017
xinleipan@gmail.com
"""

import numpy as np
from PIL import Image as Image
import matplotlib.pyplot as plt
import time,pdb,sys
import copy
from gym import spaces

# 0: black; 1 : gray; 2 : blue; 3 : green; 4 : red
COLORS = {0:[0.0,0.0,0.0], 1:[0.5,0.5,0.5], 2:[0,0,1], 3:[0,1,0], 4:[1,0,0], 6:[1,0,1], 7:[1,1,0]} 

class GridWorld(object):
    """ GridWorld MDP """

    def __init__(self, grid_size, grid_map_path, verbose=False, restart=False, show_partial=False):
        self.actions = (0,1,2,3,4) # stay, move up, down, left, right
        self.action_pos = {0:[0,0], 1:[-1, 0], 2:[1, 0], 3:[0,-1], 4:[0,1]}
        self.action_space = spaces.Discrete(len(self.actions))
        self.grid_size = grid_size
        self.states = np.random.randn(grid_size, grid_size, 3) * 0.0
        self.verbose = verbose
        self.show_part = show_partial
        self.grid_map_path = grid_map_path
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
        self.grid_unit = int(grid_size/grid_map_array.shape[0])
        self.obs = np.random.randn(int(grid_size/grid_map_array.shape[0]*3), \
                                    int(grid_size/grid_map_array.shape[0]*3),3)

        if grid_size % grid_map_array.shape[0] != 0:
            sys.exit('Grid size must be multiplies of grid map array shape!')
        self.grid_map_array_shape = grid_map_array.shape[0]
        self.start_grid = copy.deepcopy(grid_map_array)      
        self.grid_map_array = copy.deepcopy(grid_map_array)
        self.states = self.update_state(self.grid_map_array)
        self.start = (0,0)
        self.target = (0,0)
        for i in range(self.grid_map_array_shape):
            for j in range(self.grid_map_array_shape):
                this_value = grid_map_array[i,j]
                if this_value == 4:
                    self.start = (i,j)
                if this_value == 3:
                    self.target = (i,j)
        self.position = copy.deepcopy(self.start)
        self.start_state = copy.deepcopy(self.states)
        self.restart = copy.deepcopy(restart)
        self.obs = self.update_obs(self.states)        
        if verbose == True:
            self.fig = plt.figure(1)
            plt.show(block=False)
            plt.axis('off')
            self.render()

    def update_obs(self, states):
        obs = states[int((self.position[0]-1)*self.grid_unit):int((self.position[0]+2)*self.grid_unit), int((self.position[1]-1)*self.grid_unit):int((self.position[1]+2)*self.grid_unit),:]
        return obs

    def reset(self):
        self.position = copy.deepcopy(self.start)
        self.grid_map_array = copy.deepcopy(self.start_grid)
        self.states = copy.deepcopy(self.start_state)
        self.obs = self.update_obs(self.states)
        if self.verbose:
            self.render()
        if self.show_part:
            return self.obs
        elif self.show_part == False:
            return self.states

    def retarget(self, sp):
        """ set the environment start position """
        if self.start[0]==sp[0] and self.start[1]==sp[1]:
            self.reset()
            return
        elif self.start_grid[sp[0], sp[1]] != 0:
            return
        else:
            s_pos = copy.deepcopy(self.start)
            self.start_grid[s_pos[0],s_pos[1]] = 0
            self.start_grid[sp[0], sp[1]] = 4
            self.grid_map_array = copy.deepcopy(self.start_grid)
            self.start = (sp[0], sp[1])
            self.states = self.update_state(self.grid_map_array)
            self.start_state = copy.deepcopy(self.states)
            self.position = copy.deepcopy(self.start)
            self.reset()
            if self.verbose:
                self.render()
        return
    
    def change_target(self, tg):
        """ set the environment target position """
        if self.target[0] == tg[0] and self.target[1] == tg[1]:
            self.reset()
            return
        elif self.start_grid[tg[0], tg[1]] != 0:
            return
        else:
            t_pos = copy.deepcopy(self.target)
            self.start_grid[t_pos[0], t_pos[1]] = 0
            self.start_grid[tg[0], tg[1]] = 3
            self.grid_map_array = copy.deepcopy(self.start_grid)
            self.target = (tg[0], tg[1])
            self.states = self.update_state(self.grid_map_array)
            self.start_state = copy.deepcopy(self.states)
            self.position = copy.deepcopy(self.start)
            self.reset()
            if self.verbose:
                self.render()
        return        
    
    def update_state(self, grid_map_array):
        state = np.random.randn(self.grid_size, self.grid_size, 3)
        state = state * 0.0
        gs = int(self.grid_size/self.grid_map_array_shape) # grid step
        for i in range(self.grid_map_array_shape):
            for j in range(self.grid_map_array_shape):
                for k in range(3):
                    this_value = COLORS[grid_map_array[i,j]][k]
                    state[i*gs : (i+1)*gs , j*gs : (j+1)*gs, k] = this_value
        return state                
    
    def render(self):
        if self.verbose == False:
            return
        if self.show_part:
            self.obs = self.states[int((self.position[0]-1)*self.grid_unit):int((self.position[0]+2)*self.grid_unit), int((self.position[1]-1)*self.grid_unit):int((self.position[1]+2)*self.grid_unit),:]
            img = self.obs
        else:
            img = self.states
        fig = plt.figure(1)
        plt.clf()  
        plt.imshow(img)
        fig.canvas.draw()
        plt.pause(0.00001)
        
    def step(self, action):
        """ return next state, reward, finished, success """
        tmp_pos = (self.position[0] + self.action_pos[action][0],
                   self.position[1] + self.action_pos[action][1])
        if action == 0:
            self.obs = self.update_obs(self.states)
            if self.show_part:
                return (self.obs, -1, False, True)
            else:
                return (self.states, -1, False, True)
        if tmp_pos[0] < 0 or tmp_pos[0] > self.grid_map_array_shape or tmp_pos[1] < 0 or tmp_pos[1] > self.grid_map_array_shape:
            if self.verbose:
                self.render()
            self.obs = self.update_obs(self.states)
            if self.show_part:
                return (self.obs, -1, False, False)
            else:
                return (self.states, -1, False, False)
        # update stateq
        org_color = self.grid_map_array[self.position[0], self.position[1]]
        new_color = self.grid_map_array[tmp_pos[0], tmp_pos[1]]
        if new_color == 0 : # black
            # print('new color is black')
            tmp_color = self.grid_map_array[self.position[0], self.position[1]]
            if tmp_color == 4:
                self.grid_map_array[self.position[0], self.position[1]] = 0
                self.grid_map_array[tmp_pos[0], tmp_pos[1]] = tmp_color
            elif tmp_color == 6 or tmp_color == 7:
                self.grid_map_array[self.position[0], self.position[1]] = tmp_color - 4
                self.grid_map_array[tmp_pos[0], tmp_pos[1]] = 4
            self.position = copy.deepcopy(tmp_pos)
        elif new_color == 1: # gray
            if self.verbose:
                self.render()
            # print('new color is gray')
            self.obs = self.update_obs(self.states)
            if self.show_part:
                return (self.obs, -1, False, False)
            else:
                return (self.states, -1, False, False)
        elif new_color == 2 or new_color == 3: # blue or green
            self.grid_map_array[self.position[0], self.position[1]] = 0
            self.grid_map_array[tmp_pos[0], tmp_pos[1]] = new_color + 4
            self.position = copy.deepcopy(tmp_pos)
            
        self.states = self.update_state(self.grid_map_array)
        self.obs = self.update_obs(self.states)
        if self.verbose:
            self.render()
        if tmp_pos[0] == self.target[0] and tmp_pos[1] == self.target[1]:
            target_state = copy.deepcopy(self.states)
            if self.restart:
                self.states = copy.deepcopy(self.start_state)
                self.position = copy.deepcopy(self.start)
                self.grid_map_array = copy.deepcopy(self.start_grid)
                self.obs = self.update_obs(self.states)
            if self.verbose:
                self.render()
            if self.show_part:
                return (self.update_obs(target_state), 2, True, True)
            else:
                return (target_state, 2, True, True)
        else:
            if self.show_part:
                return (self.update_obs(self.states), 0, False, True)
            else:
                return (self.states, 0, False, True) 
