"""
Implementation of gridworld MDP. 
Copyright @ xinleipan
    
Xinlei Pan, 2017
xinleipan@gmail.com
"""

import numpy as np
from PIL import Image as Image
import matplotlib.pyplot as plt
import time,pdb
import copy

# 0: black; 1 : gray; 2 : blue; 3 : green; 4 : red
COLORS = {0:[0.0,0.0,0.0], 1:[0.5,0.5,0.5], 2:[0,0,1], 3:[0,1,0], 4:[1,0,0], 6:[1,0,1], 7:[1,1,0]} 

class GridWorld(object):
    """ GridWorld MDP """

    def __init__(self, grid_size, grid_map_path, verbose=False, restart=False):
        self.actions = (0,1,2,3,4) # stay, move up, down, left, right
        self.action_pos = {0:[0,0], 1:[-1, 0], 2:[1, 0], 3:[0,-1], 4:[0,1]}
        self.grid_size = grid_size
        self.states = np.random.randn(grid_size, grid_size, 3) * 0.0
        self.verbose = verbose
        if grid_size % 32 != 0:
            sys.exit('Grid size must be mutliplies of 32!')
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
        
        self.start_grid = copy.deepcopy(grid_map_array)      
        self.grid_map_array = copy.deepcopy(grid_map_array)
        self.states = self.update_state(self.grid_map_array)
        self.start = (0,0)
        self.target = (0,0)
        for i in range(32):
            for j in range(32):
                this_value = grid_map_array[i,j]
                if this_value == 4:
                    self.start = (i,j)
                if this_value == 3:
                    self.target = (i,j)
        self.position = copy.deepcopy(self.start)
        self.start_state = copy.deepcopy(self.states)
        self.restart = copy.deepcopy(restart)
        if verbose == True:
            self.fig = plt.figure(1)
            plt.show(block=False)
            plt.axis('off')
            self.render()

    def reset(self):
        self.position = copy.deepcopy(self.start)
        self.grid_map_array = copy.deepcopy(self.start_grid)
        self.states = copy.deepcopy(self.start_state)
        if self.verbose:
            self.render()
        return self.states

    def update_state(self, grid_map_array):
        state = np.random.randn(self.grid_size, self.grid_size, 3)
        state = state * 0.0
        for i in range(32):
            for j in range(32):
                for k in range(3):
                    this_value = COLORS[grid_map_array[i,j]][k]
                    state[i*8 : (i+1)*8 , j*8 : (j+1)*8, k] = this_value
        return state                
    
    def render(self):
        if self.verbose == False:
            return
        img = self.states   
        fig = plt.figure(1)
        plt.clf()  
        plt.imshow(img)
        fig.canvas.draw()
        plt.pause(0.001)
        
    def step(self, action):
        """ return next state, reward, finished, success """
        tmp_pos = (self.position[0] + self.action_pos[action][0],
                   self.position[1] + self.action_pos[action][1])
        if action == 0:
            return (self.states, -1, False, True)
        if tmp_pos[0] < 0 or tmp_pos[0] > 31 or tmp_pos[1] < 0 or tmp_pos[1] > 31:
            if self.verbose:
                self.render()
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
            return (self.states, -1, False, False)
        elif new_color == 2 or new_color == 3: # blue or green
            # print('new color is blue or green')
            self.grid_map_array[self.position[0], self.position[1]] = 0
            self.grid_map_array[tmp_pos[0], tmp_pos[1]] = new_color + 4
            self.position = copy.deepcopy(tmp_pos)
            
        self.states = self.update_state(self.grid_map_array)
        if self.verbose:
            self.render()
        if tmp_pos[0] == self.target[0] and tmp_pos[1] == self.target[1]:
            target_state = copy.deepcopy(self.states)
            # pdb.set_trace()
            if self.restart:
                self.states = copy.deepcopy(self.start_state)
                self.position = copy.deepcopy(self.start)
                self.grid_map_array = copy.deepcopy(self.start_grid)
            if self.verbose:
                self.render()
            return (target_state, 2, True, True)
        else:
            return (self.states, 1, False, True) 
