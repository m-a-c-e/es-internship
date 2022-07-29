#!/usr/bin/python
# Author: Carissa Bush
# Date:   15 July 2022

'''
This script takes in a function and two sets of inputs 
and runs two separate threads of the function, once 
with each set of input data
'''

from threading import Thread
import numpy as np 
import copy
import cv2


# # custom thread
class CustomThread(Thread):
    '''
    creates a subclass of Thread that allows threads 
    to return values from the functions they run
    '''
    def __init__(self, fun, args):  # constructor
        Thread.__init__(self)       # execute the base constructor
        self.returns = None

        self.function = fun         # input function
        self.arguments = args       # list of arguments for above function

    def run(self):                  # function executed in a new threads
        self.returns = self.function(*self.arguments)


def double_thread(fun, arg1, arg2):
    '''
    Inputs:
        fun:    function that with be run twice with threads
        [arg1]: list of arguments to be run in the first thread
        [arg2]: list of arguments to be run in the second thread

    Outputs:
        x_data: outputs of the function run with args1
        y_data: outputs of the function run with args2      
    '''
    x = CustomThread(fun, arg1)
    x.start()
    y = CustomThread(fun, arg2)
    y.start()

    x.join()
    y.join()

    x_data = x.returns
    y_data = y.returns

    return x_data, y_data


def six_thread(fun, arg1, arg2):
    l_coords = []
    l_boxes = []

    r_coords = []
    r_boxes = []

    l_threads = []
    r_threads = []

    res_l_final = None
    res_r_final = None
 
    for i in range(1, 4):
        l_t = CustomThread(fun, [arg1[0], arg1[i*2 - 1], arg1[i*2]])
        l_threads.append(l_t)
    
    for i in range(1, 4):
        r_t = CustomThread(fun, [arg2[0], arg2[i*2 - 1], arg2[i*2]])
        r_threads.append(r_t)

    for i in range(0, 3):
        l_threads[i].start()
        r_threads[i].start()

    for i in range(0, 3):
        l_threads[i].join()
    
    for i in range(0, 3):
        r_threads[i].join()

    for i in range(0, 3):
        l_data, l_b, res_l = l_threads[i].returns
        r_data, r_b, res_r = r_threads[i].returns

        if res_l_final is None or res_r_final is None:
            res_l_final = copy.copy(res_l)
            res_r_final = copy.copy(res_r)
        else:
            res_l_final += res_l
            res_r_final += res_r

        if len(l_data) == 0:
            pass
        elif len(l_coords) == 0:
            l_coords = copy.copy(l_data)
        else:
            l_coords = np.concatenate((l_coords, l_data), axis=0)

        if len(r_data) == 0:
            pass
        elif len(r_coords) == 0:
            r_coords = copy.copy(r_data)
        else:
            r_coords = np.concatenate((r_coords, r_data), axis=0)
        
        l_boxes += l_b
        r_boxes += r_b

    return l_coords, l_boxes, res_l_final, r_coords, r_boxes, res_r_final