#!/usr/bin/env python
# -*-coding:utf-8 -*-

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from numpy.random import randint, rand, choice
import numpy as np

class GA():
    def __init__(self, func, pop_size, iter_max, boundaries, precision):
        # input
        self.func = func
        self.pop_size = pop_size
        self.iter_max = iter_max
        self.boundaries = boundaries
        self.precision = precision

        # hype-parameters of GA
        self.dim_size = len(self.boundaries)
        self.bit_size = self.init_bit_size()
        self.gene_size = self.bit_size * self.dim_size
        self.value_max = 2**self.bit_size - 1
        self.mutation_rates = sorted(np.linspace(
            1e-3, 0.5, self.iter_max).tolist(), reverse=True)
        self.mutation_rate = self.mutation_rates[0]

        # variables
        self.pop = self.init_pop()
        self.Y = []
        self.X = []

        # results for plot
        self.res_Y = []
        self.res_Y_min = []
        self.res_iter = 0

    def init_bit_size(self):
        '''
        Initialize the bit size based on boundaries and precision
        '''
        temp = []
        for i in range(self.dim_size):
            t = self.boundaries[i][1] - self.boundaries[i][0]
            temp.append(t)
        index = temp.index(max(temp))
        precision_bit = self.precision / (self.boundaries[index][1] - self.boundaries[index][0])
        value_max = 1 / precision_bit
        bit_size = round(np.log2(value_max))
        return bit_size

    def init_pop(self):
        '''
        Initialize the population array and return
        Cautions: 
        * Each parameter set corresponds to one gene
        * The gene of each individual in population is unique
        '''
        n = 0
        pop = []
        while n != self.pop_size:
            p = randint(0, 2, self.gene_size).tolist()
            if p not in pop:
                pop.append(p)
                n += 1
        return pop

    def decode(self, pop):
        '''
        Decode the binary data into real data
        Steps:
        * Transform the population array into string format
        * Decode the population with string format into real data
        '''
        pop_bit = [''.join([str(g) for g in gene]) for gene in pop]
        data = [[self.boundaries[i][0] +
                 (int(gene_str[(self.bit_size * i):(self.bit_size * (i + 1))], 2) / self.value_max) *
                 (self.boundaries[i][1] - self.boundaries[i][0])
                 for i in range(self.dim_size)]
                for gene_str in pop_bit]
        return data

    def selection(self, only_sort=True):
        '''
        Elitist selection (modified): evaluate the fitness/loss and select the parent
        Steps:
        * Sort the population based on fitness
        * Selection rate for each individual is based on the squence of sorted population
        * Return the selected individual as parent for next generation
        '''
        if only_sort:
            self.pop = [p for l, p in sorted(zip(self.Y, self.pop))]
            self.X = [p for l, p in sorted(
                zip(self.Y, self.X))]
            self.Y = sorted(self.Y)
        else:
            probabilities = [(self.pop_size - i) /
                             self.pop_size for i in range(self.pop_size - 1)]
            probabilities /= np.array(probabilities).sum()
            indexes_group = [i + 1 for i in range(self.pop_size - 1)]
            index_selected = choice(
                indexes_group, 1, replace=False, p=probabilities).tolist()[0]
            parent = self.pop[index_selected].copy()
            return parent

    def crossover(self, parents):
        '''
        Create two new individuals by one point crossover
        Cautions:
        * Parents should be different
        * Crossover point shouldn't appear at both ends
        '''
        c_point = randint(2, self.gene_size - 1, 1)[0]
        # print(f'crossover point: \n{c_point}')
        c1, c2 = parents[0].copy(), parents[1].copy()
        child1 = c1[:c_point] + c2[c_point:]
        child2 = c2[:c_point] + c1[c_point:]
        return child1, child2

    def mutation(self, child):
        '''
        Create new individual by mutation based on mutation rate
        Cautions:
        * The mutation occurs during the crossover
        * The mutation rate should be large at the beginning, then becomes samller
        * Random choose several points (based on bit number and dim num) in gene to mutate
        '''
        if rand() < self.mutation_rate:
            m_num = (self.bit_size // 4) * self.dim_size
            n = 0
            m_points = []
            while n != m_num:
                m_point = randint(0, self.gene_size, 1)[0]
                if m_point not in m_points:
                    m_points.append(m_point)
                    n += 1
            # print(f'mutation points: \n{m_points}')
            for i, c in enumerate(child):
                if i in m_points:
                    if c == 1:
                        child[i] = 0
                    else:
                        child[i] = 1
        return child

    def run(self):
        '''
        Run the GA iterations
        Cautions:
        * Update mutation rate
        * Recode fitness/loss for every iterations
        '''
        # 1st iteration without sort
        self.X = self.decode(self.pop)
        self.Y = [self.func(s) for s in self.X]
        total_cal = self.pop_size
        # data for visualize
        x_real = np.linspace(0, 10, 1000)
        y_real = software(x_real).tolist()
        x_real = x_real.tolist()
        fig = plt.figure()
        # GA iterations
        old_y = min(self.Y)
        for i in range(self.iter_max):
            self.res_iter = i + 1
            # only sort the population
            self.selection(only_sort=True)
            parent_best = self.pop[0].copy()
            # print and record
            print(f'Iter: {self.res_iter}; Y: {self.Y[0]}')
            self.res_Y_min.append(self.Y[0])
            for y in self.Y:
                self.res_Y.append(y)
            # stop genetic iteration based on precision
            new_y = self.Y[0]
            if self.res_iter > 1 and old_y - new_y <= 0:
               break
            else:
                old_y = new_y
            # visualize the results
            plt.clf()
            sns.set(font_scale=0.8, style='ticks')
            matplotlib.rcParams['xtick.direction'] = 'in'
            matplotlib.rcParams['ytick.direction'] = 'in'
            plt.plot(x_real, y_real, linestyle='dashed', zorder=1)
            x_pre = np.array([x[0] for x in self.X])
            plt.scatter(x_pre, software(x_pre), c='gray', zorder=2)
            plt.scatter(self.X[0][0], software(self.X[0][0]), c='r', zorder=3)
            plt.text(0, max(y_real) * 0.95, f'X: {self.X[0][0]}\nY: {software(self.X[0][0])}',
                fontdict={'fontsize': 8, 'color': 'r', 'weight': 'bold'})
            plt.pause(0.5)
            plt.ioff()
            # Create children with size same as population size - 1
            children = [parent_best]
            while True:
                parent = self.selection(only_sort=False)
                child1, child2 = self.crossover([parent_best, parent])
                for child in [child1, child2]:
                    children.append(self.mutation(child))
                    if len(children) == self.pop_size:
                        break
                if len(children) == self.pop_size:
                    break
            self.pop = children
            # check the same item in population
            indexes_group = [[i for i, item in enumerate(
                self.pop) if item == p] for p in self.pop]
            indexes_same = []
            for indexes in indexes_group:
                if indexes not in indexes_same:
                    indexes_same.append(indexes)
            # call the software and get results
            self.X = self.decode(self.pop)
            self.Y = np.ones(self.pop_size).tolist()
            total_cal += len(indexes_same)
            for indexes in indexes_same:
                y = self.func(self.X[indexes[0]])
                for index in indexes:
                    self.Y[index] = y
            # update muation rate
            self.mutation_rate = self.mutation_rates[i]
        plt.close()
        print(f'\nX: {self.X[0]}')
        print(f'Y: {self.Y[0]}')
        print(f'Actual / Maxmum calculations: {total_cal} / {self.pop_size * self.res_iter}\n')

# %% main.py

def software(x):
    y = x + 10 * np.cos(5 * x) + 7 * np.sin(4 * x)
    return y

def call_software(x):
    '''
    Call the software and calculate the loss
    '''
    x = x[0]
    y = software(x)
    loss = (y - (-12.387779954714805))**2
    return loss

pop_size = 100
iter_max = 20
boundaries = [[0, 10]]

precision = 1e-6
target_prop = [4.66, 13.98, 23.3]

for i in range(5):
    ga = GA(call_software, pop_size, iter_max, boundaries, precision)
    ga.run()
    # visualize GA results
    sns.set(font_scale=0.8, style='ticks')
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    fig, ax = plt.subplots(2, 1, dpi=330)
    X = np.array([(np.ones(ga.pop_size) * (i + 1))
                  for i in range(ga.res_iter)]).ravel()
    ax[0].scatter(X, ga.res_Y, c='r', s=2)
    ax[0].set_ylabel('Y')
    X_min = np.linspace(1, ga.res_iter, ga.res_iter)
    ax[1].plot(X_min, ga.res_Y_min, c='gray',
               linestyle='dashed', linewidth=1, zorder=1)
    ax[1].scatter(X_min, ga.res_Y_min, c='r', s=4, zorder=2)
    ax[1].set_ylabel('Minimum Y')
    ax[1].set_xlabel('Generations')
    plt.savefig(f'res_{i + 1}.png')
    plt.close()