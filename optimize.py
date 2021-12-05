#!/usr/bin/env python
# -*-coding:utf-8 -*-

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
        self.bit_size = 64
        self.dim_size = len(self.boundaries)
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
        Elitist selection: keep the best one without crossover and mutation
        Steps:
        * Evaluate the fitness/loss and select the parent
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
            if self.res_iter > 1 and old_y - new_y <= self.precision:
               break
            else:
                old_y = new_y
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
        print(f'\nX: {self.X[0]}')
        print(f'Y: {self.Y[0]}')
        print(f'Actual / Maxmum calculations: {total_cal} / {self.pop_size * self.res_iter}\n')