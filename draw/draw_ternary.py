#!/usr/bin/env python
# -*-coding:utf-8 -*-

import ternary
import math
import random


def color_point(x, y, z, scale):
    '''
    Create color point based on (x, y, z)
    '''
    w = 255
    x_color = x * w / float(scale)
    y_color = y * w / float(scale)
    z_color = z * w / float(scale)
    r = math.fabs(w - y_color) / w
    g = math.fabs(w - x_color) / w
    b = math.fabs(w - z_color) / w
    return (r, g, b, 1.)

def generate_heatmap_data(scale):
    '''
    Create data for heatmap
    '''
    d = dict()
    for (i, j, k) in ternary.helpers.simplex_iterator(scale):
        d[(i, j, k)] = color_point(i, j, k, scale)
    return d

def random_points(n, scale):
    '''
    Create n sets of random points (x, y, z)
    Scale = x + y + z
    '''
    p = []
    for i in range(n):
        x = random.random() * scale
        y = random.random() * (scale - x)
        z = scale - x - y
        p.append((x, y, z))
    return p

def shannon_entropy(p):
    '''
    Computes the Shannon Entropy at a distribution in the simplex
    '''
    s = 0.
    for i in range(len(p)):
        try:
            s += p[i] * math.log(p[i])
        except ValueError:
            continue
    return -1.*s

if __name__ == '__main__':
    # Boundary and Gridlines
    scale = 100 
    figure, tax = ternary.figure(scale=scale)

    # Draw Boundary and Gridlines
    tax.boundary(linewidth=2.0)
    tax.gridlines(color="black", multiple=10, linewidth=0.5)
    # tax.gridlines(color="blue", multiple=0.1, linewidth=0.5)

    # Set Axis labels and Title
    fontsize = 12
    tax.set_title("Simplex Boundary and Gridlines", fontsize=fontsize)
    tax.left_axis_label("Left label X", fontsize=fontsize)
    tax.right_axis_label("Right label Y", fontsize=fontsize)
    tax.bottom_axis_label("Bottom label Z", fontsize=fontsize)

    # Set ticks
    tax.ticks(axis='lbr', linewidth=1, multiple=10)

    # Remove default Matplotlib Axes
    tax.clear_matplotlib_ticks()

    # Remove default Matplotlib Spines
    tax.get_axes().spines['top'].set_visible(False)
    tax.get_axes().spines['bottom'].set_visible(False)
    tax.get_axes().spines['left'].set_visible(False)
    tax.get_axes().spines['right'].set_visible(False)

    # Scatter
    # data = random_points(30, scale=scale)
    # tax.scatter(data, marker='s', color='red', label="Red Squares")
    # data = random_points(30, scale=scale)
    # tax.scatter(data, marker='D', color='green', label="Green Diamonds")

    # Heatmap: dual-triangular; hexagonal; triangular
    # data = generate_heatmap_data(scale)
    # tax.heatmap(data, style="hexagonal", use_rgba=True)
    # tax.heatmapf(shannon_entropy, boundary=True, style="triangular")

    # Show
    ternary.plt.show()
