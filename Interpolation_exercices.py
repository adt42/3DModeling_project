#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 08:27:33 2024

@author: adrientarquinio
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from numpy.polynomial import Polynomial

def f1(t): #ex1
    return 5*t*np.exp(-0.5*t)

def f2(x): #ex2
    return np.sin(x) + np.sin(5*x)
    

#--------------
# EX1

t = np.linspace(0, 10, 11)
y = f1(t)

tq = np.linspace(0, 10, 44)
inter = sc.interpolate.interp1d(t, y, 'linear')

cubicspline = sc.interpolate.CubicSpline(t, y)

plt.figure(figsize=((20,15)))
plt.scatter(t, y, facecolors='none', edgecolors='b', label ='f(t)')
plt.plot(tq, inter(tq), '-r', label ='f(t) interpolation linear')
plt.plot(tq, cubicspline(tq), 'k', label ='f(t) interpolation spline')
plt.xlabel('t')
plt.ylabel('f1(t)')
plt.legend()
plt.title('Linear and Spline Interpolation Comparison (Exercice 1)')
plt.grid()
plt.show()

#--------------------
# EX2
n = 5

x = np.linspace(0, 2*np.pi, n+1)
y2 = f2(x)

cubicspline2 = sc.interpolate.CubicSpline(x, y2)

xq = np.linspace(0, 2*np.pi,200)

poly = Polynomial.fit(x, y2, 6)

error_cubic = np.abs(cubicspline2(xq) - f2(xq))
error_poly = np.abs(poly(xq) - f2(xq))

plt.figure(figsize=((20,15)), num='Exercice 2')

plt.subplot(1,2,1)
plt.scatter(x, y2, facecolors='none', edgecolors='b', label ='Real data')
plt.plot(xq, cubicspline2(xq), 'r', label ='Spline method')
plt.plot(xq, f2(xq), '^k', label ='Real curve')
plt.text(0.75, 2, f'Error:{np.sum(error_cubic)}')
plt.grid()
plt.xlabel('x')
plt.ylabel('f2(x)')
plt.legend()
plt.title('Spline Interpolation')

plt.subplot(1,2,2)
plt.scatter(x, y2, facecolors='none', edgecolors='b', label ='Real data')
plt.plot(xq, poly(xq), 'r', label ='6 order polynome')
plt.plot(xq, f2(xq), '^k', label ='Real curve')
plt.text(0.75, 2, f'Error:{np.sum(error_poly)}')
plt.grid()
plt.xlabel('x')
plt.ylabel('f2(x)')
plt.legend()
plt.title('Polinomial Interpolation')

plt.show()

#------------
