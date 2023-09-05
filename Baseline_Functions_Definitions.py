#!/usr/bin/env python
# coding: utf-8


#This module "Baseline_Functions_Definitions" includes functions that the questionnaire procedure and experimental framework
#are based off of. The functions include:
#1. z_expectation_variance
#2. g_fun
#4. g_fun_linear_regression



import numpy as np
import pandas as pd
import scipy.integrate
import math
from sklearn import linear_model


#Compute expectation and variance of Z random variable parameterized by m and v

def z_expectation_variance(m,v):
    #m is question mean
    #v is question variance
    
    #MARCH 28 2023: CHANGED TO 35 (from 30) due to some errors in calculation. At integ_bound = 30, there were instances 
    #where variance was calculated to be less than 0. 
    integ_bound = 35.0
    
    #Set up functions for calculating expectation and variance of Z
    fun1 = lambda z: ((1 + math.e**(-m - math.sqrt(v)*z))**(-1))*(math.e**((-z**2)/2))/math.sqrt(2*math.pi)
    fun2 = lambda z: z*((1 + math.e**(-m - math.sqrt(v)*z))**(-1))*(math.e**((-z**2)/2))/math.sqrt(2*math.pi)
    fun3 = lambda z: z*z*((1 + math.e**(-m - math.sqrt(v)*z))**(-1))*(math.e**((-z**2)/2))/math.sqrt(2*math.pi)
    
    #Calculate expectation and variance of Z. C is the normalization constant that ensures the pdf of Z
    #integrates to be 1. 
    C = scipy.integrate.quad(fun1, -integ_bound, integ_bound)[0]
    mu_z = scipy.integrate.quad(fun2, -integ_bound, integ_bound)[0]/C
    var_z = (scipy.integrate.quad(fun3, -integ_bound, integ_bound)[0] / C) - mu_z**2
    
    return [mu_z, var_z]


def g_fun(m,v,n=2):
    #m and v are question mean and variance arguments
    
    #A bound for integration so to prevent numerical issues.
    #MARCH 28 2023: CHANGED TO 35 (from 30) due to some errors in calculation. At integ_bound = 30, there were instances 
    #where variance was calculated to be less than 0. 
    integ_bound = 35.0
    
    #fun1 represents the pdf of the Z(m,v) random variable before normalizing
    fun1 = lambda z: ((1 + math.e**(-m - math.sqrt(v)*z))**(-1))*(math.e**((-z**2)/2))/math.sqrt(2*math.pi)
    
    #fun2 will be used for calculating the expectation of Z(m,v)
    fun2 = lambda z: z*((1 + math.e**(-m - math.sqrt(v)*z))**(-1))*(math.e**((-z**2)/2))/math.sqrt(2*math.pi)
    
    #fun3 will be used in calculating the variance of Z(m,v)
    fun3 = lambda z: z*z*((1 + math.e**(-m - math.sqrt(v)*z))**(-1))*(math.e**((-z**2)/2))/math.sqrt(2*math.pi)
    
    #fun4 will be used in calculating the expectation of Z(-m,v)
    fun4 = lambda z: z*(1-(1 + math.e**(-m - math.sqrt(v)*z))**(-1))*(math.e**((-z**2)/2))/math.sqrt(2*math.pi)
    
    #fun5 will be used in calculating the variance of Z(-m,v)
    fun5 = lambda z: z*z*(1-(1 + math.e**(-m - math.sqrt(v)*z))**(-1))*(math.e**((-z**2)/2))/math.sqrt(2*math.pi)
    
    #C is the normalization constant of the pdf for Z. This is the p(m,v) term.
    C = scipy.integrate.quad(fun1, -integ_bound, integ_bound)[0]
    
    #Calculate variance for Z(m,v) and Z(-m,v)
    if C < 10**(-6):
        C = 0
        Sig_z_1 = 1
    else:
        mu_z_1 = scipy.integrate.quad(fun2, -integ_bound, integ_bound)[0]/C
        Sig_z_1 = (scipy.integrate.quad(fun3, -integ_bound, integ_bound)[0]/C) - mu_z_1**2
    
    if (1-C) < 10**(-6):
        C = 1
        Sig_z_2 = 1
    else:
        mu_z_2 = scipy.integrate.quad(fun4, -integ_bound, integ_bound)[0]/(1-C)
        Sig_z_2 = (scipy.integrate.quad(fun5, -integ_bound, integ_bound)[0]/(1-C)) - mu_z_2**2
    
    #print('Left-branch, right-branch value: ' + str(Sig_z_1) + ' ' + str(Sig_z_2))

    #g_fun can take on arbitrary n
    return [C*Sig_z_1**(1/float(n)) + (1-C)*Sig_z_2**(1/float(n)), C*Sig_z_1**(1/float(n)), (1-C)*Sig_z_2**(1/float(n))]


#A function for doing linear regression on the log(g_fun).
def g_fun_linear_regression(m_lower,m_upper,v_lower,v_upper,m_axis_num,v_axis_num):
    #m_lower is the lower bound for the grid in variable m
    #m_upper is the upper bound for the grid in variable m
    #v_lower is the lower bound for the grid in variable v
    #v_upper is the upper bound for the grid in variable v
    #m_axis_num and v_axis_num are the number of points used to make the m-axis and v-axis.
    #The total number of grid points is m_axis_num*v_axis_num.
    
    #construct grid data for m and v variables
    m_grid = np.linspace(m_lower, m_upper, num = m_axis_num)
    v_grid = np.linspace(v_lower, v_upper, num = v_axis_num)
    
    #initiate an array to collect data on log(g)
    gfun_array = np.zeros((m_axis_num*v_axis_num,3))
    
    #Format/collect data
    for i in range(0,m_axis_num):
        for j in range(0,v_axis_num):
            #print(m_grid[i],v_grid[j])
            gfun_array[i*v_axis_num + j] = [m_grid[i],v_grid[j],math.log(g_fun(m_grid[i],v_grid[j])[0])]
        
    df_gfunction = pd.DataFrame(gfun_array, columns = ['m','v','g'])

    Y = df_gfunction['g'] # dependent variable
    X = df_gfunction[['m', 'v']] # independent variable
    lm = linear_model.LinearRegression()
    lm.fit(X, Y) # fitting the model

    m_coeff = lm.coef_[0]
    v_coeff = lm.coef_[1]

    return[m_coeff,v_coeff]