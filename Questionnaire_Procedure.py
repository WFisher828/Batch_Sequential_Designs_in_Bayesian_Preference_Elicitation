#!/usr/bin/env python
# coding: utf-8


#This module 'Questionnaire_Procedure' has a function which is able to be used to perform
#one-step update. This function is:
#1. moment_matching_update

import numpy as np
import scipy.integrate
import math
import random # MARCH 3 2023

#import sys
#sys.path.append(r'C:\Users\wsfishe\Desktop\PreferenceElicitationCode')
from Baseline_Functions_Definitions import z_expectation_variance

def moment_matching_update(x,y,mu_prior,Sig_prior):
    #x and y are a question pair, x is preferred over y.
    #mu_prior and Sig_prior are expectation and covariance matrix
    #Make sure x, y, mu_prior, and Sig_prior are numpy arrays
    x_vec = np.array(x)
    y_vec = np.array(y)
    mu_prior_vec = np.array(mu_prior)
    Sig_prior_vec = np.array(Sig_prior)
    
    #Define question expectation and question variance
    v = x_vec - y_vec
    mu_v = np.dot(mu_prior_vec,v)
    Sig_dot_v = np.dot(Sig_prior_vec,v)
    Sig_v = np.dot(v,Sig_dot_v)
    
    #Save np.dot(Sig_prior_vec,v) as a variable (DONE)
    
    #Calculate expectation and variance for Z random variable
    
    [mu_z, var_z] = z_expectation_variance(mu_v,Sig_v)
    
    
    #Calculate the update expectation and covariance matrix for 
    #posterior
    mu_posterior = mu_prior_vec + (mu_z/math.sqrt(Sig_v))*Sig_dot_v
    Sig_posterior = ((var_z-1)/Sig_v)*np.outer(Sig_dot_v,Sig_dot_v) + Sig_prior_vec
    
    return mu_posterior, Sig_posterior