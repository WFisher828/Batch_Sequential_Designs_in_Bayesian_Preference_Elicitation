#!/usr/bin/env python
# coding: utf-8



#This module "Experiment_Framework" has functions which are used in conducting numerical experiments. 
#These functions include:
#1. product_diff_list
#2. question_extractor
#3. norm_AO_MO_data_generation

import numpy as np
import itertools
import random
import scipy.integrate #Feb27 2023
import math # Feb 27 2023

import time

#import sys
#sys.path.append(r'C:\Users\wsfishe\Desktop\PreferenceElicitationCode')
from Baseline_Functions_Definitions import z_expectation_variance
from Questionnaire_Procedure import moment_matching_update



#Define a set that has all the differences between binary products

def product_diff_list(n):
    #n: the number of attributes of the products
    
    #Example of itertools.product:
    #itertools.product(range(2), repeat = 3) --> 000 001 010 011 100 101 110 111
    p_d_l = list(itertools.product([-1.0,0.0,1.0],repeat = n))
    
    #Return the index of the tuple with all 0s.
    zero_index = p_d_l.index(tuple([0]*n))

    #Note that at this point, product_diff_list contains some redundant information. Due to
    #the symmetry of the one-step and two-step acquisition function in terms of question mean, question pairs such as 
    #(-1,-1,-1,...,-1) and (1,1,1,...,1) (i.e. negative multiples) will evaluate as the same under the one-step and two-step
    #acquisition functions. Due to the structure of product_diff_list, we can remove every question pair before and including
    #the question pair with all zero entries in order to remove this redundant information.
    for i in range(0,zero_index + 1):
        p_d_l.pop(0)
    
    p_d_l = [np.array(a) for a in p_d_l]
    return p_d_l



#Given a trinary vector of 0, 1, and -1, find two binary products whose difference is the trinary vector.
def question_extractor(prod):
    #prod: This is a trinary vector of 0, 1, and -1 that represents the difference between two products, which
    #are represented by binary vectors.
    
    x = [0]*len(prod)
    y = [0]*len(prod)
    for i in range(0,len(prod)):
        if prod[i] == 1.0:
            x[i] = 1.0
            y[i] = 0.0
        if prod[i] == 0.0:
            x[i] = 0.0
            y[i] = 0.0
        if prod[i] == -1.0:
            x[i] = 0.0
            y[i] = 1.0
    return x,y



#This function is used to generate data to estimate the parameters in the normalized AO model. The normalized AO model
#is given by log(D-err/Det^(1/2)(Sig)) ~ AM/||L*mu|| + AV/||S*Sig|| + AO/||S*Sig|| + ||L*mu|| + ||S*Sig||. AM, AV, and AO denote
#the average question mean, average quesiton variance, and average question orthogonality of a given design under prior
#N(mu, Sig). L and S denote varying signal and noise levels, respectively. The normalized AO model is used in our
#optimization procedure so that we will not have to refit the parameters in the optimization model everytime the user
#answers a batch. The idea is that varying L and S enough should encompass a wide enough range so that mu and Sig will
#be within this range after updating.
#In this function, we also include MO so that we may fit a maximum orthogonality model.

#!!! rng will need to be set before calling this function !!!

def norm_AO_MO_data_generation(init_mu, init_Sig, batch_size, L, S, num_random_batches, num_true_partworths):
    #init_mu: This is the initial expectation of the partworths.
    #init_Sig: This is the initial covariance matrix of the partworths.
    #batch_size: This is the number of questions in each batch
    #L: This is a vector which holds varying levels of signal (multiply with mu). For example,
    #we could have L = [0.25,1.0,4.0]
    #S: This is a vector which holds varying levels of noise (multiply with Sig). For example,
    #we could have S = [0.25,1.0,4.0]
    #num_random_batches: This is the number of random batches that we will generate for collecting data on log(D-err),
    #AM, AV, and AO (and MO). This set of random batches will be used for each level combination of L and S.
    #num_true_partworths: This is the number of true/baseline partworths we will use to evaluate the d-error of a design.
    
    attr_num = len(init_mu)
    
    #Create lists to store average orthogonality and max orthogonality, as well as d-error and average question mean and
    #average question variance, and ||L*mu|| and ||S*Sig|| as well.
    average_orthogonality = []
    
    maximum_orthogonality = []
    
    average_question_mean = []
    
    average_question_variance = []
    
    average_d_error = []
    
    L_mu = []
    
    S_Sig = []
    
    init_sqrt_determinant = []
    
    #Create a list of all products.
    prod_list = product_diff_list(attr_num)
    
    #Construct the set of batch designs
    batch_set = [[] for i in range(num_random_batches)]
    for i in range(num_random_batches):
        random_question_matrix = random.sample(prod_list,batch_size)
        for m in range(batch_size):
            #random_question_vec = random.sample(prod_list,1)[0]
            #[x,y] = question_extractor(random_question_vec)
            [x,y] = question_extractor(random_question_matrix[m])
            batch_set[i].append([x,y])
    
    #Record the scaled norm of mu and Sig for each combination of L and S
    for l in L:
        for s in S:
            for i in range(num_random_batches):
                L_mu.append(l*np.linalg.norm(init_mu,2))
                S_Sig.append(s*np.linalg.norm(init_Sig,2))
                init_sqrt_determinant.append(np.sqrt(np.linalg.det(s*init_Sig)))
    
    #Calculate AM, AV, AO and MO for each of the batches
    for l in L:
        for s in S:
            for i in range(num_random_batches):
                random_batch_question_mean = []
                random_batch_question_variance = []
                random_batch_orthogonality = []
        
                for p in range(batch_size):
                    x_p = np.array(batch_set[i][p][0])
                    y_p = np.array(batch_set[i][p][1])
                    random_batch_question_mean.append(np.abs(np.dot(l*init_mu,x_p - y_p)))
                    random_batch_question_variance.append(np.dot(x_p - y_p, np.dot(s*init_Sig,x_p - y_p)))
                    for q in range(p+1, batch_size):
                        x_q = np.array(batch_set[i][q][0])
                        y_q = np.array(batch_set[i][q][1])
                        random_batch_orthogonality.append(np.abs(np.dot(x_p - y_p, np.dot(s*init_Sig,x_q - y_q))))
                
                #We use this if statement in case the batch size is 1 because
                #if the batch size is 1 then there are no orthogonality terms.
                if len(random_batch_orthogonality) > 0:
                    average_orthogonality.append(np.mean(np.array(random_batch_orthogonality)))
                    maximum_orthogonality.append(np.max(np.array(random_batch_orthogonality)))
        
                average_question_mean.append(np.mean(np.array(random_batch_question_mean)))
                average_question_variance.append(np.mean(np.array(random_batch_question_variance)))
            
    #Calculate the D-error.
    for l in L:
        #print('L: '+str(l))
        for s in S:
            #print('S: '+str(s))
            true_partworths = []
            for t in range(num_true_partworths):
                #This is where rng needs to be set beforehand!
                true_partworths.append(rng.multivariate_normal(l*init_mu,s*init_Sig))
                
            gumbel_errors = [[[np.random.gumbel(0,1) for k in range(2)] for j in range(batch_size)] for i in range(num_true_partworths)]
            
            for i in range(num_random_batches):
                #Create a list for the batch that will store the final determinant value for each simulation
                #corresponding to each baseline partworth.
                batch_simulate_d_values = []
                
                #Simulate d-efficiency over baseline partworths
                for j in range(len(true_partworths)):
                #Each time we start with a new partworth, we must use the initial prior parameters.
                    mu = l*init_mu
                    Sig = s*init_Sig
                    
                    #Each simulation goes through the questions in the random batch.
                    for k in range(batch_size):
                    #Set x and y
                        x = batch_set[i][k][0]
                        y = batch_set[i][k][1]
                
                        #These temp variables will be used in the choice model below in case the user prefers y over x.
                        x_temp = x
                        y_temp = y
                        
                        gum_x = gumbel_errors[j][k][0]
                        gum_y = gumbel_errors[j][k][1]
                        #See preference between two products
                        if (np.dot(true_partworths[j],np.array(y)) + gum_y) >= (np.dot(true_partworths[j],np.array(x)) + gum_x):
                            x = y_temp
                            y = x_temp
                            
                        #Perform moment matching after choice is made.
                        [mu, Sig] = moment_matching_update(x,y,mu,Sig)
                        
                    #After the questionnaire for a baseline partworth is complete, we append the square root of the determinant
                    #of the final covariance matrix.
                    batch_simulate_d_values.append(np.sqrt(np.linalg.det(Sig)))
                    
                #We average the d-values from the simulation for a batch and store it in a list.
                average_d_error.append(np.mean(batch_simulate_d_values))
                
    return average_orthogonality, maximum_orthogonality, average_question_mean, average_question_variance, L_mu, S_Sig, init_sqrt_determinant, average_d_error