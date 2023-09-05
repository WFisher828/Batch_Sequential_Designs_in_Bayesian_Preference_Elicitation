#!/usr/bin/env python
# coding: utf-8

# In[1]:


#This module 'Batch_Design_and_Rollout' includes functions to construct a batch design as well as
#doing the rollout method.

#Functions in this module include:

#orthogonal_constraint_feas
#batch_design_delta_penalty #THIS WILL BE PREFERRED OVER batch_design_delta_refine
#batch_design_AO #!!!!THIS WAS ADDED ON JUNE 20, 2023!!!!
#batch_design_delta_refine
#question_selection_prob
#rollout
#monte_carlo_rollout
#rollout_with_batch_design_acquisition


# In[2]:


import gurobipy as gp
from gurobipy import GRB
import numpy as np
import sklearn.datasets
import scipy.integrate
import scipy.stats #import bernoulli
import math
from matplotlib import pyplot as plt

import time


import sys
sys.path.append(r'C:\Users\wsfishe\Desktop\PreferenceElicitationCode')
from Baseline_Functions_Definitions import z_expectation_variance, g_fun, g_fun_linear_regression
from Questionnaire_Procedure import moment_matching_update, g_opt, two_step_g_acq


# In[3]:


#THIS WILL NEED TO BE MOVED!!!
#mu_log_coeff_6,Sig_log_coeff_6 = g_fun_linear_regression(0,12.70,0.1,42.0,24,84)
#print(mu_log_coeff_6,Sig_log_coeff_6)


# In[4]:


#THIS WILL NEED TO BE MOVED!!!
#This is used to instantiate the coefficient values for optimization when there are 12 attributes.
#mu_log_coeff_12,Sig_log_coeff_12 = g_fun_linear_regression(0, 24.50, 0.1, 156.0, 49, 312)
#print(mu_log_coeff_12,Sig_log_coeff_12)


# In[5]:


#This function is used to check the feasibility of the orthogonality constraints used in the batch design for a given 
#tolerance delta. That is, it checks the feasibility of -delta <= q_i * Sig * q_j <= delta for i not equal to j
def orthogonal_constraint_feas(mu,Sig,delta,batch_size,t_lim=30):
    #mu: expectation of prior on beta
    #Sig: Covariance matrix of prior on beta
    #delta: This is a parameter in the batch-design model that is used to control the level of orthogonality of 
    #the various questions in the batch design.
    #batch_size: the number of questions we want to return in our batch design. This should be less or equal to the number
    #of attributes
    #t_lim: a time limit on the running time of the optimization procedure. Not sure if t=100 is sufficient at the moment.
    
    #This is the number of attributes for the products
    n = len(Sig[0])
    
    m = gp.Model("mip1")
    m.setParam('Timelimit',t_lim)
    
    #Set up the x_i and y_i, i = 1,...,batchsize
    X = m.addMVar((batch_size,n),vtype = GRB.BINARY)
    Y = m.addMVar((batch_size,n),vtype = GRB.BINARY)
    
    #Set up the objective function. We use an objective function of 0 just to check the feasibility of the orthogonality
    #constraints
    m.setObjective(0,GRB.MINIMIZE)
    
    #Set up the constraints that force the products in question i to be different
    for i in range(batch_size):
        m.addConstr(X[i]@X[i] - X[i]@Y[i] - Y[i]@X[i] + Y[i]@Y[i] >= 1)
        m.addConstr(mu@X[i] - mu@Y[i] >= 0)
    
    #Set up the Sigma-orthogonality constraint for all questions i and j, i not equal to j.
    for i in range(batch_size):
        for j in range(i+1,batch_size):
            m.addConstr(X[i]@Sig@X[j] - X[i]@Sig@Y[j] - Y[i]@Sig@X[j] + Y[i]@Sig@Y[j] - delta <= 0)
            m.addConstr(X[i]@Sig@X[j] - X[i]@Sig@Y[j] - Y[i]@Sig@X[j] + Y[i]@Sig@Y[j] + delta >= 0)
            
    m.optimize()
    
    feasibility = True
    
    #If the time limit was exceeded for finding a feasible solution, we assume the problem is infeasible.
    if m.Status == GRB.TIME_LIMIT:
        feasibility = False
        
    return feasibility


# In[6]:


#This function constructs a batch design of size k <= (number of attributes) where we enforce mutual 
#Sigma-orthogonality between the k questions. The orthogonality condition makes it so that the D-error minimization
#can be written as a product of g (one-step lookahead) functions. For this function, delta is considered as a continuous
#variable and is penalized in the objective function.

def batch_design_delta_penalty(mu,Sig,batch_size,mu_log_coeff,Sig_log_coeff,M=100,t_lim=100):
    #mu: expectation of prior on beta
    #Sig: Covariance matrix of prior on beta
    #batch_size: the number of questions we want to return in our batch design. This should be less or equal to the number
    #of attributes
    #mu_log_coeff: the estimated coefficient c_1 that goes with m in the linear model log(g) = c_1*m + c_2*v
    #Sig_log_coeff: the estimated coefficient c_2 that goes with v in the linear model log(g) = c_1m + c_2*v
    #M: this is a parameter which will be used as a constant to penalize the orthogonality constraint term delta.
    #t_lim: a time limit on the running time of the optimization procedure. Not sure if t=100 is sufficient at the moment.
    
    #This is the number of attributes for the products
    n = len(Sig[0])
    
    #These are terms corresponding to the linear and quadratic terms in the objective function.
    mu_s = mu_log_coeff*mu
    Sig_s = Sig_log_coeff*Sig
    
    m = gp.Model("mip1")
    m.setParam('Timelimit',t_lim)
    #m.params.NonConvex = 2
    
    #Set up the x_i and y_i, i = 1,...,batchsize
    X = m.addMVar((batch_size,n),vtype = GRB.BINARY)
    Y = m.addMVar((batch_size,n),vtype = GRB.BINARY)
    delta = m.addVar(lb=0.0, vtype = GRB.CONTINUOUS)
    
    #Set up the objective function, which is the sum of (batch_size) linearized g functions.
    m.setObjective(sum([mu_s@X[i] - mu_s@Y[i] + X[i]@Sig_s@X[i] -X[i]@(2.0*Sig_s)@Y[i] + 
                   Y[i]@Sig_s@Y[i]  for i in range(batch_size)]) + M*delta,GRB.MINIMIZE)
    
    #Set up the constraints that force the products in question i to be different, as well as forcing the symmetry
    #exploitation condition.
    for i in range(batch_size):
        m.addConstr(X[i]@X[i] - X[i]@Y[i] - Y[i]@X[i] + Y[i]@Y[i] >= 1)
        m.addConstr(mu@X[i] - mu@Y[i] >= 0)
    
    #Set up the Sigma-orthogonality constraint for all questions i and j, i not equal to j.
    for i in range(batch_size):
        for j in range(i+1,batch_size):
            m.addConstr(X[i]@Sig@X[j] - X[i]@Sig@Y[j] - Y[i]@Sig@X[j] + Y[i]@Sig@Y[j] - delta <= 0)
            m.addConstr(X[i]@Sig@X[j] - X[i]@Sig@Y[j] - Y[i]@Sig@X[j] + Y[i]@Sig@Y[j] + delta >= 0)
    
    m.optimize()
    
    #This will be the list of products
    Q = [ [] for i in range(batch_size)]
    
    for i in range(batch_size):
        Q[i].append(X[i].X)
        Q[i].append(Y[i].X)
        
    return[Q,delta.X]


# In[7]:


#This function constructs a batch design based off of average question mean, average question variance, and average
#question orthogonality. For the average question orthogonality, we take the absolute value of the summands rather than
#the square. We also normalize mu and Sig in the objective so that we do not need to keep on refitting the parameters 
#that go with question mean, question variance, and question orthogonality.

def batch_design_AO(mu,Sig,batch_size,quest_mean_log_coeff,quest_var_log_coeff,quest_orth_log_coeff,t_lim = 100):
    #mu: expectation of prior on beta
    #Sig: Covariance matrix of prior on beta
    #batch_size: the number of questions we want to return in our batch design. This should be less or equal to the number
    #of attributes
    #quest_mean_log_coeff: this is a fitting parameter that goes with the average question mean and is obtained 
    #by fitting a linear model log (D-err/Init_det) ~ AM/||l*mu|| + AV/||s*Sig|| + AO/||s*Sig|| + ||l*mu|| + ||s*Sig|| and using the fitted parameter that goes with
    #AM/||l*mu||
    #quest_var_log_coeff: this is a fitting parameter that goes with the average question variance and is obtained 
    #by fitting a linear model log (D-err/Init_det) ~ AM/||l*mu|| + AV/||s*Sig|| + AO/||s*Sig|| + ||l*mu|| + ||s*Sig|| and using the fitted parameter that goes with
    #AV/||s*Sig||
    #quest_orth_log_coeff: this is a fitting parameter that goes with the average question orthogonality and is obtained 
    #by fitting a linear model log (D-err/Init_det) ~ AM/||l*mu|| + AV/||s*Sig|| + AO/||s*Sig|| + ||l*mu|| + ||s*Sig|| and using the fitted parameter that goes with
    #AO/||s*Sig||
    #(l,s) are scaling parameters for mu and Sig that divide the space into different signal-to-noise ratio regions.
    #t_lim: this is the max amount of time we want to take to construct the batch
    
    #This is the number of attributes for the products
    n = len(Sig[0])
    
    m = gp.Model("mip1")
    m.setParam('Timelimit',t_lim)
    
    #calculate 2-norms of mu and Sigma
    mu_2norm = np.linalg.norm(mu,2)
    Sig_2norm = np.linalg.norm(Sig,2)
    
    #List of tuples for delta variable
    delta_tuples = []
    for i in range(batch_size):
        for j in range(i+1,batch_size):
            delta_tuples.append((i,j))
    
    #Set up the x_i and y_i, i = 1,...,batchsize
    X = m.addMVar((batch_size,n),vtype = GRB.BINARY)
    Y = m.addMVar((batch_size,n),vtype = GRB.BINARY)
    Delta = m.addVars(delta_tuples, lb=0.0, vtype = GRB.CONTINUOUS)
    
    #Set up the objective function.
    m.setObjective((quest_mean_log_coeff/(batch_size*mu_2norm))*sum([mu@X[i] - mu@Y[i] for i in range(batch_size)]) + 
                   (quest_var_log_coeff/(batch_size*Sig_2norm))*sum([X[i]@Sig@X[i] - X[i]@(2.0*Sig)@Y[i] + 
                   Y[i]@Sig@Y[i] for i in range(batch_size)]) + 
                   (quest_orth_log_coeff/(batch_size*(batch_size-1)*Sig_2norm/2))*sum([Delta[i,j] for i in range(batch_size) for j in range(i+1,batch_size)]),GRB.MINIMIZE)
    
    #Set up the constraints that force the products in question i to be different, as well as forcing the symmetry
    #exploitation condition.
    for i in range(batch_size):
        m.addConstr(X[i]@X[i] - X[i]@Y[i] - Y[i]@X[i] + Y[i]@Y[i] >= 1)
        m.addConstr(mu@X[i] - mu@Y[i] >= 0)
        
    #Set up the Sigma-orthogonality constraint for all questions i and j, i not equal to j. Also add constraints
    #to make sure that questions within a batch are different, including with respect to switching order of products in
    #the questions.
    for i in range(batch_size):
        for j in range(i+1,batch_size):
            m.addConstr(X[i]@Sig@X[j] - X[i]@Sig@Y[j] - Y[i]@Sig@X[j] + Y[i]@Sig@Y[j] - Delta[i,j] <= 0)
            m.addConstr(X[i]@Sig@X[j] - X[i]@Sig@Y[j] - Y[i]@Sig@X[j] + Y[i]@Sig@Y[j] + Delta[i,j] >= 0)
            m.addConstr(X[i]@X[i] - X[i]@Y[i] - X[i]@X[j] + X[i]@Y[j] -
                       Y[i]@X[i] + Y[i]@Y[i] + Y[i]@X[j] - Y[i]@Y[j] -
                       X[j]@X[i] + X[j]@Y[i] + X[j]@X[j] - X[j]@Y[j] +
                       Y[j]@X[i] - Y[j]@Y[i] - Y[j]@X[j] + Y[j]@Y[j] >= 1)
            m.addConstr(X[i]@X[i] - X[i]@Y[i] - X[i]@Y[j] + X[i]@X[j] -
                       Y[i]@X[i] + Y[i]@Y[i] + Y[i]@Y[j] - Y[i]@X[j] -
                       Y[j]@X[i] + Y[j]@Y[i] + Y[j]@Y[j] - Y[j]@X[j] +
                       X[j]@X[i] - X[j]@Y[i] - X[j]@Y[j] + X[j]@X[j] >= 1)
            
    m.optimize()
    
    #This will be the list of products
    Q = [ [] for i in range(batch_size)]
    D = [ [] for i in range(batch_size-1)]
    
    for i in range(batch_size):
        Q[i].append(X[i].X)
        Q[i].append(Y[i].X)
        
    for i in range(batch_size):
        for j in range(i+1, batch_size):
            D[i].append(Delta[i,j].X)
        
    return[Q,D]


# In[8]:


#THIS WILL NEED TO BE MOVED!!!
#Example for batch_design_AO
#mu_ao = np.array(6*[1.0])
#Sig_ao = 1.0*np.identity(6)
#batch_size_ao = 4
#alpha_ao = 0.2813
#kappa_ao = -0.1019
#gamma_ao = 0.0577
#print(batch_design_AO(mu_ao,Sig_ao,batch_size_ao,alpha_ao,kappa_ao,gamma_ao))


# In[9]:


#This function constructs a batch design based off of average question mean, average question variance, and MAXIMUM
#question orthogonality. For the average question orthogonality, we take the absolute value of the summands rather than
#the square. We also normalize mu and Sig in the objective so that we do not need to keep on refitting the parameters 
#that go with question mean, question variance, and question orthogonality.

def batch_design_MO(mu,Sig,batch_size,quest_mean_log_coeff,quest_var_log_coeff,quest_orth_log_coeff,t_lim = 100):
    #mu: expectation of prior on beta
    #Sig: Covariance matrix of prior on beta
    #batch_size: the number of questions we want to return in our batch design. This should be less or equal to the number
    #of attributes
    #quest_mean_log_coeff: this is a fitting parameter that goes with the average question mean and is obtained 
    #by fitting a linear model log (D-err/Init_det) ~ AM/||l*mu|| + AV/||s*Sig|| + MO/||s*Sig|| + ||l*mu|| + ||s*Sig|| and using the fitted parameter that goes with
    #AM/||l*mu||
    #quest_var_log_coeff: this is a fitting parameter that goes with the average question variance and is obtained 
    #by fitting a linear model log (D-err/Init_det) ~ AM/||l*mu|| + AV/||s*Sig|| + MO/||s*Sig|| + ||l*mu|| + ||s*Sig|| and using the fitted parameter that goes with
    #AV/||s*Sig||
    #quest_orth_log_coeff: this is a fitting parameter that goes with the average question orthogonality and is obtained 
    #by fitting a linear model log (D-err/Init_det) ~ AM/||l*mu|| + AV/||s*Sig|| + MO/||s*Sig|| + ||l*mu|| + ||s*Sig|| and using the fitted parameter that goes with
    #MO/||s*Sig||
    #(l,s) are scaling parameters for mu and Sig that divide the space into different signal-to-noise ratio regions.
    #t_lim: this is the max amount of time we want to take to construct the batch
    
    #This is the number of attributes for the products
    n = len(Sig[0])
    
    m = gp.Model("mip1")
    m.setParam('Timelimit',t_lim)
    
    #calculate 2-norms of mu and Sigma
    mu_2norm = np.linalg.norm(mu,2)
    Sig_2norm = np.linalg.norm(Sig,2)
    
    #Set up the x_i and y_i, i = 1,...,batchsize
    X = m.addMVar((batch_size,n),vtype = GRB.BINARY)
    Y = m.addMVar((batch_size,n),vtype = GRB.BINARY)
    delta = m.addVar(lb=0.0, vtype = GRB.CONTINUOUS)
    
    #set up the objective function
    m.setObjective((quest_mean_log_coeff/(batch_size*mu_2norm))*sum([mu@X[i] - mu@Y[i] for i in range(batch_size)]) +
                  (quest_var_log_coeff/(batch_size*Sig_2norm))*sum([X[i]@Sig@X[i] - X[i]@(2.0*Sig)@Y[i] + 
                   Y[i]@Sig@Y[i] for i in range(batch_size)]) + (quest_orth_log_coeff/Sig_2norm)*delta,GRB.MINIMIZE)
    
    #Set up the constraints that force the products in question i to be different, as well as forcing the symmetry
    #exploitation condition.
    for i in range(batch_size):
        m.addConstr(X[i]@X[i] - X[i]@Y[i] - Y[i]@X[i] + Y[i]@Y[i] >= 1)
        m.addConstr(mu@X[i] - mu@Y[i] >= 0)
    
    #Set up the Sigma-orthogonality constraint for all questions i and j, i not equal to j. Also add constraints
    #to make sure that questions within a batch are different, including with respect to switching order of products in
    #the questions.
    for i in range(batch_size):
        for j in range(i+1,batch_size):
            m.addConstr(X[i]@Sig@X[j] - X[i]@Sig@Y[j] - Y[i]@Sig@X[j] + Y[i]@Sig@Y[j] - delta <= 0)
            m.addConstr(X[i]@Sig@X[j] - X[i]@Sig@Y[j] - Y[i]@Sig@X[j] + Y[i]@Sig@Y[j] + delta >= 0)
            m.addConstr(X[i]@X[i] - X[i]@Y[i] - X[i]@X[j] + X[i]@Y[j] -
                       Y[i]@X[i] + Y[i]@Y[i] + Y[i]@X[j] - Y[i]@Y[j] -
                       X[j]@X[i] + X[j]@Y[i] + X[j]@X[j] - X[j]@Y[j] +
                       Y[j]@X[i] - Y[j]@Y[i] - Y[j]@X[j] + Y[j]@Y[j] >= 1)
            m.addConstr(X[i]@X[i] - X[i]@Y[i] - X[i]@Y[j] + X[i]@X[j] -
                       Y[i]@X[i] + Y[i]@Y[i] + Y[i]@Y[j] - Y[i]@X[j] -
                       Y[j]@X[i] + Y[j]@Y[i] + Y[j]@Y[j] - Y[j]@X[j] +
                       X[j]@X[i] - X[j]@Y[i] - X[j]@Y[j] + X[j]@X[j] >= 1)
    
    m.optimize()
    
    #This will be the list of products
    Q = [ [] for i in range(batch_size)]
    
    for i in range(batch_size):
        Q[i].append(X[i].X)
        Q[i].append(Y[i].X)
        
    return[Q,delta.X]


# In[10]:


#THIS WILL NEED TO BE MOVED!!!
#Example for batch_design_MO
#mu_mo = np.array(6*[1.0])
#Sig_mo = 1.0*np.identity(6)
#batch_size_mo = 4
#alpha_mo = 0.2798
#kappa_mo = -0.1000
#gamma_mo = 0.0251
#print(batch_design_MO(mu_mo,Sig_mo,batch_size_mo,alpha_mo,kappa_mo,gamma_mo))


# In[11]:


#This function constructs a batch design of size k <= (number of attributes) where we enforce mutual 
#Sigma-orthogonality between the k questions. The orthogonality condition makes it so that the D-error minimization
#can be written as a product of g (one-step lookahead) functions. In this function, we are given a initial delta which is 
#assumed to be feasible and refine this delta to make it smaller using the function orthogonal_constraint_feas

def batch_design_delta_refine(mu,Sig,batch_size,mu_log_coeff,Sig_log_coeff,start_delta,t_lim=100):
    #mu: expectation of prior on beta
    #Sig: Covariance matrix of prior on beta
    #batch_size: the number of questions we want to return in our batch design. This should be less or equal to the number
    #of attributes
    #mu_log_coeff: the estimated coefficient c_1 that goes with m in the linear model log(g) = c_1*m + c_2*v
    #Sig_log_coeff: the estimated coefficient c_2 that goes with v in the linear model log(g) = c_1m + c_2*v
    #start_delta: this is the delta value that we start the orthogonality constraint with. Before the optimization procedure,
    #we attempt to find a smaller value of delta so that the orthogonality constraint is tighter.
    #M: this is a parameter which will be used as a constant to penalize the orthogonality constraint term delta.
    #t_lim: a time limit on the running time of the optimization procedure. Not sure if t=100 is sufficient at the moment.
    
    #This is the number of attributes for the products
    n = len(Sig[0])
    
    #These are terms corresponding to the linear and quadratic terms in the objective function.
    mu_s = mu_log_coeff*mu
    Sig_s = Sig_log_coeff*Sig
    
    m = gp.Model("mip1")
    m.setParam('Timelimit',t_lim)
    #m.params.NonConvex = 2
    
    #Set up the x_i and y_i, i = 1,...,batchsize
    X = m.addMVar((batch_size,n),vtype = GRB.BINARY)
    Y = m.addMVar((batch_size,n),vtype = GRB.BINARY)
    
    #Find a small value of delta so that the problem is feasible. ASSUME we start with a feasible delta (so we pick delta
    #to be large enough)
    delta = start_delta
    feasible = True
    while feasible:
        delta = delta/2.0
        feasible = orthogonal_constraint_feas(mu,Sig,delta,batch_size,t_lim=15)
        if not feasible:
            delta = 2.0*delta
        print(delta)
        

    #Set up the objective function, which is the sum of (batch_size) linearized g functions.
    m.setObjective(sum([mu_s@X[i] - mu_s@Y[i] + X[i]@Sig_s@X[i] -X[i]@(2.0*Sig_s)@Y[i] + 
                   Y[i]@Sig_s@Y[i]  for i in range(batch_size)]),GRB.MINIMIZE)
    
    #Set up the constraints that force the products in question i to be different, as well as forcing the symmetry
    #exploitation condition.
    for i in range(batch_size):
        m.addConstr(X[i]@X[i] - X[i]@Y[i] - Y[i]@X[i] + Y[i]@Y[i] >= 1)
        m.addConstr(mu@X[i] - mu@Y[i] >= 0)
    
    #Set up the Sigma-orthogonality constraint for all questions i and j, i not equal to j.
    for i in range(batch_size):
        for j in range(i+1,batch_size):
            m.addConstr(X[i]@Sig@X[j] - X[i]@Sig@Y[j] - Y[i]@Sig@X[j] + Y[i]@Sig@Y[j] - delta <= 0)
            m.addConstr(X[i]@Sig@X[j] - X[i]@Sig@Y[j] - Y[i]@Sig@X[j] + Y[i]@Sig@Y[j] + delta >= 0)
    
    m.optimize()
    
    #This will be the list of products
    Q = [ [] for i in range(batch_size)]
    
    for i in range(batch_size):
        Q[i].append(X[i].X)
        Q[i].append(Y[i].X)
        
    return[Q,delta]


# In[12]:


#THIS WILL NEED TO BE MOVED!!!
#An example of the batch design delta penalty algorithm.
#rng = np.random.default_rng(10000) 
#np.random.seed(10000)

#num_penalty = 6
#mu_test_penalty = rng.uniform(low = -1.0, high = 1.0, size = num_penalty)#np.array(num_penalty*[1.0])#rng.uniform(low = -1.0, high = 1.0, size = num_penalty)
#Sig_test_penalty = sklearn.datasets.make_spd_matrix(num_penalty)#np.identity(num_penalty)#sklearn.datasets.make_spd_matrix(num_penalty)
#batch_size_penalty = 4
#print('mu: ' + str(mu_test_penalty))
#print('Sig: ' + str(Sig_test_penalty))
#print(batch_design_delta_penalty(mu_test_penalty,Sig_test_penalty,batch_size_penalty,mu_log_coeff_6,Sig_log_coeff_6,M = 0.01))


# In[13]:


#WE MIGHT DELETE THIS FULLY!!!
#Create a graph showcasing changing of orthogonality delta with parameter M
#M_values = np.arange(0.01,1.01,0.01)
#M_val_len = len(M_values)
#delta_values = np.array(M_val_len*[0.0])
#for i in range(M_val_len):
    #delta_values[i] = batch_design_delta_penalty(mu_test_penalty,Sig_test_penalty,batch_size_penalty,
                                                 #mu_log_coeff_6,Sig_log_coeff_6,M = M_values[i])[1]

#print(delta_values)
#plt.title("Batch Orthogonality Variation with Changing Orthogonality Parameter") 
#plt.xlabel("Orthogonality Parameter") 
#plt.ylabel("Batch Orthogonality Measure") 
#plt.plot(M_values,delta_values) 
#plt.show()


# In[14]:


#WE MIGHT DELETE THIS FULLY!!!
#M_values_01 = np.arange(0.02,1.01,0.01)
#delta_values_01 = delta_values[1:]
#plt.title("Batch Orthogonality Variation with Changing Orthogonality Parameter") 
#plt.xlabel("Orthogonality Parameter") 
#plt.ylabel("Batch Orthogonality Measure") 
#plt.plot(M_values_01,delta_values_01) 
#plt.show()


# In[15]:


#WE WILL NEED TO MOVE THIS!!!
#An example of the batch design delta refine algorithm.
#rng = np.random.default_rng(10000) 
#np.random.seed(10000)

#num_refine = 6
#mu_test_refine = rng.uniform(low = -1.0, high = 1.0, size = num_refine)#np.array(num*[1.0])
#Sig_test_refine = sklearn.datasets.make_spd_matrix(num_refine)#np.identity(num)
#batch_size_refine = 4
#delta_start_refine = 2.0
#print(batch_design_delta_refine(mu_test_refine,Sig_test_refine,batch_size_refine,
                                #mu_log_coeff_6,Sig_log_coeff_6,delta_start_refine))


# In[16]:


#WE WILL NEED TO MOVE THIS
#An example of the feasibility checking algorithm
#rng = np.random.default_rng(10000) 
#np.random.seed(10000)

#feas_num = 6
#Sig_feas_test = sklearn.datasets.make_spd_matrix(feas_num)
#delta_feas = 0.0025
#batch_feas = 4
#print(orthogonal_constraint_feas(Sig_feas_test,delta_feas,batch_feas))


# In[17]:


#This function is used for calculating the probability of a user picking x over y.
def question_selection_prob(mu,Sig,x,y):
    #mu is mean parameter of the prior
    #Sig is the covariance matrix of the prior
    #x is the "preferred" product ( i.e, we will calculate P(x>y) )
    #y is the secondary product
    integ_bound = 30.0
    
    mu,Sig = np.array(mu),np.array(Sig)
    x,y = np.array(x),np.array(y)
    
    #Set up function to calculate the probability of choosing x over y
    m = np.dot(mu,x-y)
    v = np.dot(x-y,np.dot(Sig,x-y))
    fun1 = lambda z: ((1 + math.e**(-m - math.sqrt(v)*z))**(-1))*(math.e**((-z**2)/2))/math.sqrt(2*math.pi)
    
    probability = scipy.integrate.quad(fun1, -integ_bound, integ_bound)[0]
    
    return probability


# In[18]:


#THIS WILL NEED TO BE MOVED!!!
#This is to look at the question_selection_prob function
#rng = np.random.default_rng(100) 
#np.random.seed(100)
#num_prob = 6
#mu_test_prob = rng.uniform(low = -1.0, high = 1.0, size = num_prob)#np.array(num*[1.0])
#Sig_test_prob = sklearn.datasets.make_spd_matrix(num_prob)# np.identity(num_prob)
#print(mu_test_prob)
#print(Sig_test_prob)
#x_test_prob = num_prob*[1]
#y_test_prob = num_prob*[0]
#print('prob x over y:' + str(question_selection_prob(mu_test_prob,Sig_test_prob,x_test_prob,y_test_prob)))


# In[19]:


#This code is to perform full enumerative rollout 

def rollout(mu,Sig,x,y,rollout_length,mu_log_coeff,Sig_log_coeff):
    #mu: This is the initial mean parameter that we start with
    #Sig: This is the initial covariance matrix we start with
    #x,y: These are the two products (question) that we start the rollout process with
    #rollout_length: This is how far ahead we wish to "rollout" the initial question (x,y) under the one
    #step lookahead base policy
    #mu_log_coeff: the estimated coefficient c_1 that goes with m in the linear model log(g) = c_1*m + c_2*v
    #Sig_log_coeff: the estimated coefficient c_2 that goes with v in the linear model log(g) = c_1m + c_2*v
    #trimming_parameter: This is the probability of success parameter we use in the bernoulli random variable that is used
    #to determine whether we perform trimming at the kth level in the trajectory tree.
    
    #Calculate the probability of user picking x over y and y over x
    prob_x = question_selection_prob(mu,Sig,x,y)
    prob_y = 1.0 - prob_x
    
    #Update mu and Sig depending on if the user picks x or y.
    mu_x,Sig_x = moment_matching_update(x,y,mu,Sig)
    mu_y,Sig_y = moment_matching_update(y,x,mu,Sig)
    
    #Save updated parameters and probabilities
    N_x = [mu_x,Sig_x,prob_x]
    N_y = [mu_y,Sig_y,prob_y]
    
    N = [N_x,N_y]
    
    for i in range(rollout_length):
        #Instantiate a list that will hold nodes at the (i+1)th level after N = [N_x,N_y]
        Node_list = []
        
        #start_time_roll_node = time.perf_counter()
        for node in N:
            
            #Extract mean, covariance matrix, and (accumulated) probability.
            mu_n = node[0]
            Sig_n = node[1]
            prob_n = node[2]
            
            #Based off of the mean and covariance matrix, find the optimal one-step query, (x_n,y_n)
            x_n,y_n = g_opt(mu_n,Sig_n,mu_log_coeff,Sig_log_coeff)[1:] 
            
            #Calculate the probability of the user choosing xn or yn
            prob_xn = question_selection_prob(mu_n,Sig_n,x_n,y_n)
            prob_yn = 1.0 - prob_xn
            
            
            #Calculate the accumulated probability up to this node
            accumulate_prob_xn = prob_xn * prob_n
            accumulate_prob_yn = prob_yn * prob_n

            #Perform a moment matching update to get new parameters mu and Sig for both cases when the user
            #prefers x_n and y_n.
            mu_nx,Sig_nx = moment_matching_update(x_n,y_n,mu_n,Sig_n)
            mu_ny,Sig_ny = moment_matching_update(y_n,x_n,mu_n,Sig_n)

            #Store the parameters and accumulated probability in nodes.
            N_xn = [mu_nx,Sig_nx,accumulate_prob_xn]
            N_yn = [mu_ny,Sig_ny,accumulate_prob_yn]

            Node_list.append(N_xn)
            Node_list.append(N_yn)

                
        N = Node_list
        #print("Rollout Layer " + str(i) + ": ",time.perf_counter() - start_time_roll_node, "seconds")
    #print('Node List:' + str(N))
    weighted_det_sum = sum(np.sqrt(np.linalg.det(node[1]))*node[2] for node in N)
    
    return weighted_det_sum


# In[20]:


#This code is used to perform rollout using monte-carlo method

def monte_carlo_rollout(mu,Sig,x,y,rollout_length,mu_log_coeff,Sig_log_coeff,sample_budget):
    #mu: This is the initial mean parameter that we start with
    #Sig: This is the initial covariance matrix we start with
    #x,y: These are the two products (question) that we start the rollout process with
    #rollout_length: This is how far ahead we wish to "rollout" the initial question (x,y) under the one
    #step lookahead base policy
    #mu_log_coeff: The estimated coefficient c_1 that goes with m in the linear model log(g) = c_1*m + c_2*v
    #Sig_log_coeff: The estimated coefficient c_2 that goes with v in the linear model log(g) = c_1m + c_2*v
    #sample_budget: This is the number of trajectories that we want to use.
    
    #Calculate the probability of user picking x over y and y over x
    Sig_list = []
    
    prob_x = question_selection_prob(mu,Sig,x,y)
    prob_y = 1.0 - prob_x
    
    #Look at which product is preferred
    if prob_x >= prob_y:
        prefer_prob = prob_x 
        prefer_product = x
        not_prefer_product = y
    else:
        prefer_prob = prob_y 
        prefer_product = y
        not_prefer_product = x
    
    #Sample 'sample_budget' number of trajectories.
    for i in range(sample_budget):
        
        #Create a bernoulli random variable with probability parameter equal to the probability of the product having
        #the higher probability of selection between x and y.
        path_selector = scipy.stats.bernoulli.rvs(prefer_prob)
        
        #Perform moment matching according to the value of the bernoulli random variable, path_selector. If path_selector
        #is 1.0, we will go down the branch with higher probability. If path_selector is 0.0, we will go down the branch 
        #with lower probability.
        if path_selector == 1.0:
            mu_n,Sig_n = moment_matching_update(prefer_product,not_prefer_product,mu,Sig)
        else:
            mu_n,Sig_n = moment_matching_update(not_prefer_product,prefer_product,mu,Sig)
        
        #We start the rollout process on initial question (x,y)
        for j in range(rollout_length):
            
            #Solve the one-step lookahead problem to get the next question pair
            x_n,y_n = g_opt(mu_n,Sig_n,mu_log_coeff,Sig_log_coeff)[1:] 
            
            #Calculate the probability of x_n and y_n given mu_n and Sig_n
            prob_xn = question_selection_prob(mu_n,Sig_n,x_n,y_n)
            prob_yn = 1.0 - prob_xn
            
            #Check which of x_n and y_n has a higher probability. 
            if prob_xn >= prob_yn:
                prefer_prob_n = prob_xn 
                prefer_product_n = x_n
                not_prefer_product_n = y_n
            else:
                prefer_prob_n = prob_yn 
                prefer_product_n = y_n
                not_prefer_product_n = x_n
            
            #Create a bernoulli random variable with parameter prefer_prob_n.
            path_selector_n = scipy.stats.bernoulli.rvs(prefer_prob_n)
            
            #Perform moment matching according to the value of the bernoulli random variable, path_selector_n. 
            #If path_selector_n
            #is 1.0, we will go down the branch with higher probability. If path_selector is 0.0, we will go down the branch 
            #with lower probability.
            if path_selector_n == 1.0:
                mu_n,Sig_n = moment_matching_update(prefer_product_n,not_prefer_product_n,mu_n,Sig_n)
            else:
                mu_n,Sig_n = moment_matching_update(not_prefer_product_n,prefer_product_n,mu_n,Sig_n)
                
        
        #After we finish one trajectory, we append the resulting covariance matrix to a list.
        Sig_list.append(Sig_n)
        
    
    #Calculate an estimate for the determinant. We use the sample average of the determinants of the covariances coming from
    #different trajectories.
    determinant_estimate = (1/sample_budget)*sum(np.sqrt(np.linalg.det(S)) for S in Sig_list)
    
    return determinant_estimate


# In[21]:


#THIS WILL NEED TO BE MOVED!!!
#Example of rollout method:
#rng = np.random.default_rng(100) 
#np.random.seed(100)
#num_attr = 6
#roll_len = 8
#mu_roll = num_attr*[1.0]#rng.uniform(low = -1.0, high = 1.0, size = num_attr)#
#Sig_roll = np.identity(num_attr)#sklearn.datasets.make_spd_matrix(num_attr)#
#x_roll = [1.0,1.0,1.0,0.0,0.0,0.0]#num_attr*[1.0]
#y_roll = [0.0,0.0,0.0,1.0,1.0,1.0]#num_attr*[0.0]

#start_time_roll_t1 = time.perf_counter()
#print(rollout(mu_roll,Sig_roll,x_roll,y_roll,roll_len,mu_log_coeff_6,Sig_log_coeff_6))
#print("Overall Rollout: ",time.perf_counter() - start_time_roll_t1, "seconds")


# In[22]:


#THIS WILL NEED TO BE MOVED!!!
#Example of monte_carlo_rollout method:
#rng = np.random.default_rng(100) 
#np.random.seed(100)
#num_attr = 12
#roll_len = 11
#budget = 50
#mu_mcroll = num_attr*[1.0]#rng.uniform(low = -1.0, high = 1.0, size = num_attr)#num_attr*[0.5]
#Sig_mcroll = np.identity(num_attr)#sklearn.datasets.make_spd_matrix(num_attr)#np.identity(num_attr)
#x_mcroll = [1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0]#num_attr*[1.0]
#y_mcroll = [0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0]#num_attr*[0.0]

#start_time_mcroll_t1 = time.perf_counter()
#print(monte_carlo_rollout(mu_mcroll,Sig_mcroll,x_mcroll,y_mcroll,roll_len,mu_log_coeff_12,Sig_log_coeff_12,budget))
#print("Overall MC-Rollout: ",time.perf_counter() - start_time_mcroll_t1, "seconds")


# In[23]:


#This function is used for constructing a batch design and performing rollout on this batch design, returning the question
#amongst the batch that results in the lowest average determinant value

def rollout_with_batch_design_acquisition(mu,Sig,mu_log_coeff,Sig_log_coeff,batch_size,rollout_length,MC_budget,include_one_step = False,
                                          penalty_term = 100):
    #mu: This is the initial mean parameter that we start with
    #Sig: This is the initial covariance matrix we start with
    #mu_log_coeff: The estimated coefficient c_1 that goes with m in the linear model log(g) = c_1*m + c_2*v
    #Sig_log_coeff: The estimated coefficient c_2 that goes with v in the linear model log(g) = c_1m + c_2*v
    #batch_size: the number of questions we want to return in our batch design. This should be less or equal to the number
    #of attributes
    #rollout_length: This is how far ahead we wish to "rollout" the initial question (x,y) under the one
    #step lookahead base policy
    #MC_budget: This is the budget we allow for monte carlo method of rollout. At this point in time, we will use MC if
    #rollout_length is greater than 8.
    #include_one_step: This determines whether we want to include the one-step optimal question within our batch. This can
    #help ensure that rollout performs at least as well as one-step look ahead. Default value is False.
    #penalty_term: This is used to set the penalty level for orthogonality in the orthogonal batch design optimization problem. A higher penalty
    #term will lead to a more Sigma_orthogonal design, while a lower penalty term will lead to less Sigma_orthogonality in the design
    
    #Construct the batch based off of mu,Sig, and batch_size
    batch = batch_design_delta_penalty(mu,Sig,batch_size,mu_log_coeff,Sig_log_coeff,M = penalty_term)[0]
    
    #If desired, include the one-step look ahead optimal question within this batch to help ensure performance
    #is at least as good as one-step look ahead
    if include_one_step:
        [one_step_x,one_step_y] = g_opt(mu,Sig,mu_log_coeff,Sig_log_coeff)[1:]
        batch.append([one_step_x,one_step_y])
    print(batch)
    #For each question in the batch, perform rollout of length rollout_length and save the average of the determinant
    #values for each question. If rollout_length is greater than or equal to 8, use MC method instead, 
    #as it seems enumeration becomes slow after this point.
    cov_avg_det_values = []
    if rollout_length >= 8:
        for question in batch:
            x = question[0]
            y = question[1]
            avg_det_value = monte_carlo_rollout(mu,Sig,x,y,rollout_length,mu_log_coeff,Sig_log_coeff,MC_budget)
            cov_avg_det_values.append(avg_det_value)
    else:
        for question in batch:
            x = question[0]
            y = question[1]
            avg_det_value = rollout(mu,Sig,x,y,rollout_length,mu_log_coeff,Sig_log_coeff)
            cov_avg_det_values.append(avg_det_value)
    
    
    #Pick the question with the lowest average determinant value. Call it opt_question
    min_index = np.argmin(np.array(cov_avg_det_values))
    opt_question = batch[min_index]
    
    return opt_question
    


# In[24]:


#THIS WILL NEED TO BE MOVED OR DELETED!!!
#An example of rollout_with_batch_design_acquisition, we also look at the one-step lookahead case:

#rng = np.random.default_rng(100) 
#np.random.seed(100)

#rollacq_num_attr = 6
#rollacq_roll_len = 2
#rollacq_MC_budget = 50
#rollacq_batch_size = 4
#mu_naught = np.array([0.0,0.0,-2.0,3.0,8.0,0.0])#rng.uniform(low = -1.0, high = 1.0, size = rollacq_num_attr)
#Sig_naught = np.identity(rollacq_num_attr)#sklearn.datasets.make_spd_matrix(rollacq_num_attr)#

#Sig_naught = np.diag([4.0,4.0,0.25,0.25,0.25,4.0])
#rho = 0.1
#for i in range(rollacq_num_attr):
    #if i==0:
        #Sig_naught[i,0] = 1.0
        #Sig_naught[i,1] = rho
    #if i == rollacq_num_attr-1:
        #Sig_naught[i,rollacq_num_attr-1] = 1.0
        #Sig_naught[i,rollacq_num_attr-2] = rho 
    #if i>0 and i<(rollacq_num_attr-1):
        #for j in range(i-1,i+2):
            #Sig_naught[i,j] = rho**abs(i-j)

#Sig_naught = np.zeros([rollacq_num_attr,rollacq_num_attr])
#for i in range(rollacq_num_attr):
    #for j in range(rollacq_num_attr):
        #Sig_naught[i,j] = rho**abs(i-j)

#print(mu_naught)
#print(Sig_naught)
#rollout_opt = rollout_with_batch_design_acquisition(mu_naught,Sig_naught,mu_log_coeff_6,Sig_log_coeff_6,rollacq_batch_size,
                                           #rollacq_roll_len,rollacq_MC_budget,include_one_step = True,penalty_term = 0.1)
#onestep_opt = g_opt(mu_naught,Sig_naught,mu_log_coeff_6,Sig_log_coeff_6)[1:]

#print('rollout_opt: ' + str(rollout_opt))
#print('onestep_opt: ' + str(onestep_opt))
#print('rollout_value of rollout solution: ' + str(rollout(mu_naught,
                                                         #Sig_naught,
                                                         #rollout_opt[0],
                                                         #rollout_opt[1],
                                                         #rollacq_roll_len,
                                                         #mu_log_coeff_6,
                                                         #Sig_log_coeff_6)))
#print('rollout_value of onestep solution: ' + str(rollout(mu_naught,
                                                         #Sig_naught,
                                                         #onestep_opt[0],
                                                         #onestep_opt[1],
                                                         #rollacq_roll_len,
                                                         #mu_log_coeff_6,
                                                         #Sig_log_coeff_6)))

#[mu_update_roll,Sig_update_roll] = moment_matching_update(rollout_opt[0],rollout_opt[1],mu_naught,Sig_naught)
#[mu_update_onestep,Sig_update_onestep] = moment_matching_update(onestep_opt[0],onestep_opt[1],mu_naught,Sig_naught)

#print('Determinant of Sig_update_roll: ' + str(np.linalg.det(Sig_update_roll)))
#print('Determinant of Sig_update_onestep: ' + str(np.linalg.det(Sig_update_onestep)))

#print('Sig_update_roll: ' + str(Sig_update_roll))
#print('Sig_update_onestep: ' + str(Sig_update_onestep))
#print('mu_update_roll: ' + str(mu_update_roll))
#print('mu_update_onestep: ' + str(mu_update_onestep))


# In[25]:


#THIS MAY JUST BE DELETED!!!
#Second Rollout iteration
#rng = np.random.default_rng(100) 
#np.random.seed(100)

#rollacq_num_attr = 6
#rollacq_roll_len_1 = 2
#rollacq_MC_budget = 50 #NOT NEEDED since we are only doing 4 lookahead
#rollacq_batch_size = 4
#mu_roll_1 = mu_update_roll#rng.uniform(low = -1.0, high = 1.0, size = rollacq_num_attr)
#Sig_roll_1 = Sig_update_roll#sklearn.datasets.make_spd_matrix(rollacq_num_attr)#

#rollout_opt_1 = rollout_with_batch_design_acquisition(mu_roll_1,Sig_roll_1,mu_log_coeff_6,Sig_log_coeff_6,rollacq_batch_size,
                                           #rollacq_roll_len_1,rollacq_MC_budget,include_one_step = True)

#print('rollout_opt_1: ' + str(rollout_opt_1))

#[mu_roll_2,Sig_roll_2] = moment_matching_update(rollout_opt_1[0],rollout_opt_1[1],mu_roll_1,Sig_roll_1)

#print('Determinant of Sig_roll_2: ' + str(np.linalg.det(Sig_roll_2)))

#print('Sig_roll_2: ' + str(Sig_roll_2))
#print('mu_roll_2: ' + str(mu_roll_2))


# In[26]:


#THIS MAY NEED TO BE DELETED!!!
#Second onestep iteration
#rng = np.random.default_rng(100) 
#np.random.seed(100)


#mu_onestep_1 = mu_update_onestep#rng.uniform(low = -1.0, high = 1.0, size = rollacq_num_attr)
#Sig_onestep_1 = Sig_update_onestep#sklearn.datasets.make_spd_matrix(rollacq_num_attr)#

#onestep_opt_1 = g_opt(mu_onestep_1,Sig_onestep_1,mu_log_coeff_6,Sig_log_coeff_6)[1:]


#print('onestep_opt_1: ' + str(onestep_opt_1))

#[mu_onestep_2,Sig_onestep_2] = moment_matching_update(onestep_opt_1[0],onestep_opt_1[1],mu_onestep_1,Sig_onestep_1)

#print('Determinant of Sig_onestep_2: ' + str(np.linalg.det(Sig_onestep_2)))

#print('Sig_onestep_2: ' + str(Sig_onestep_2))
#print('mu_onestep_2: ' + str(mu_onestep_2))


# In[27]:


def coordinate_exchange_acq(mu,Sig,mu_log_coeff,Sig_log_coeff,batch_size,rollout_length,MC_budget,rel_gap_threshold,
                           include_batch = False, include_one_step = True):
    #mu: This is the initial mean parameter that we start with
    #Sig: This is the initial covariance matrix we start with
    #mu_log_coeff: The estimated coefficient c_1 that goes with m in the linear model log(g) = c_1*m + c_2*v
    #Sig_log_coeff: The estimated coefficient c_2 that goes with v in the linear model log(g) = c_1m + c_2*v
    #batch_size: the number of questions we want to return in our batch design. This should be less or equal to the number
    #of attributes
    #rollout_length: This is how far ahead we wish to "rollout" the initial question (x,y) under the one
    #step lookahead base policy
    #MC_budget: This is the budget we allow for monte carlo method of rollout. At this point in time, we will use MC if
    #rollout_length is greater than 8.
    #rel_gap_threshold: This is used to see if the perturbed question outperforms the current question
    #include_batch: This determines whether we also iterate over the batch questions.
    #include_one_step: This determines whether we want to include the one-step optimal question within our batch. This can
    #help ensure that rollout performs at least as well as one-step look ahead.
    
    #question_component tells us how many components are in (x,y)
    attr = len(mu)
    question_component = 2*attr
    
    #Used to store the initial set of questions coming from batch and one-step method
    init_set_of_questions = []
    
    #Used to store the set of questions after performing coordinate exchange on the initial batch
    final_set_of_questions = []
    
    #This will store the rollout values of the final set of questions
    final_det_values = []
    
    #Determine if the initial set of questions includes the batch design
    if include_batch:
        init_set_of_questions = batch_design_delta_penalty(mu,Sig,batch_size,mu_log_coeff,Sig_log_coeff)[0]
    
    #Determine if the initial set of questions includes the one-step optimal solution
    if include_one_step:
        [one_step_x,one_step_y] = g_opt(mu,Sig,mu_log_coeff,Sig_log_coeff)[1:]
        init_set_of_questions.append([one_step_x,one_step_y])
    
    #Need to use monte_carlo if rollout_length is greater than 8
    if rollout_length>=8:
        #iterate over all the questions
        for question in init_set_of_questions:
            current_question = [q[:] for q in question]#question[:]
            x = current_question[0]
            y = current_question[1]
            #perform monte carlo rollout on the current question. Store its value
            current_roll_value = monte_carlo_rollout(mu,Sig,x,y,rollout_length,mu_log_coeff,Sig_log_coeff,MC_budget)
            #This counter is used to determine when the coordinate exchange process stops for this question.
            print('first roll value: ' + str(current_roll_value))
            print('first question: ' + str(current_question))
            counter = 0
            #Begin the coordinate exchange process for the current question
            while counter < question_component:
                #Initiate a variable called perturb question. This question will start as the same as the current question,
                #but later on we will change a single attribute in one of the products and see if that improves the rollout
                #value.
                perturb_question = [cq[:] for cq in current_question]
                #If the counter is less than the number of product attributes, we will change one of the entries in 'x'
                if counter < attr:
                    #Changing an entry in 'x'
                    perturb_question[0][counter] = abs(1.0-current_question[0][counter])
                    x_perturb = perturb_question[0]
                    y_perturb = perturb_question[1]
                    #Make sure the perturbed 'x' and 'y' are not equal. If so, check its rollout value.
                    if np.dot(np.array(x_perturb)-np.array(y_perturb),np.array(x_perturb)-np.array(y_perturb))>0:
                        perturb_roll_value = monte_carlo_rollout(mu,Sig,x_perturb,y_perturb,rollout_length,mu_log_coeff,
                                                                 Sig_log_coeff,MC_budget)
                #If counter is greater than number of product attributes, we will change one of the entries in 'y'.
                else:
                    perturb_question[1][counter-attr] = abs(1.0 - current_question[1][counter-attr])
                    x_perturb = perturb_question[0]
                    y_perturb = perturb_question[1]
                    if np.dot(np.array(x_perturb)-np.array(y_perturb),np.array(x_perturb)-np.array(y_perturb))>0:
                        perturb_roll_value = monte_carlo_rollout(mu,Sig,x_perturb,y_perturb,rollout_length,mu_log_coeff,
                                                                 Sig_log_coeff,MC_budget)
                
                #If the perturbed question's rollout value outperforms the current question's rollout value by some
                #relative threshold, then we will replace the current question with the perturbed question
                if (current_roll_value - perturb_roll_value)/current_roll_value >= rel_gap_threshold:
                    current_question = [q[:] for q in perturb_question]#perturb_question
                    current_roll_value = perturb_roll_value
                    counter = 0
                    print('current_question: ' + str(current_question))
                    print('current_roll_value: ' + str(current_roll_value))
                else:
                    counter = counter + 1
            
            #After the coordinate exchange process, we place the resulting question and its rollout value in a list.
            final_set_of_questions.append(current_question)
            final_det_values.append(current_roll_value)
    
    #Use regular rollout if rollout length is less than 8. Same coordinate exchange process as in the 
    #monte carlo rollout method    
    else:
        for question in init_set_of_questions:
            current_question = [q[:] for q in question]
            x = current_question[0]
            y = current_question[1]
            current_roll_value = rollout(mu,Sig,x,y,rollout_length,mu_log_coeff,Sig_log_coeff)
            print('first roll value: ' + str(current_roll_value))
            print('first question: ' + str(current_question))
            counter = 0
            while counter < question_component:
                perturb_question = [cq[:] for cq in current_question]
                if counter < attr:
                    perturb_question[0][counter] = abs(1.0-current_question[0][counter])
                    x_perturb = perturb_question[0]
                    y_perturb = perturb_question[1]
                    if np.dot(np.array(x_perturb)-np.array(y_perturb),np.array(x_perturb)-np.array(y_perturb))>0:
                        perturb_roll_value = rollout(mu,Sig,x_perturb,y_perturb,rollout_length,mu_log_coeff,
                                                                 Sig_log_coeff)
                else:
                    perturb_question[1][counter-attr] = abs(1.0 - current_question[1][counter-attr])
                    x_perturb = perturb_question[0]
                    y_perturb = perturb_question[1]
                    if np.dot(np.array(x_perturb)-np.array(y_perturb),np.array(x_perturb)-np.array(y_perturb))>0:
                        perturb_roll_value = rollout(mu,Sig,x_perturb,y_perturb,rollout_length,mu_log_coeff,
                                                                 Sig_log_coeff)
                    
                if (current_roll_value - perturb_roll_value)/current_roll_value >= rel_gap_threshold:
                    current_question = [q[:] for q in perturb_question]
                    current_roll_value = perturb_roll_value
                    counter = 0
                    print('current_question: ' + str(current_question))
                    print('current_roll_value: ' + str(current_roll_value))
                else:
                    counter = counter + 1
                
            final_set_of_questions.append(current_question)
            final_det_values.append(current_roll_value)
    
    #Find the minimimum rollout value among all the questions that have went through coordinate exchange. Pick the one with
    #the smallest rollout value.
    min_index = np.argmin(np.array(final_det_values))
    rollout_coordinate_opt_question = final_set_of_questions[min_index]
    
    return rollout_coordinate_opt_question


# In[28]:


#THIS WILL NEED TO BE MOVED!!!
#Example for coordinate_exchange
#rng = np.random.default_rng(100) 
#np.random.seed(100)

#coord_mu = np.array(12*[1.0])#rng.uniform(low = -1.0, high = 1.0, size = 12)#np.array(6*[1.0])#np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])#np.array([-0.25,0.5,-0.75,1.0,-1.25,1.5]) 
#coord_Sig = np.identity(12)#sklearn.datasets.make_spd_matrix(12)#np.identity(6)##np.identity(12)#1.25*np.array([[1.0,-0.5,0.25,-0.125,0.0625,-0.03125],
                      #[-0.5,1.0,-0.5,0.25,-0.125,0.0625],
                     #[0.25,-0.5,1.0,-0.5,0.25,-0.125],
                     #[-0.125,0.25,-0.5,1.0,-0.5,0.25],
                     #[0.0625,-0.125,0.25,-0.5,1.0,-0.5],
                     #[-0.03125,0.0625,-0.125,0.25,-0.5,1.0]])       #np.diag([2.0,0.5,1.0,1.0,1.0,1.0])
                    #np.diag([1.25,0.8,1.5,0.3333,2,0.5])
#print(coord_mu,coord_Sig)
#coord_batch_size = 0
#coord_roll_len = 3
#coord_MC_budget = 0#50
#coord_rel_gap = 0.0001
#start_time = time.perf_counter()
#print(coordinate_exchange_acq(coord_mu,coord_Sig,mu_log_coeff_12,Sig_log_coeff_12,coord_batch_size,coord_roll_len,
                             #coord_MC_budget,coord_rel_gap,include_batch = False))
#print(time.perf_counter() - start_time, "seconds")


# In[29]:


#THIS WILL NEED TO BE MOVED OR DELETED!!!
#Example for batch
#rng = np.random.default_rng(99) 
#np.random.seed(99)

#batch_mu = np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])#np.array([-0.25,0.5,-0.75,1.0,-1.25,1.5])#np.array([1.0,1.0,1.0,1.0,1.0,1.0])# #rng.uniform(low = -1.0, high = 1.0, size = 6)
#batch_Sig = np.identity(12)#1.25*np.array([[1.0,-0.5,0.25,-0.125,0.0625,-0.03125],
                        #[-0.5,1.0,-0.5,0.25,-0.125,0.0625],
                     #[0.25,-0.5,1.0,-0.5,0.25,-0.125],
                     #[-0.125,0.25,-0.5,1.0,-0.5,0.25],
                     #[0.0625,-0.125,0.25,-0.5,1.0,-0.5],
                     #[-0.03125,0.0625,-0.125,0.25,-0.5,1.0]])  
                    #np.identity(6)#  #sklearn.datasets.make_spd_matrix(6)#np.diag([2.0,0.5,1.0,1.0,1.0,1.0])
                    #np.diag([1.25,0.8,1.5,0.3333,2,0.5])
#print(batch_mu,batch_Sig)

#roll_batch_size = 4

#batch_roll_len = 3

#batch_roll_MC_budget = 0#50

#start_time_batch = time.perf_counter()
#print(rollout_with_batch_design_acquisition(batch_mu,batch_Sig,mu_log_coeff_12,Sig_log_coeff_12,roll_batch_size,
                                           #batch_roll_len,batch_roll_MC_budget,include_one_step = True,penalty_term = 10))
#print(time.perf_counter() - start_time_batch, "seconds")

