#!/usr/bin/env python
# coding: utf-8


#This module 'Questionnaire_Procedure' has all of the functions needed to do one-step and 
#two-step questionnaire, along with the two-step acquisition function since it depends on functions
#coming from this module. It includes the following functions:
#1. moment_matching_update
#2. g_opt
#3. two_stage_g_opt
#4. two_step_g_acq (WE INCLUDE THIS HERE BECAUSE IT DEPENDS ON moment_matching_update AND g_opt, THEMATICALLY THIS
#SHOULD NOT BE HERE)



import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.integrate
import math
import random # MARCH 3 2023

import time
import sklearn.datasets

import sys
sys.path.append(r'C:\Users\wsfishe\Desktop\PreferenceElicitationCode')
from Baseline_Functions_Definitions import z_expectation_variance, g_fun #mu_log_coeff, Sig_log_coeff
#Note that when we do 'from Baseline_Functions_Definitions import ...' we are importing:

#FUNCTIONS:
#1. z_expectation_variance
#2. g_fun



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


#Formulate an optimization problem to get the optimal solution to the one-step lookahead problem

def g_opt(mu, Sig, mu_log_coeff, Sig_log_coeff):
    #mu: expectation of prior on beta
    #Sig: covariance matrix of prior on beta
    #mu_log_coeff: the estimated coefficient c_1 that goes with m in the linear model log(g) = c_1*m + c_2*v
    #Sig_log_coeff: the estimated coefficient c_2 that goes with v in the linear model log(g) = c_1m + c_2*v
    
    #n is number of attributes
    n = len(Sig[0])
    
    #Log coefficients:
    mu_s = mu_log_coeff*mu 
    Sig_s = Sig_log_coeff*Sig 
    
    # Create a new model
    m = gp.Model("mip1")
    m.setParam('OutputFlag', 0)
    #m.setParam('MIPGap', 0)
    #m.params.DualReductions = 0
    
    #Set up x and y binary vectors, other variables
    x = m.addMVar(shape = n, vtype = GRB.BINARY, name = "x")
    y = m.addMVar(shape = n, vtype = GRB.BINARY, name = "y")
    
    #Objective function, constant values obtained from R lm function, regression on log(g)
    m.setObjective(mu_s@x - mu_s@y + x@Sig_s@x - y@Sig_s@x - x@Sig_s@y + y@Sig_s@y,
                   GRB.MINIMIZE)

    #Set up constraint so that x and y are different
    m.addConstr(x@x - x@y - y@x + y@y >= 1)
    
    #We want mu(x-y) >= 0 due to the symmetry of the g function
    m.addConstr(mu@x - mu@y >= 0)
    
    m.optimize()
    
    #Return solution x and y
    Vars = m.getVars()
    x_sol = []
    y_sol = []
    for u in range(0,n):
        x_sol.append(Vars[u].x)
        
    for w in range(n,2*n):
        y_sol.append(Vars[w].x)
    
    
    return [m.objVal,x_sol,y_sol]


#Formulate an optimization problem to get the optimal solution to the one-step lookahead problem when there are multiple
#levels for each attribute. The products are not binarized before the preference learning procedure.

#ADDED March 6, 2023

def g_opt_multi_lvl(mu, Sig, mu_log_coeff, Sig_log_coeff, num_lvl_vec):
    #mu: expectation of prior on beta
    #Sig: covariance matrix of prior on beta
    #mu_log_coeff: the estimated coefficient c_1 that goes with m in the linear model log(g) = c_1*m + c_2*v
    #Sig_log_coeff: the estimated coefficient c_2 that goes with v in the linear model log(g) = c_1*m + c_2*v
    #num_lvl_vec: this is a vector containing the number of levels for each attribute. For example, [4,3,5] denotes
    #three attributes with 4, 3, and 5 levels respectively.
    
    #n is number of attributes
    n = len(Sig[0])
    
    #Log coefficients:
    mu_s = mu_log_coeff*mu 
    Sig_s = Sig_log_coeff*Sig 
    
    # Create a new model
    m = gp.Model("mip1")
    m.setParam('OutputFlag', 0)
    m.params.NonConvex = 0
    
    #This is the total number of binary variables we will have.
    bin_var_len = sum([(math.floor(math.log2(L - 1)) + 1) for L in num_lvl_vec])
    
    #Write out the binary basis values for each attribute and save them in a list
    basis = []
    for i in range(n):
        basis_i = [2**j  for j in range(math.floor(math.log2(num_lvl_vec[i] - 1)) + 1)]
        basis.append(basis_i)
        
    
    #Set up the binary encoding xb_i and yb_i, i = 1,...,n
    xb = m.addMVar(shape = bin_var_len,vtype = GRB.BINARY)
    yb = m.addMVar(shape = bin_var_len,vtype = GRB.BINARY)
    
    
    #Objective function, constant values obtained from R lm function, regression on log(g)
    #MARCH 14, 2023
    m.setObjective(gp.quicksum([mu_s[i]*(gp.quicksum([basis[i][j]*xb[sum([(math.floor(math.log2(num_lvl_vec[k] - 1)) + 1) 
                                                                     for k in range(i)]) + j] for j in range(len(basis[i]))]) -
                                    gp.quicksum([basis[i][j]*yb[sum([(math.floor(math.log2(num_lvl_vec[k] - 1)) + 1) 
                                                                     for k in range(i)]) + j] for j in range(len(basis[i]))])) for i in range(n)]) +
                  gp.quicksum([gp.quicksum([(gp.quicksum([basis[s][j]*xb[sum([(math.floor(math.log2(num_lvl_vec[k] - 1)) + 1) for k in range(s)]) + j] 
                                                          for j in range(len(basis[s]))])*gp.quicksum([basis[r][j]*xb[sum([(math.floor(math.log2(num_lvl_vec[k] - 1)) + 1) for k in range(r)]) + j]
                                                                                                       for j in range(len(basis[r]))]) +
                                            gp.quicksum([basis[s][j]*yb[sum([(math.floor(math.log2(num_lvl_vec[k] - 1)) + 1) for k in range(s)]) + j] 
                                                         for j in range(len(basis[s]))])*gp.quicksum([basis[r][j]*yb[sum([(math.floor(math.log2(num_lvl_vec[k] - 1)) + 1) for k in range(r)]) + j] 
                                                                                                      for j in range(len(basis[r]))]) -
                                            gp.quicksum([basis[s][j]*xb[sum([(math.floor(math.log2(num_lvl_vec[k] - 1)) + 1) for k in range(s)]) + j] for j in range(len(basis[s]))])*
                                            gp.quicksum([basis[r][j]*yb[sum([(math.floor(math.log2(num_lvl_vec[k] - 1)) + 1) for k in range(r)]) + j] for j in range(len(basis[r]))]) -
                                            gp.quicksum([basis[r][j]*xb[sum([(math.floor(math.log2(num_lvl_vec[k] - 1)) + 1) for k in range(r)]) + j] for j in range(len(basis[r]))])*
                                            gp.quicksum([basis[s][j]*yb[sum([(math.floor(math.log2(num_lvl_vec[k] - 1)) + 1) for k in range(s)]) + j] for j in range(len(basis[s]))]))*Sig_s[s,r] 
                                            for s in range(n)]) for r in range(n)]), GRB.MINIMIZE)
    
    
    #Constraint (C3)', ensures the products are different. At least one of the x components is different from y component.
    m.addConstr(gp.quicksum([(xb[i]-yb[i])*(xb[i]-yb[i]) for i in range(bin_var_len)]) >= 1)
    
    
    #We want mu(x-y) >= 0 due to the symmetry of the g function
    #MARCH 14, 2023
    m.addConstr(gp.quicksum([mu[i]*(gp.quicksum([basis[i][j]*xb[sum([(math.floor(math.log2(num_lvl_vec[k] - 1)) + 1) 
                                                                     for k in range(i)]) + j] for j in range(len(basis[i]))]) -
                                    gp.quicksum([basis[i][j]*yb[sum([(math.floor(math.log2(num_lvl_vec[k] - 1)) + 1) 
                                                                     for k in range(i)]) + j] for j in range(len(basis[i]))])) for i in range(n)]) >= 0)
    
    
    #Adding constraints (NEWC1)' and (NEWC2)'
    #MARCH 14, 2023
    m.addConstr(gp.quicksum( [basis[0][j]*xb[j] for j in range(len(basis[0]))] ) <= num_lvl_vec[0]-1)
    m.addConstr(gp.quicksum( [basis[0][j]*yb[j] for j in range(len(basis[0]))] ) <= num_lvl_vec[0]-1)
    
    #(adding (NEWC1)' and (NEWC2)' continued)
    #MARCH 14, 2023
    for i in range(1,n):
        m.addConstr(gp.quicksum([basis[i][j]*xb[sum([(math.floor(math.log2(num_lvl_vec[k] - 1)) + 1) for k in range(i)]) + j] for j in range(len(basis[i]))]) <= num_lvl_vec[i]-1)
        m.addConstr(gp.quicksum([basis[i][j]*yb[sum([(math.floor(math.log2(num_lvl_vec[k] - 1)) + 1) for k in range(i)]) + j] for j in range(len(basis[i]))]) <= num_lvl_vec[i]-1)
    
    m.optimize()
    
    #Convert the binary variables xb and yb back into multi-level quantitative attributes for x and y
    x = np.array([sum(xb.X[sum([(math.floor(math.log2(num_lvl_vec[k] - 1)) + 1) for k in range(i)]) + j]*basis[i][j] for j in range(len(basis[i]))) for i in range(n)])
    y = np.array([sum(yb.X[sum([(math.floor(math.log2(num_lvl_vec[k] - 1)) + 1) for k in range(i)]) + j]*basis[i][j] for j in range(len(basis[i]))) for i in range(n)])
    
    
    return [m.objVal,x,y,xb.X,yb.X]


#Formulate an optimization problem to get the optimal solution to the one-step lookahead problem when there are multiple
#levels for each attribute. The products are binarized before the preference learning procedure.

#ADDED March 7, 2023

def g_opt_multi_lvl_v2(mu_b, Sig_b, mu_log_coeff, Sig_log_coeff, num_lvl_vec):
    
    #n is number of attributes
    n = len(Sig_b[0])
    
    #Log coefficients:
    mu_s = mu_log_coeff*mu_b
    Sig_s = Sig_log_coeff*Sig_b 
    
    # Create a new model
    m = gp.Model("mip1")
    m.setParam('OutputFlag', 0)
    
    #This is the total number of binary variables we will have.
    bin_var_len = sum([(L - 1) for L in num_lvl_vec])
    
    #Set up the binary encoding xb_i and yb_i.
    xb = m.addMVar(shape = bin_var_len,vtype = GRB.BINARY)
    yb = m.addMVar(shape = bin_var_len,vtype = GRB.BINARY)
    
    #Objective function, constant values obtained from R lm function, regression on log(g)
    m.setObjective(mu_s@xb - mu_s@yb + xb@Sig_s@xb - yb@Sig_s@xb - xb@Sig_s@yb + yb@Sig_s@yb,
                   GRB.MINIMIZE)
    
    #Set up constraint so that x and y are different
    m.addConstr(xb@xb - xb@yb - yb@xb + yb@yb >= 1)
    
    #We want mu(x-y) >= 0 due to the symmetry of the g function
    m.addConstr(mu_b@xb - mu_b@yb >= 0)
    
    #Add constraints (C3)' and (C5)'.
    m.addConstr(gp.quicksum([xb[i] for i in range(num_lvl_vec[0]-1)]) <= 1)
    m.addConstr(gp.quicksum([yb[i] for i in range(num_lvl_vec[0]-1)]) <= 1)
    
    for j in range(1,n-1):
        m.addConstr(gp.quicksum([xb[i] for i in range(sum([(num_lvl_vec[k] - 1) for k in range(j)]) + 1,sum([(num_lvl_vec[k] - 1) for k in range(j+1)]))]) <= 1)
        m.addConstr(gp.quicksum([yb[i] for i in range(sum([(num_lvl_vec[k] - 1) for k in range(j)]) + 1,sum([(num_lvl_vec[k] - 1) for k in range(j+1)]))]) <= 1)
    m.optimize()
    
    return [m.objVal,xb.X,yb.X]


#Define optimization problem for approximate two-step acquisition, exploiting orthogonality property
#|q_1 * Sig * q_0| < epsilon.

#include time_limit as a parameter. Looking at attributes equal to 12, it appears 100 seconds is sufficient for most cases.
def two_stage_g_opt(mu, Sig, mu_log_coeff, Sig_log_coeff, epsilon, t_lim = 100):
    #mu: expectation of prior on beta
    #Sig: Covariance matrix of prior on beta
    #mu_log_coeff: the estimated coefficient c_1 that goes with m in the linear model log(g) = c_1*m + c_2*v
    #Sig_log_coeff: the estimated coefficient c_2 that goes with v in the linear model log(g) = c_1m + c_2*v
    #epsilon: bound for orthogonality contraint, should be small.
    
    #number of attributes
    n = len(Sig[0])
    
    #Scale mu and Sig by the parameters we received from linear approximation
    mu_s = mu_log_coeff*mu
    Sig_s = Sig_log_coeff*Sig
    
    # Create a new model
    m = gp.Model("mip1")
    m.setParam('Timelimit', t_lim)
    #m.setParam('OutputFlag', 0)
    
    #Set up x_0, y_0, x_1, and y_1 binary vectors, other variables
    x_0 = m.addMVar(shape = n, vtype = GRB.BINARY, name = "x_0")
    y_0 = m.addMVar(shape = n, vtype = GRB.BINARY, name = "y_0")
    x_1 = m.addMVar(shape = n, vtype = GRB.BINARY, name = "x_1")
    y_1 = m.addMVar(shape = n, vtype = GRB.BINARY, name = "y_1")
    
    #Objective function, coefficient values obtained from R lm function, regression on g
    #for 0<=mu<=3 and sig>=2.5 
    
    m.setObjective(mu_s@x_0 - mu_s@y_0 + mu_s@x_1 - mu_s@y_1 + x_0@Sig_s@x_0 -x_0@(2.0*Sig_s)@y_0 + 
                   y_0@Sig_s@y_0 + x_1@Sig_s@x_1 - x_1@(2.0*Sig_s)@y_1 + y_1@Sig_s@y_1,
                   GRB.MINIMIZE)
    
    #Set up constraint so that x_0 and y_0 are different
    
    m.addConstr(x_0@x_0 - x_0@y_0 - y_0@x_0 + y_0@y_0 >= 1)
    
    #Set up constraint so that x_1 and y_1 are different
    
    m.addConstr(x_1@x_1 - x_1@y_1 - y_1@x_1 + y_1@y_1 >= 1)
    
    #Set up mu_v_0, where v_0 = x_0-y_0. We want mu_v_0 >= 0 due to the symmetry of the g function

    m.addConstr(mu@x_0 - mu@y_0 >= 0)
    
    #Set up mu_v_1, where v_1 = x_1-y_1. We want mu_v_1 >= 0 due to the symmetry of the g function

    m.addConstr(mu@x_1 - mu@y_1 >= 0)
    
    #Set up orthogonality constraint
    
    m.addConstr(x_1@Sig@x_0 - x_1@Sig@y_0 - y_1@Sig@x_0 + y_1@Sig@y_0 <= epsilon)
    m.addConstr(x_1@Sig@x_0 - x_1@Sig@y_0 - y_1@Sig@x_0 + y_1@Sig@y_0 >= -epsilon)
    
    m.optimize()
    
    #Return solutions x_0,y_0 and x_1,y_1
    Vars = m.getVars()
    x_0_sol = []
    y_0_sol = []
    x_1_sol = []
    y_1_sol = []
    for u in range(0,n):
        x_0_sol.append(Vars[u].x)
        
    for w in range(n,2*n):
        y_0_sol.append(Vars[w].x)
    
    for u in range(2*n,3*n):
        x_1_sol.append(Vars[u].x)
        
    for w in range(3*n,4*n):
        y_1_sol.append(Vars[w].x)
    
    
    return [x_0_sol,y_0_sol,x_1_sol,y_1_sol]


#A function to evaluate the two_step value of a question given prior parameters mu_0 and Sig_0
#x_0 and y_0 are a question pair.
#This is the exact two-step g value.
def two_step_g_acq(mu_0,Sig_0,mu_log_coeff,Sig_log_coeff,x_0,y_0):
    #mu_0 and Sig_0: These are the parameters (expectation and covariance) of the prior distribution
    #mu_log_coeff and Sig_log_coeff: These are parameters used in the linear model approximation log(g) = c_1*m + c_2*v
    #x_0 and y_0: These are a question pair that we are interested in evaluating
    
    #Ensure that the given arguments are numpy arrays for processing below.
    x_0_vec = np.array(x_0)
    y_0_vec = np.array(y_0)
    mu_0_vec = np.array(mu_0)
    Sig_0_vec = np.array(Sig_0)
    
    #Define first stage variables
    m_0 = np.dot(mu_0_vec,x_0_vec - y_0_vec)
    v_0 = np.dot(x_0_vec-y_0_vec,np.dot(Sig_0_vec,x_0_vec-y_0_vec))
    
    #Gather the posterior information given the two scenarios where the individual picks x over y or they pick
    #y over x
    [mu_10,Sig_10] = moment_matching_update(x_0_vec,y_0_vec,mu_0_vec,Sig_0_vec)
    [mu_11,Sig_11] = moment_matching_update(y_0_vec,x_0_vec,mu_0_vec,Sig_0_vec)
    
    #Solve g_opt(mu_10,Sig_10)[0] and g_opt(mu_11,Sig_11)[0] in order to get optimal questions for each scenario, where
    #each scenario is the individual picking x or y.
    
    [x_10,y_10] = g_opt(mu_10,Sig_10,mu_log_coeff,Sig_log_coeff)[1:]
    [x_11,y_11] = g_opt(mu_11,Sig_11,mu_log_coeff,Sig_log_coeff)[1:]
    
    #Define second stage variables.
    m_10 = np.dot(np.array(mu_10),np.array(x_10) - np.array(y_10))
    v_10 = np.dot(np.array(x_10) - np.array(y_10),np.dot(np.array(Sig_10),np.array(x_10)-np.array(y_10)))
    m_11 = np.dot(np.array(mu_11),np.array(x_11) - np.array(y_11))
    v_11 = np.dot(np.array(x_11) - np.array(y_11),np.dot(np.array(Sig_11),np.array(x_11)-np.array(y_11)))
    
    #Calculate the two-step value. fst_stg_g_sum_term are the two summation terms of g(m_0,v_0)
    fst_stg_g_sum_term = g_fun(m_0,v_0)[1:]
    two_step_g = g_fun(m_10,v_10)[0]*fst_stg_g_sum_term[0] + g_fun(m_11,v_11)[0]*fst_stg_g_sum_term[1]
    
    return [two_step_g,x_10,y_10,x_11,y_11]

#A function to evaluate the two_step value of a question given prior parameters mu_0 and Sig_0
#x_0 and y_0 are a question pair, where x_0 and y_0 have attributes with multiple quantitative levels.
#This is the exact two-step g value.

def multlvl_two_step_g_acq(mu_0,Sig_0,mu_log_coeff,Sig_log_coeff,x_0,y_0,num_lvl_vec):
    #mu_0 and Sig_0: These are the parameters (expectation and covariance) of the prior distribution
    #mu_log_coeff and Sig_log_coeff: These are parameters used in the linear model approximation log(g) = c_1*m + c_2*v
    #x_0 and y_0: These are a question pair that we are interested in evaluating
    #num_lvl_vec: this is a vector containing the number of levels for each attribute. For example, [4,3,5] denotes
    #three attributes with 4, 3, and 5 levels respectively.
    
    #Ensure that the given arguments are numpy arrays for processing below.
    x_0_vec = np.array(x_0)
    y_0_vec = np.array(y_0)
    mu_0_vec = np.array(mu_0)
    Sig_0_vec = np.array(Sig_0)
    
    #Define first stage variables
    m_0 = np.dot(mu_0_vec,x_0_vec - y_0_vec)
    v_0 = np.dot(x_0_vec-y_0_vec,np.dot(Sig_0_vec,x_0_vec-y_0_vec))
    
    #Gather the posterior information given the two scenarios where the individual picks x over y or they pick
    #y over x
    [mu_10,Sig_10] = moment_matching_update(x_0_vec,y_0_vec,mu_0_vec,Sig_0_vec)
    [mu_11,Sig_11] = moment_matching_update(y_0_vec,x_0_vec,mu_0_vec,Sig_0_vec)
    
    #Solve g_opt(mu_10,Sig_10)[0] and g_opt(mu_11,Sig_11)[0] in order to get optimal questions for each scenario, where
    #each scenario is the individual picking x or y.
    
    [x_10,y_10] = g_opt_multi_lvl(mu_10,Sig_10,mu_log_coeff,Sig_log_coeff,num_lvl_vec)[1:3]
    [x_11,y_11] = g_opt_multi_lvl(mu_11,Sig_11,mu_log_coeff,Sig_log_coeff,num_lvl_vec)[1:3]
    
    #Define second stage variables.
    m_10 = np.dot(np.array(mu_10),np.array(x_10) - np.array(y_10))
    v_10 = np.dot(np.array(x_10) - np.array(y_10),np.dot(np.array(Sig_10),np.array(x_10)-np.array(y_10)))
    m_11 = np.dot(np.array(mu_11),np.array(x_11) - np.array(y_11))
    v_11 = np.dot(np.array(x_11) - np.array(y_11),np.dot(np.array(Sig_11),np.array(x_11)-np.array(y_11)))
    
    #Calculate the two-step value. fst_stg_g_sum_term are the two summation terms of g(m_0,v_0)
    fst_stg_g_sum_term = g_fun(m_0,v_0)[1:]
    two_step_g = g_fun(m_10,v_10)[0]*fst_stg_g_sum_term[0] + g_fun(m_11,v_11)[0]*fst_stg_g_sum_term[1]
    
    return [two_step_g,x_10,y_10,x_11,y_11]