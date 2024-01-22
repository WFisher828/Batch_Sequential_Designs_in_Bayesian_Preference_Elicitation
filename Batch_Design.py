#!/usr/bin/env python
# coding: utf-8



#This module 'Batch_Design' includes functions to construct a batch design.

#Functions in this module include:

#batch_design_AO 
#batch_design_MO

import gurobipy as gp
from gurobipy import GRB
import numpy as np

#This function constructs a batch design based off of average question mean, average question variance, and average
#question orthogonality. For the average question orthogonality, we take the absolute value of the summands rather than
#the square. We also normalize mu and Sig in the objective so that we do not need to keep on refitting the parameters 
#that go with question mean, question variance, and question orthogonality.

def batch_design_AO(mu,Sig,batch_size,quest_mean_log_coeff,quest_var_log_coeff,quest_orth_log_coeff,t_lim = 100,logfile=False):
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
    #logfile: determine whether to print out a logfile of the optimization procedure.

    #Make sure that quest_orth_log_coeff is greater or equal to zero. Otherwise, we will
    #have an unbounded optimization problem. In most situations, the fitting procedure
    #will result in a positive value for quest_orth_log_coeff, but very rarely the fitting
    #procedure will give a statistically non-significant but negative value for
    #quest_orth_log_coeff that makes the optimization problem bounded. When the quest_orth_log_coeff
    #is less than 0, we decide to set it equal to 0. This will result in a bounded optimization problem,
    #but the quality of the solution in terms of D-error may not be sufficient because we are no
    #longer controlling orthogonality in the objective function.
    if quest_orth_log_coeff<0.0:
        quest_orth_log_coeff = 0.0

    # This is the number of attributes for the products
    n = len(Sig[0])
    
    m = gp.Model("mip1")
    m.setParam('Timelimit',t_lim)
    if logfile:
        m.setParam('LogFile',"Batch_AO_batchsize"+str(batch_size)+"_meancoeff_"+str(quest_mean_log_coeff)+"_varcoeff_"+
               str(quest_var_log_coeff)+"_orthcoeff_"+str(quest_orth_log_coeff)+"_v5.txt")
    
    #calculate 2-norms of mu and Sigma
    mu_2norm = np.linalg.norm(mu,2)
    Sig_2norm = np.linalg.norm(Sig,2)
    
    #List of tuples for delta variable
    if batch_size > 1:
        delta_tuples = []
        for i in range(batch_size):
            for j in range(i+1,batch_size):
                delta_tuples.append((i,j))
    
    #Set up the x_i and y_i, i = 1,...,batchsize
    X = m.addMVar((batch_size,n),vtype = GRB.BINARY)
    Y = m.addMVar((batch_size,n),vtype = GRB.BINARY)
    if batch_size > 1:
        Delta = m.addVars(delta_tuples, lb=0.0, vtype = GRB.CONTINUOUS)
    
    #Set up the objective function.
    if batch_size > 1:
        m.setObjective((quest_mean_log_coeff/(batch_size*mu_2norm))*sum([mu@X[i] - mu@Y[i] for i in range(batch_size)]) + 
                       (quest_var_log_coeff/(batch_size*Sig_2norm))*sum([X[i]@Sig@X[i] - X[i]@(2.0*Sig)@Y[i] + 
                       Y[i]@Sig@Y[i] for i in range(batch_size)]) + 
                           (quest_orth_log_coeff/(batch_size*(batch_size-1)*Sig_2norm/2))*sum([Delta[i,j] for i in range(batch_size) for j in range(i+1,batch_size)]),GRB.MINIMIZE)
        
    if batch_size == 1:
        m.setObjective((quest_mean_log_coeff/(batch_size*mu_2norm))*sum([mu@X[i] - mu@Y[i] for i in range(batch_size)]) + 
                       (quest_var_log_coeff/(batch_size*Sig_2norm))*sum([X[i]@Sig@X[i] - X[i]@(2.0*Sig)@Y[i] + 
                       Y[i]@Sig@Y[i] for i in range(batch_size)]),GRB.MINIMIZE)
    
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


#This function constructs a batch design based off of average question mean, average question variance, and MAXIMUM
#question orthogonality. For the average question orthogonality, we take the absolute value of the summands rather than
#the square. We also normalize mu and Sig in the objective so that we do not need to keep on refitting the parameters 
#that go with question mean, question variance, and question orthogonality.

def batch_design_MO(mu,Sig,batch_size,quest_mean_log_coeff,quest_var_log_coeff,quest_orth_log_coeff,t_lim = 100,logfile=False):
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

    # Make sure that quest_orth_log_coeff is greater or equal to zero. Otherwise, we will
    # have an unbounded optimization problem. In most situations, the fitting procedure
    # will result in a positive value for quest_orth_log_coeff, but very rarely the fitting
    # procedure will give a statistically non-significant but negative value for
    # quest_orth_log_coeff that makes the optimization problem bounded. When the quest_orth_log_coeff
    # is less than 0, we decide to set it equal to 0. This will result in a bounded optimization problem,
    # but the quality of the solution in terms of D-error may not be sufficient because we are no
    # longer controlling orthogonality in the objective function.
    if quest_orth_log_coeff < 0.0:
        quest_orth_log_coeff = 0.0

    #This is the number of attributes for the products
    n = len(Sig[0])
    
    m = gp.Model("mip1")
    m.setParam('Timelimit',t_lim)
    if logfile:
        m.setParam('LogFile',"Batch_MO_batchsize"+str(batch_size)+"_meancoeff_"+str(quest_mean_log_coeff)+"_varcoeff_"+
               str(quest_var_log_coeff)+"_orthcoeff_"+str(quest_orth_log_coeff)+"_v5.txt")
    
    #calculate 2-norms of mu and Sigma
    mu_2norm = np.linalg.norm(mu,2)
    Sig_2norm = np.linalg.norm(Sig,2)
    
    #Set up the x_i and y_i, i = 1,...,batchsize
    X = m.addMVar((batch_size,n),vtype = GRB.BINARY)
    Y = m.addMVar((batch_size,n),vtype = GRB.BINARY)
    if batch_size > 1:
        delta = m.addVar(lb=0.0, vtype = GRB.CONTINUOUS)
    
    #set up the objective function
    if batch_size > 1:
        m.setObjective((quest_mean_log_coeff/(batch_size*mu_2norm))*sum([mu@X[i] - mu@Y[i] for i in range(batch_size)]) +
                      (quest_var_log_coeff/(batch_size*Sig_2norm))*sum([X[i]@Sig@X[i] - X[i]@(2.0*Sig)@Y[i] + 
                       Y[i]@Sig@Y[i] for i in range(batch_size)]) + (quest_orth_log_coeff/Sig_2norm)*delta,GRB.MINIMIZE)
    
    if batch_size == 1:
         m.setObjective((quest_mean_log_coeff/(batch_size*mu_2norm))*sum([mu@X[i] - mu@Y[i] for i in range(batch_size)]) +
                      (quest_var_log_coeff/(batch_size*Sig_2norm))*sum([X[i]@Sig@X[i] - X[i]@(2.0*Sig)@Y[i] + 
                       Y[i]@Sig@Y[i] for i in range(batch_size)]),GRB.MINIMIZE)
            
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