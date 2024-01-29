#!/usr/bin/env python
# coding: utf-8


import numpy as np
import scipy.integrate
import math
import gurobipy as gp
from gurobipy import GRB
import itertools
import random


#Compute expectation and variance of Z random variable parameterized by m and v

def z_expectation_variance(m,v):
    #m is question mean. Should be a real number
    #v is question variance. Should be a real number greater than 0
    
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


#This function is for conducting moment-matching using the updating equations in proposition 1. This returns the posterior
#expectation and covariance matrix.
def moment_matching_update(x,y,mu_prior,Sig_prior):
    #x and y: These are numpy arrays of binary variables.
    #mu_prior: prior expectation over the DM's partworth. Should be a numpy array.
    #Sig_prior: prior covariance matrix over the DM's partworth. Should be a square two-dimensional numpy array
    #having rows and columns with same number of entries corresponding to mu_prior.
    
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



#Define a set that has all the differences between binary products

def product_diff_list(n):
    #n: the number of attributes of the products
    
    #Example of itertools.product:
    #itertools.product(range(2), repeat = 3) --> 000 001 010 011 100 101 110 111
    p_d_l = list(itertools.product([-1.0,0.0,1.0],repeat = n))
    
    #Return the index of the tuple with all 0s.
    zero_index = p_d_l.index(tuple([0]*n))

    #Note that at this point, product_diff_list contains some redundant information. Due to
    #the symmetry of the one-step acquisition function in terms of question mean, question pairs such as 
    #(-1,-1,-1,...,-1) and (1,1,1,...,1) (i.e. negative multiples) will evaluate as the same under the one-step
    #acquisition function. Due to the structure of product_diff_list, we can remove every question pair before and including
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
    #init_mu: This is the initial expectation of the partworths. Should be a numpy array.
    
    #init_Sig: This is the initial covariance matrix of the partworths. Should be a square two-dimensional numpy array
    #having rows and columns with same number of entries corresponding to init_mu.
    
    #batch_size: This is the number of questions in each batch. Should be an integer greater than or equal to one.
    
    #L: This is a vector which holds varying levels of signal (multiply with mu). For example,
    #we could have L = [0.25,1.0,4.0]
    
    #S: This is a vector which holds varying levels of noise (multiply with Sig). For example,
    #we could have S = [0.25,1.0,4.0]
    
    #num_random_batches: This is the number of random batches that we will generate for collecting data on log(D-err),
    #AM, AV, and AO (and MO). This set of random batches will be used for each level combination of L and S. Should be an integer
    #greater than or equal to one.
    
    #num_true_partworths: This is the number of true/baseline partworths we will use to evaluate the d-error of a design. Should be
    #an integer greater than or equal to one.
    
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
                    
                #We average the d-values from the simulation for a batch and store it in a list. This is the D-error of the batch i
                #under distribution N(L*mu, S*Sig).
                average_d_error.append(np.mean(batch_simulate_d_values))
                
    return average_orthogonality, maximum_orthogonality, average_question_mean, average_question_variance, L_mu, S_Sig, init_sqrt_determinant, average_d_error


#This function constructs a batch design based off of average question mean, average question variance, and average
#question orthogonality. For the average question orthogonality, we take the absolute value of the summands rather than
#the square. We also normalize mu and Sig in the objective so that we do not need to keep on refitting the parameters 
#that go with question mean, question variance, and question orthogonality.

def batch_design_AO(mu,Sig,batch_size,quest_mean_log_coeff,quest_var_log_coeff,quest_orth_log_coeff,t_lim = 100,logfile=False):
    #mu: expectation of prior on the DM's partworth. Should be a numpy array.
    
    #Sig: Covariance matrix of prior on the DM's partworth.  Should be a square two-dimensional numpy array
    #having rows and columns with same number of entries corresponding to mu.
    
    
    #batch_size: the number of questions we want to return in our batch design. This should be less or equal to the number
    #of attributes (length of mu).
    
    #quest_mean_log_coeff: this is a fitting parameter that goes with the average question mean and is obtained 
    #by fitting a linear model log (D-err/Init_det) ~ AM/||l*mu|| + AV/||s*Sig|| + AO/||s*Sig|| + ||l*mu|| + ||s*Sig|| and using the fitted parameter that goes with
    #AM/||l*mu||.
    
    #quest_var_log_coeff: this is a fitting parameter that goes with the average question variance and is obtained 
    #by fitting a linear model log (D-err/Init_det) ~ AM/||l*mu|| + AV/||s*Sig|| + AO/||s*Sig|| + ||l*mu|| + ||s*Sig|| and using the fitted parameter that goes with
    #AV/||s*Sig||.
    
    #quest_orth_log_coeff: this is a fitting parameter that goes with the average question orthogonality and is obtained 
    #by fitting a linear model log (D-err/Init_det) ~ AM/||l*mu|| + AV/||s*Sig|| + AO/||s*Sig|| + ||l*mu|| + ||s*Sig|| and using the fitted parameter that goes with
    #AO/||s*Sig||.
    
    #In the above three comments regarding the coefficients, (l,s) are scaling parameters for mu and Sig that divide the space into 
    #different signal-to-noise ratio regions.
    
    #t_lim: this is the max amount of time we want to take to construct the batch
    #logfile: determine whether to print out a logfile of the optimization procedure.

    #Make sure that quest_orth_log_coeff is greater or equal to zero. Otherwise, we will
    #have an unbounded optimization problem. In most situations, the fitting procedure
    #will result in a positive value for quest_orth_log_coeff, but very rarely the fitting
    #procedure will give a statistically non-significant but negative value for
    #quest_orth_log_coeff that makes the optimization problem unbounded. When the quest_orth_log_coeff
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
    #mu: expectation of prior on the DM's partworth. Should be a numpy array.
    
    #Sig: Covariance matrix of prior on the DM's partworth.  Should be a square two-dimensional numpy array
    #having rows and columns with same number of entries corresponding to mu.
    
    
    #batch_size: the number of questions we want to return in our batch design. This should be less or equal to the number
    #of attributes (length of mu).
    
    #quest_mean_log_coeff: this is a fitting parameter that goes with the average question mean and is obtained 
    #by fitting a linear model log (D-err/Init_det) ~ AM/||l*mu|| + AV/||s*Sig|| + MO/||s*Sig|| + ||l*mu|| + ||s*Sig|| and using the fitted parameter that goes with
    #AM/||l*mu||
    
    #quest_var_log_coeff: this is a fitting parameter that goes with the average question variance and is obtained 
    #by fitting a linear model log (D-err/Init_det) ~ AM/||l*mu|| + AV/||s*Sig|| + MO/||s*Sig|| + ||l*mu|| + ||s*Sig|| and using the fitted parameter that goes with
    #AV/||s*Sig||
    
    #quest_orth_log_coeff: this is a fitting parameter that goes with the average question orthogonality and is obtained 
    #by fitting a linear model log (D-err/Init_det) ~ AM/||l*mu|| + AV/||s*Sig|| + MO/||s*Sig|| + ||l*mu|| + ||s*Sig|| and using the fitted parameter that goes with
    #MO/||s*Sig||
    
    #In the above three comments, (l,s) are scaling parameters for mu and Sig that divide the space into different 
    #signal-to-noise ratio regions.
    
    #t_lim: this is the max amount of time we want to take to construct the batch
    
    #logfile: determine whether to print out a logfile of the optimization procedure.

    # Make sure that quest_orth_log_coeff is greater or equal to zero. Otherwise, we will
    # have an unbounded optimization problem. In most situations, the fitting procedure
    # will result in a positive value for quest_orth_log_coeff, but very rarely the fitting
    # procedure will give a statistically non-significant but negative value for
    # quest_orth_log_coeff that makes the optimization problem unbounded. When the quest_orth_log_coeff
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

