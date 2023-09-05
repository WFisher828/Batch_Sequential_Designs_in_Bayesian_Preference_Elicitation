#!/usr/bin/env python
# coding: utf-8

# In[1]:


#This module "Experiment_Framework" has functions which are used in conducting numerical experiments. 
#These functions include:
#1. product_diff_list
#2. question_extractor
#3. enum_two_step
#4. enum_two_step_opt_worst
#5. quantile_test_enum_data
#6. MSE_det_test 
#7. new_mse_det_experiment

# (IGNORE) 6,7,8 can be consolidated (AND HAVE BEEN). MAKE A FUNCTION TO PLOT DATA FROM MSE_det_test, THIS WILL REPLACE WHAT 
# MSE_test_2 and det_test_2 did.


# In[2]:


import numpy as np
import pandas as pd
import gurobipy as gp # Feb 27 2023
from gurobipy import GRB # Feb 27 2023
import itertools
import matplotlib.pyplot as plt
import random
import seaborn as sns
from itertools import combinations
import sklearn.datasets
import scipy.integrate #Feb27 2023
import math # Feb 27 2023

import time

import sys
sys.path.append(r'C:\Users\wsfishe\Desktop\PreferenceElicitationCode')
from Questionnaire_Procedure import moment_matching_update, g_opt, two_stage_g_opt,two_step_g_acq, g_opt_multi_lvl, multlvl_two_step_g_acq #Feb 27, 2023 was " * "
from Baseline_Functions_Definitions import g_fun_linear_regression
from Batch_Design_and_Rollout import batch_design_delta_penalty, question_selection_prob, rollout, monte_carlo_rollout,rollout_with_batch_design_acquisition, coordinate_exchange_acq

#From Questionnaire_Procedure we are importing:

#PACKAGES:
# gurobipy
# numpy
# scipy.integrate
# math

#FUNCTIONS:
# z_expectation_variance (originally from Baseline_Functions_Definitions)
# g_fun (originally from Baseline_Functions_Definitions)
# moment_matching_update
# g_opt
# two_stage_g_opt
# two_step_g_acq

#!!NO LONGER USING!!
#VARIABLES:
#mu_log_coeff = 0.03596804494858049
#Sig_log_coeff = -0.020785433813507195


# In[3]:


#THIS WILL NEED TO BE MOVED!!!
#This is used to instantiate the coefficient values for optimization when there are 6 attributes.
#mu_log_coeff_6,Sig_log_coeff_6 = g_fun_linear_regression(0,12.70,0.1,42.0,24,84)
#print(mu_log_coeff_6,Sig_log_coeff_6)


# In[4]:


#REMOVE!!!
#print(mu_log_coeff_6)


# In[5]:


#THIS MAY BE REMOVED!!!
#This is used to instantiate the coefficient values for optimization when there are 12 attributes.
#mu_log_coeff_12,Sig_log_coeff_12 = g_fun_linear_regression(0, 24.50, 0.1, 156.0, 49, 312)
#print(mu_log_coeff_12,Sig_log_coeff_12)


# In[6]:


#THIS MAY BE REMOVED!!!
#This is used to instantiate the linear approximation coefficient values for optimization in the
#multi-level case when there are 6 attributes and 3 levels per attribute. 
#mu_log_coeff_6_mult_3, Sig_log_coeff_6_mult_3 = g_fun_linear_regression(0.0, 36.0, 0.1, 2*4.0*42.0, 72, 2*2*4*42)


# In[7]:


#REMOVE!!!
#print(mu_log_coeff_6_mult_3,Sig_log_coeff_6_mult_3)


# In[8]:


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


# In[9]:


#Define a set that has all the differences between quantitative multi-level products. For simplicity, we assume that
#the number of levels are the same among all attributes

def multlvl_product_diff_list(n,num_lvl):
    #n: the number of attributes of the products
    #num_lvl: this is a integer declaring the number of levels for each attribute. For example, n=3, num_lvl=4 denotes
    #three attributes with 4 quantitative levels each. Each attribute in this case
    #would be encode with levels 0, 1, 2, and 3.
    
    ml_p_d_l = list(itertools.product(range(-(num_lvl-1),num_lvl), repeat = n))
    
    #Return the index of the tuple with all 0s.
    zero_index = ml_p_d_l.index(tuple([0]*n))
    
    #Note that at this point, product_diff_list contains some redundant information. Due to
    #the symmetry of the one-step and two-step acquisition function in terms of question mean, question pairs such as 
    #(-1,-1,-1,...,-1) and (1,1,1,...,1) (i.e. negative multiples) will evaluate as the same under the one-step and two-step
    #acquisition functions. Due to the structure of product_diff_list, we can remove every question pair before and including
    #the question pair with all zero entries in order to remove this redundant information.
    for i in range(0,zero_index + 1):
        ml_p_d_l.pop(0)
    
    ml_p_d_l = [np.array(a) for a in ml_p_d_l]

    return ml_p_d_l


# In[10]:


#THIS MAY BE MOVED!!!
#Example8: product_diff_list
#ex8_n = 3
#print(product_diff_list(ex8_n))


# In[11]:


#THIS MAY BE MOVED!!!
#Example8a: multlvl_product_diff_list
#ex8a_n = 3
#ex8a_num_lvl = 3
#print(multlvl_product_diff_list(ex8a_n,ex8a_num_lvl))


# In[12]:


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


# In[13]:


#Given a vector d with entries coming from -(m-1), -(m-2),...,0,...,m-2,m-1, find two vectors x and y with entries
#in 0,1,...,m-2,m-1 whose difference is d.
def multlvl_question_extractor(prod,num_lvl):
    #prod: This is a vector with entries coming from -(m-1), -(m-2),...,0,...,m-2,m-1 
    #that represents the difference between two products, where the two products have attributes with entrie coming from
    #0,1,...,m-2,m-1
    #num_lvl: This is the number of levels in each attribute of the product.
    
    n = len(prod)
    
    x = [0]*n
    y = [0]*n
    for i in range(n):
        for j in range(-(num_lvl - 1),num_lvl):
            if prod[i]==j and j<0:
                x[i] = 0
                y[i] = j
            if prod[i]==j and j==0:
                x[i] = 0
                y[i] = 0
            if prod[i]==j and j>0:
                x[i] = j
                y[i] = 0
    
    return x,y          


# In[14]:


#A function which returns a list of all the enumerated two-step acquisition values of a given product set. We use the exact
#two-step g function. We return the prod_diff_set in case there was sampling done and we have interest in the sampled
#question pairs. We also return a list of all the first stage and two second stage questions.
def enum_two_step(mu_vec, Sig_mat, mu_log_coeff, Sig_log_coeff, prod_diff_set, prod_samp_num = 0):
    #mu_vec: expectation of prior on beta
    #Sig_mat: covariance matrix of prior on beta
    #mu_log_coeff and Sig_log_coeff: coefficients that are used in the optimization prblem.
    #prod_diff_set: set of question pairs we are enumerating over. This should be created by using product_diff_list.
    #prod_samp_num: Should be a positive integer. Used in obtaining a random sample of product pairs if needed.
    
    #Sample a number of product pairs from the product pairs list, if needed in the case where
    #there are a large number of attributes. Will possibly need a seed for random.sample()
    if prod_samp_num>0:
        prod_diff_set = random.sample(prod_diff_set,prod_samp_num)
    
    #define a list to store enumerated two-step values, along with a list to save the first stage and two second stage
    #questions.
    prod_diff_len = len(prod_diff_set)
    two_step_g_val = [0]*prod_diff_len
    first_stage_second_stage_questions = [0]*prod_diff_len
    
    #calculate two-step values for all question pairs
    for i in range(0, prod_diff_len):
        x_0,y_0 = question_extractor(prod_diff_set[i])
        two_step = two_step_g_acq(mu_vec,Sig_mat,mu_log_coeff,Sig_log_coeff,x_0,y_0)
        two_step_g_val[i] = two_step[0]
        first_stage_second_stage_questions[i] = [x_0,y_0,two_step[1],two_step[2],two_step[3],two_step[4]]
        
    return two_step_g_val,prod_diff_set,first_stage_second_stage_questions


# In[15]:


#A function which returns a list of all the enumerated two-step acquisition values of a given product set in the
#multi-level attribute setting. We use the exact
#two-step g function. We return the prod_diff_set in case there was sampling done and we have interest in the sampled
#question pairs. We also return a list of all the first stage and two second stage questions.

def multlvl_enum_two_step(mu_vec, Sig_mat, mu_log_coeff, Sig_log_coeff, multlvl_prod_diff_set, num_lvl, prod_samp_num = 0):
    #mu_vec: expectation of prior on beta
    #Sig_mat: covariance matrix of prior on beta
    #mu_log_coeff and Sig_log_coeff: coefficients that are used in the optimization prblem.
    #multlvl_prod_diff_set: set of question pairs we are enumerating over. This should be created by using multlvl_product_diff_list.
    #num_lvl: This is the number of levels in each attribute of the product.
    #prod_samp_num: Should be a positive integer. Used in obtaining a random sample of product pairs if needed.
    
    #Sample a number of product pairs from the product pairs list, if needed in the case where
    #there are a large number of attributes. Will possibly need a seed for random.sample()
    if prod_samp_num>0:
        multlvl_prod_diff_set = random.sample(multlvl_prod_diff_set,prod_samp_num)
        
    #define a list to store enumerated two-step values, along with a list to save the first stage and two second stage
    #questions.
    multlvl_prod_diff_len = len(multlvl_prod_diff_set)
    multlvl_two_step_g_val = [0]*multlvl_prod_diff_len
    first_stage_second_stage_questions = [0]*multlvl_prod_diff_len
    
    #calculate two-step values for all question pairs
    n = len(mu_vec)
    for i in range(0, multlvl_prod_diff_len):
        x_0,y_0 = multlvl_question_extractor(multlvl_prod_diff_set[i],num_lvl)
        multlvl_two_step = multlvl_two_step_g_acq(mu_vec,Sig_mat,mu_log_coeff,Sig_log_coeff,x_0,y_0,[num_lvl]*n)
        multlvl_two_step_g_val[i] = multlvl_two_step[0]
        first_stage_second_stage_questions[i] = [x_0,y_0,multlvl_two_step[1],multlvl_two_step[2],multlvl_two_step[3],multlvl_two_step[4]]
        
    return multlvl_two_step_g_val,multlvl_prod_diff_set,first_stage_second_stage_questions


# In[16]:


#NEEDS TO BE MOVED!!!
#Ex1
#mu_vec_1 = np.array([1.0]*6)
#Sig_mat_1 = np.identity(6)
#prod_1 = product_diff_list(6)
#print(enum_two_step(mu_vec_1,Sig_mat_1,mu_log_coeff_6,Sig_log_coeff_6,prod_1))


# In[17]:


#THIS NEEDS TO BE MOVED!!!
#Ex1a
#mu_vec_1a = np.array([1.0,1.0,1.0,1.0])
#Sig_mat_1a = np.identity(4)
#prod_1a = multlvl_product_diff_list(4,4)
#num_lvl_1a = 4
#print(multlvl_enum_two_step(mu_vec_1a,Sig_mat_1a,mu_log_coeff_12,Sig_log_coeff_12,prod_1a,num_lvl_1a))


# In[18]:


#A function which returns the best performing solution and their two-step acquisition values, along with the corresponding
#first stage and two second stage questions.
#We use the exact two-step g function. Used for experimenting and gaining insight into two-step acquisition.
#NOTE: THIS CAN BE USED FOR MULTILEVEL CASE.

def enum_two_step_opt(two_step_values,first_second_stage_question):
    #two_step_values: A list of two_step values. This should come from the function enum_two_step
    
    
    #Find min index of the two-step enumeration. Use these values to return the optimal two-step value,
    #along with their corresponding questions.
    two_step_array = np.array(two_step_values)
    min_index = np.argmin(two_step_array)
    max_index = np.argmax(two_step_array)
    opt_x0 = first_second_stage_question[min_index][0]
    opt_y0 = first_second_stage_question[min_index][1]
    opt_x10 = first_second_stage_question[min_index][2]
    opt_y10 = first_second_stage_question[min_index][3]
    opt_x11 = first_second_stage_question[min_index][4]
    opt_y11 = first_second_stage_question[min_index][5]
    opt_val = two_step_values[min_index]

    return [opt_val,opt_x0,opt_y0,opt_x10,opt_y10,opt_x11,opt_y11]


# In[19]:


#THIS NEEDS TO BE MOVED!!!
#Solve the two-step problem via enumeration for 6 attributes and look at the structure of the optimal solution,
#in particular look at the relationship between the first stage and the two second stage solutions. 

#rng = np.random.default_rng(100) 
#np.random.seed(100)
#random.seed(100)

#attribute numbers for enumeration
#attr_enum = 6

#Prior Parameters
#mu_vec_enum = np.array([-0.25,0.5,-0.75,1.0,-1.25,1.5]) #np.array([1.0]*attr_enum)#rng.uniform(low = -1.0, high = 1.0, size = attr_enum)
#Sig_mat_enum = 1.25*np.array([[1.0,-0.5,0.25,-0.125,0.0625,-0.03125],
                      #[-0.5,1.0,-0.5,0.25,-0.125,0.0625],
                     #[0.25,-0.5,1.0,-0.5,0.25,-0.125],
                     #[-0.125,0.25,-0.5,1.0,-0.5,0.25],
                     #[0.0625,-0.125,0.25,-0.5,1.0,-0.5],
                     #[-0.03125,0.0625,-0.125,0.25,-0.5,1.0]])#np.identity(attr_enum)#sklearn.datasets.make_spd_matrix(attr_enum)

#List of question pairs we will enumerate on
#prod_list_enum = product_diff_list(attr_enum)
#print(len(prod_list_enum))

#Do the enumeration procedure
#start_time_twostep = time.perf_counter()
#enumerated_solution_list = enum_two_step(mu_vec_enum,Sig_mat_enum,mu_log_coeff_6,Sig_log_coeff_6,prod_list_enum)
#print(time.perf_counter() - start_time_twostep, "seconds")

#Save the two-step g values and the corresponding first stage and 2 second stage questions
#opt_sol_questions = enum_two_step_opt(enumerated_solution_list[0],enumerated_solution_list[2])
#print(opt_sol_questions)

#Print prior parameters
#print(mu_vec_enum)
#print(Sig_mat_enum)


# In[20]:


#THIS WILL NEED TO BE MOVED!!!
#Create a matrix to save values measuring the inner product under 
#Sig_mat_enum of the 3 question pairs from above.
#ortho_meas_matr = np.zeros((3,3))
#q_0 = np.array(opt_sol_questions[1]) - np.array(opt_sol_questions[2])
#q_10 = np.array(opt_sol_questions[3]) - np.array(opt_sol_questions[4])
#q_11 = np.array(opt_sol_questions[5]) - np.array(opt_sol_questions[6])
#question_list = [q_0,q_10,q_11]
#print(opt_sol_questions)
#print(question_list)
#for i in range(3):
    #for j in range(3):
        #ortho_meas_matr[i,j] = np.dot(question_list[i],np.dot(Sig_mat_enum,question_list[j]))
#print(ortho_meas_matr)

#Look at the optimal solution from the approximate two-step method. Can be used to compare with the true
#optimal solution.
#two_step_apprx_opt = two_stage_g_opt(mu_vec_enum,Sig_mat_enum,mu_log_coeff_6,Sig_log_coeff_6,0.01)
#print(two_step_apprx_opt)


# In[21]:


#REMOVE!!!
#print(two_step_g_acq(mu_vec_enum,Sig_mat_enum,mu_log_coeff_6,Sig_log_coeff_6,two_step_apprx_opt[2],two_step_apprx_opt[3]))


# In[22]:


#This code is used to get the quantile of our two-step approximate solution value relative to the optimal solution value
#as well as the one-step solution value relative to the optimal solution value. Also returns
#a dataframe containing all of the enumeration data from each replication

def quantile_test_enum_data(attr_num,prod_diff_list,rep_num,scale,location,mu_log_coeff,Sig_log_coeff,epsilon_0 = 0.1):
    #attr_num: the number of attributes the products have.
    
    #prod_diff_list: A list of product pairs we will enumerate over for each replication. 
    #Products should have same number of attributes as attr_num.
    
    #rep_num: the number of replications we wish to make
    
    #scale: A parameter to increase/decrease the determinant of the random covariance matrix
    
    #location: A parameter to increase/decrease the magnitude of the partworth estimator.
    
    #mu_log_coeff and Sig_log_coeff: linear coefficients that are used in the optimization problem
    
    #epsilon_0: A parameter that is used in the two-step optimization problem with orthogonality constraints. 
    #It sets the degree of orthogonality, and we want it to be small.
    
    
    #create two lists to store quantile data for one-step and two-step
    quant_val_one_step = [0.0]*rep_num
    quant_val_two_step = [0.0]*rep_num
    
    one_step_sol_val_list = [0.0]*rep_num
    two_step_sol_val_list = [0.0]*rep_num
    
    enum_df = pd.DataFrame()
    
    #Collect quantile data
    for i in range(0, rep_num):
        #prior parameters
        mu_0_rand = location*rng.uniform(low = -1.0, high = 1.0, size = attr_num) #use rng, this is Unif(-1.0,1.0)
        Sig_0_rand = scale*sklearn.datasets.make_spd_matrix(attr_num) #NEED SEED FOR THIS
        
        #Solve the two-step(with orthgonality constraint) and one-step optimization problems
        [w_0,z_0,w_1,z_1] = two_stage_g_opt(mu_0_rand,Sig_0_rand,mu_log_coeff,Sig_log_coeff,epsilon_0)
        [x_0, y_0] = g_opt(mu_0_rand,Sig_0_rand,mu_log_coeff,Sig_log_coeff)[1:]
        
        #Save the two-step acquisition function values of the one-step and two-step solutions
        one_step_sol_val = two_step_g_acq(mu_0_rand, Sig_0_rand,mu_log_coeff,Sig_log_coeff, x_0, y_0)[0]
        two_step_appr_sol_val_0 = two_step_g_acq(mu_0_rand, Sig_0_rand,mu_log_coeff,Sig_log_coeff, w_0,z_0)[0]
        two_step_appr_sol_val_1 = two_step_g_acq(mu_0_rand, Sig_0_rand,mu_log_coeff,Sig_log_coeff, w_1,z_1)[0]
        two_step_appr_sol_val = min(two_step_appr_sol_val_0,two_step_appr_sol_val_1)
        
        one_step_sol_val_list[i] = one_step_sol_val
        two_step_sol_val_list[i] = two_step_appr_sol_val
        
        #Create a list storing all of the enumerated values of all question pairs evaluated
        #under two-step acquisition function. Return a dataframe containing enumeration if needed.
        enum_data = enum_two_step(mu_0_rand,Sig_0_rand,mu_log_coeff,Sig_log_coeff,prod_diff_list)[0]
        enum_df['Trial: %s'%(str(i))] = enum_data
        
        #Calculate the quantile values of one-step and two-step. The best performing solution will have a quant
        #value of 1.0 and the worst performing solution will have a quant value of 0.
        enum_data_len = len(enum_data)
        one_step_quantile = [1 for x in enum_data if x >= one_step_sol_val]
        two_step_quantile = [1 for x in enum_data if x >= two_step_appr_sol_val]
        quant_val_one_step[i] = sum(one_step_quantile)/enum_data_len
        quant_val_two_step[i] = sum(two_step_quantile)/enum_data_len
     
    return [quant_val_one_step, quant_val_two_step, enum_df, one_step_sol_val_list, two_step_sol_val_list]


# In[23]:


#THIS NEEDS TO BE MOVED!!!
#Create Quantile Data with Two Boxplots for one-step vs two-step. 
#rng = np.random.default_rng(100) 
#np.random.seed(100)
#random.seed(100)
#These are the parameters we will use
#attr_num_quant = 6
#prod_diff_list_quant = product_diff_list(attr_num_quant) # TAKE A SAMPLE IF attr_num_quant IS LARGE
#sample_size = 3**6
#prod_diff_list_quant = random.sample(prod_diff_list_quant,sample_size)
#rep_num_quant = 100
#scale_quant = 1.0
#location_quant = 8.0


# In[24]:


#THIS NEEDS TO BE MOVED!!!
#Create Quantile Data with Two Boxplots for one-step vs two-step, as well as enumeration data.
#quant_data = quantile_test_enum_data(attr_num_quant,prod_diff_list_quant,rep_num_quant,scale_quant,location_quant,
                                    #mu_log_coeff_6,Sig_log_coeff_6)


# In[25]:


#THIS NEEDS TO BE MOVED!!!
#Create Quantile Data with Two Boxplots for one-step vs two-step

#Place quant_data into a dataframe and convert to csv
#zipped_quant = list(zip(quant_data[0], quant_data[1]))
#df_quantile = pd.DataFrame(zipped_quant,columns = ['one-step','two-step'])
#df_quantile.to_csv(r'C:\Users\wsfishe\Desktop\quantile_Aug5_%s_%s_%s_%s.csv'%(str(attr_num_quant),str(rep_num_quant),
                                                                             #str(scale_quant),str(location_quant)),index=False,header=True)

#quant_data[2].to_csv(r'C:\Users\wsfishe\Desktop\enum_data_Aug5_%s_%s_%s_%s.csv'%(str(attr_num_quant),str(rep_num_quant),
                                                                             #str(scale_quant),str(location_quant)),index=False,header=True)

#zipped_solval = list(zip(quant_data[3], quant_data[4]))
#df_solval = pd.DataFrame(zipped_solval,columns = ['one-step','two-step'])
#df_solval.to_csv(r'C:\Users\wsfishe\Desktop\quantile_solval_July11_%s_%s_%s_%s.csv'%(str(attr_num_quant),str(rep_num_quant),
                                                                             #str(scale_quant),str(location_quant)),index=False,header=True)


# In[26]:


#THIS NEEDS TO BE MOVED!!!
#Plot boxplot of quantile information
#labels = ['one-step', 'two_step']
#plt.boxplot([quant_data[0],quant_data[1]], vert=True, patch_artist=True, labels=labels) 
#plt.ylabel("Percentile of Solution")
#plt.title("Quantiles of Approximate Solutions: n=%s, scale = %s,Location = %s, Replication = %s"%(str(attr_num_quant),
                                                                                                  #str(scale_quant),
                                                                                              #str(location_quant),
                                                                                                  #str(rep_num_quant)))
#plt.show()


# In[27]:


#February 27, 2023. This function is used to check the one-step optimal solution's quantile as ranked against
#all of the two-step feasible solutions (two-step) function values. We enumerate all of the possible solution's two-step
#acquisition value, and see where the one-step optimal solution lies within this list.
def one_step_sol_two_step_quantile(attr_num,prod_diff_list,rep_num,scale,location,mu_log_coeff,Sig_log_coeff):
    #attr_num: the number of attributes the products have.
    
    #prod_diff_list: A list of product pairs we will enumerate over for each replication. 
    #Products should have same number of attributes as attr_num.
    
    #rep_num: the number of replications we wish to make
    
    #scale: A parameter to increase/decrease the determinant of the random covariance matrix
    
    #location: A parameter to increase/decrease the magnitude of the partworth estimator.
    
    #mu_log_coeff and Sig_log_coeff: linear coefficients that are used in the optimization problem
    
    #create lists to store quantile data for one-step under two-step solution
    quant_val_one_step = [0.0]*rep_num
    one_step_sol_val_list = [0.0]*rep_num
    
    enum_df = pd.DataFrame()
    
    #Collect quantile data
    for i in range(0, rep_num):
        #prior parameters
        mu_0_rand = location*rng.uniform(low = -1.0, high = 1.0, size = attr_num) #use rng, this is Unif(-1.0,1.0)
        print(mu_0_rand)
        Sig_0_rand = scale*sklearn.datasets.make_spd_matrix(attr_num) #NEED SEED FOR THIS
        
        #Solve the one-step optimization problem's optimal solution
        [x_0, y_0] = g_opt(mu_0_rand,Sig_0_rand,mu_log_coeff,Sig_log_coeff)[1:]
        
        #Save the two-step acquisition function values of the one-step solution
        one_step_sol_val = two_step_g_acq(mu_0_rand, Sig_0_rand,mu_log_coeff,Sig_log_coeff, x_0, y_0)[0]
        
        one_step_sol_val_list[i] = one_step_sol_val
        
        #Create a list storing all of the enumerated values of all question pairs evaluated
        #under two-step acquisition function. Return a dataframe containing enumeration if needed.
        enum_data = enum_two_step(mu_0_rand,Sig_0_rand,mu_log_coeff,Sig_log_coeff,prod_diff_list)[0]
        enum_df['Trial: %s'%(str(i))] = enum_data
        
        #Calculate the quantile values of one-step and two-step. The best performing solution will have a quant
        #value of 1.0 and the worst performing solution will have a quant value of 0.
        enum_data_len = len(enum_data)
        one_step_quantile = [1 for x in enum_data if x >= one_step_sol_val]
        quant_val_one_step[i] = sum(one_step_quantile)/enum_data_len
        print('iteration: ' +str(i))
     
    return [quant_val_one_step, enum_df, one_step_sol_val_list]


# In[28]:


#THIS NEEDS TO BE MOVED!!!
#Create Quantile Data with Boxplot for one-step performance in terms of two-step acquisition value.
#rng = np.random.default_rng(100) 
#np.random.seed(100)
#random.seed(100)
#These are the parameters we will use
#attr_num_OS_quant = 6
#prod_diff_list_OS_quant = product_diff_list(attr_num_OS_quant) # TAKE A SAMPLE IF attr_num_OS_quant IS LARGE
#sample_size = 3**6
#prod_diff_list_onestep_quant = random.sample(prod_diff_list_OS_quant,sample_size)
#rep_num_OS_quant = 100
#scale_OS_quant = 4.0
#location_OS_quant = 1.0


# In[29]:


#THIS NEEDS TO BE MOVED!!!
#Create Quantile Data with one Boxplot for one-step, as well as enumeration data of two-step function.
#quant_data_OS = one_step_sol_two_step_quantile(attr_num_OS_quant,prod_diff_list_OS_quant,rep_num_OS_quant,scale_OS_quant,location_OS_quant,
                                    #mu_log_coeff_6,Sig_log_coeff_6)


# In[30]:


#THIS NEEDS TO BE MOVED!!!
#labels_OS = ['one-step']
#plt.boxplot(quant_data_OS[0], vert=True, patch_artist=True, labels=labels_OS)
#plt.ylabel("Percentile of Solution")

#plt.title("Quantiles of One-Step Solutions (Two-Step Value): n=%s, scale = %s,Location = %s, Replication = %s"%(str(attr_num_OS_quant),
                                                                                                  #str(scale_OS_quant),
                                                                                              #str(location_OS_quant),
                                                                                                  #str(rep_num_OS_quant)))
#plt.show()


# In[31]:


#March 15, 2023. This function is used to check the one-step optimal solution's quantile as ranked against
#all of the two-step feasible solutions (two-step) function values in the multi-level setting. We enumerate all of the possible solution's two-step
#acquisition value, and see where the one-step optimal solution lies within this list.

def multlvl_one_step_sol_two_step_quantile(attr_num,multlvl_prod_diff_list,rep_num,scale,location,mu_log_coeff,Sig_log_coeff, num_lvl):
    #attr_num: the number of attributes the products have.
    
    #multlvl_prod_diff_list: A list of product pairs we will enumerate over for each replication. 
    #Products should have same number of attributes as attr_num. We assume each attribute has same number of levels.
    
    #rep_num: the number of replications we wish to make
    
    #scale: A parameter to increase/decrease the determinant of the random covariance matrix
    
    #location: A parameter to increase/decrease the magnitude of the partworth estimator.
    
    #mu_log_coeff and Sig_log_coeff: linear coefficients that are used in the optimization problem
    
    #num_lvl: This is the number of levels in each attribute of the product.
    
    quant_val_one_step = [0.0]*rep_num
    one_step_sol_val_list = [0.0]*rep_num
    
    enum_df = pd.DataFrame()
    
    #Collect quantile data
    for i in range(0, rep_num):
        #prior parameters
        mu_0_rand = location*rng.uniform(low = -1.0, high = 1.0, size = attr_num) #use rng, this is Unif(-1.0,1.0)
        print(mu_0_rand)
        Sig_0_rand = scale*sklearn.datasets.make_spd_matrix(attr_num) #NEED SEED FOR THIS
        
        #Solve the one-step optimization problem's optimal solution
        [x_0, y_0] = g_opt_multi_lvl(mu_0_rand,Sig_0_rand,mu_log_coeff,Sig_log_coeff,[num_lvl]*attr_num)[1:3]
        
        #Save the two-step acquisition function values of the one-step solution
        one_step_sol_val = multlvl_two_step_g_acq(mu_0_rand, Sig_0_rand,mu_log_coeff,Sig_log_coeff, x_0, y_0,[num_lvl]*attr_num)[0]
        
        one_step_sol_val_list[i] = one_step_sol_val
        
        #Create a list storing all of the enumerated values of all question pairs evaluated
        #under two-step acquisition function. Return a dataframe containing enumeration if needed.
        enum_data = multlvl_enum_two_step(mu_0_rand,Sig_0_rand,mu_log_coeff,Sig_log_coeff,multlvl_prod_diff_list,num_lvl)[0]
        enum_df['Trial: %s'%(str(i))] = enum_data
        
        #Calculate the quantile values of one-step and two-step. The best performing solution will have a quant
        #value of 1.0 and the worst performing solution will have a quant value of 0.
        enum_data_len = len(enum_data)
        one_step_quantile = [1 for x in enum_data if x >= one_step_sol_val]
        quant_val_one_step[i] = sum(one_step_quantile)/enum_data_len
        print('iteration: ' +str(i))
        
    return [quant_val_one_step, enum_df, one_step_sol_val_list]


# In[32]:


#THIS NEEDS TO BE MOVED!!!
#Create Quantile Data with Boxplot for one-step performance in terms of two-step acquisition value.
#rng = np.random.default_rng(10000) 
#np.random.seed(10000)
#random.seed(10000)
#These are the parameters we will use
#attr_num_multlvl_OS_quant = 6
#num_lvl_OS_quant = 3
#prod_diff_list_multlvl_OS_quant = multlvl_product_diff_list(attr_num_multlvl_OS_quant,num_lvl_OS_quant) # TAKE A SAMPLE IF attr_num_OS_quant IS LARGE
#sample_size = 3**6
#prod_diff_list_onestep_quant = random.sample(prod_diff_list_OS_quant,sample_size)
#rep_num_OS_quant = 30
#scale_OS_quant = 2.0
#location_OS_quant = 1.0
#print(len(prod_diff_list_multlvl_OS_quant))


# In[33]:


#THIS NEEDS TO BE MOVED!!!
#Create Quantile Data with one Boxplot for one-step, as well as enumeration data of two-step function.
#multlvl_quant_data_OS = multlvl_one_step_sol_two_step_quantile(attr_num_multlvl_OS_quant,prod_diff_list_multlvl_OS_quant,rep_num_OS_quant,scale_OS_quant,location_OS_quant,
                                    #mu_log_coeff_6_mult_3,Sig_log_coeff_6_mult_3,num_lvl_OS_quant)


# In[34]:


#THIS NEEDS TO BE MOVED!!!
#labels_OS = ['one-step']
#plt.boxplot(multlvl_quant_data_OS[0], vert=True, patch_artist=True, labels=labels_OS)
#plt.ylabel("Percentile of Solution")

#plt.title("Quantiles of One-Step Solutions (Two-Step Value,Multi-Level): n=%s, scale = %s,Location = %s, Replication = %s, #levels = %s "%(str(attr_num_multlvl_OS_quant),
                                                                                                  #str(scale_OS_quant),
                                                                                              #str(location_OS_quant),
                                                                                                  #str(rep_num_OS_quant), str(num_lvl_OS_quant)))
#plt.show()


# In[35]:


#This function is used to generate an nxn matrix M(r) such that M_ij = r^|i-j| for 0<r<1. This matrix is called a 
#Kac-Murdock-Szego matrix.
def KMS_Matrix(n,r):
    #n: this is the number of rows and columns of the matrix
    #r: this is the coefficient given above that determines the value of the matrice's entries 
    
    
    M = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            M[i,j] = r**abs(i-j)
        
    return M


# In[36]:


#THIS NEEDS TO BE MOVED!!!
#print(1.25*KMS_Matrix(6,-0.5))
#print(np.linalg.det(1.25*KMS_Matrix(12,-0.5)))


# In[37]:


#This function is used to compare one-step and two-step in a sequential fashion. We start the experiment with
#some prior information (expectation and covariance matrix), and sample a number of individuals
#(partworth vectors) from this prior. We then run the two questionnaires (one-step and two-step) on all of the individuals,
#recording the MSE between the estimator and their true partworth vector after each question, and we also record their MSE
#and determinant information before starting the questionnaire.
def MSE_det_test(attr_num,mu_scale,Sig_scale,num_rep,num_beta,noise_par,mu_log_coeff,Sig_log_coeff, use_max_min_mse_beta = True,
                 rand_samp_par = True, epsilon_0 = 0.1,question_num=16,r=0.5,t = 100):
    #attr_num: number of attributes the products should have.
    
    #mu_scale: used to create the expectation of the normal distribution we sample true partworth vectors from.
    
    #Sig_Scale: used to create the covariance matrix of the normal distribution we sample true partworth vectors from.
    
    #num_rep: The number of times we run the sequential experiment on ONE individual partworth vector.
    
    #num_beta: The number of individuals (partworth vectors) we sample for our experiment.
    
    #noise_par: This is a parameter which is used to increase the weight of the individuals' true partworth when making decision
    #between x and y. The higher this weight is, the less effect the gumbel random variable has on user choice.
    
    #mu_log_coeff and Sig_log_coeff: linear coefficients that are used in the optimization problem
    
    #use_approx_two_step: Determines whether we use the approximate two-step method or the true two-step method via enumeration
    
    #use_max_min_mse_beta: Determines if we only experiment using the two betas with max and min MSE compared to the
    #intial prior expectation.
    
    #rand_samp_par: This boolean parameter decides whether we use randomly generated expectation and 
    #covariance matrix or if we use
    #constant vector and diagonal covariance matrix for the sampling of true partworth vectors.
    
    #epsilon_0: This is used in the two-step optimization problem to set the degree of orthogonality between
    #the first-stage and second-stage question pairs
    
    #question_num: number of questions in the questionnaire.
    
    #r: coefficient for the KMS matrix, which acts as the prior covariance matrix.
    
    #t: this is the time limit on the approximate two-step method.
    
    
    #Use rng to generate random. SHOULD rng BE INSIDE THE FUNCTION?
    #rng = np.random.default_rng(0)
    
    #Decide whether we use randomly generated prior.
    if rand_samp_par:
        pop_prior_mu = mu_scale*rng.uniform(size = attr_num)
        pop_prior_Sig = Sig_scale*sklearn.datasets.make_spd_matrix(attr_num)
    else:
        pop_prior_mu = np.array(attr_num*[mu_scale])
        pop_prior_Sig = Sig_scale*np.identity(attr_num)
    
    #Used to store partworth vectors
    true_betas = []

    #Sample from prior distribution to get partworths
    for i in range(0,num_beta):
        true_betas.append(rng.multivariate_normal(pop_prior_mu,pop_prior_Sig))
    
    #Start the experiment
    mu_start = pop_prior_mu #rng.uniform(low = -0.1, high = 0.1, size = attr_num)#mu_scale*np.array(attr_num*[0.01])
    Sig_start = pop_prior_Sig #Sig_scale*KMS_Matrix(attr_num,r)
    
    #If use_max_min_mse_beta is true, then we only use the betas with the smallest and largest MSE from the experiment.
    #MSE(true_beta/||true_beta||, initial_est/||initial_est||)
    if use_max_min_mse_beta:
        init_mse_values = num_beta*[0.0]
        for i in range(0,num_beta):
            init_mse_values[i] =  np.square(np.subtract(true_betas[i]/np.linalg.norm(true_betas[i],ord = 2),
                                                            mu_start/np.linalg.norm(mu_start, ord = 2))).mean()
        init_mse_max_index = np.argmax(np.array(init_mse_values))
        init_mse_min_index = np.argmin(np.array(init_mse_values))
        true_betas = [true_betas[init_mse_min_index],true_betas[init_mse_max_index]]
    
    beta_len = len(true_betas)
    
    #These will be used to store MSE and covariance matrix determinant values after each question is asked.
    one_step_det = [[[] for j in range(beta_len)] for i in range(question_num + 1)]
    one_step_mse = [[[] for j in range(beta_len)] for i in range(question_num + 1)]
    
    true_two_step_det = [[[] for j in range(beta_len)] for i in range(question_num + 1)]
    true_two_step_mse = [[[] for j in range(beta_len)] for i in range(question_num + 1)]
    
    appr_two_step_det = [[[] for j in range(beta_len)] for i in range(question_num + 1)]
    appr_two_step_mse = [[[] for j in range(beta_len)] for i in range(question_num + 1)]
    
    
    for b in range(0,beta_len):
        for i in range(0,num_rep):
            #Start each individual with the same prior information for both one-step and two-step methods. Store
            #starting determinant value and MSE
            one_step_mu = mu_start
            one_step_Sig = Sig_start
            true_two_step_mu = mu_start
            true_two_step_Sig = Sig_start
            appr_two_step_mu = mu_start
            appr_two_step_Sig = Sig_start
            #redundant
            one_step_det[0][b].append(np.linalg.det(Sig_start))
            true_two_step_det[0][b].append(np.linalg.det(Sig_start))
            appr_two_step_det[0][b].append(np.linalg.det(Sig_start))

            one_step_mse[0][b].append(np.square(np.subtract(true_betas[b]/np.linalg.norm(true_betas[b],ord = 2),
                                                            one_step_mu/np.linalg.norm(one_step_mu, ord = 2))).mean())
            true_two_step_mse[0][b].append(np.square(np.subtract(true_betas[b]/np.linalg.norm(true_betas[b],ord = 2),
                                                            true_two_step_mu/np.linalg.norm(true_two_step_mu, ord = 2))).mean())
            appr_two_step_mse[0][b].append(np.square(np.subtract(true_betas[b]/np.linalg.norm(true_betas[b],ord = 2),
                                                            appr_two_step_mu/np.linalg.norm(appr_two_step_mu, ord = 2))).mean())
            for j in range(1,question_num + 1):
                #get questions for one step
                [one_step_x,one_step_y] = g_opt(one_step_mu,one_step_Sig,mu_log_coeff,Sig_log_coeff)[1:]
                
                #get questions for approx two step
                [w_0,z_0,w_1,z_1] = two_stage_g_opt(appr_two_step_mu,appr_two_step_Sig,mu_log_coeff,Sig_log_coeff,epsilon_0,t_lim=t)
                two_step_appr_sol_val_0 = two_step_g_acq(appr_two_step_mu, appr_two_step_Sig,mu_log_coeff,Sig_log_coeff, w_0,z_0)
                two_step_appr_sol_val_1 = two_step_g_acq(appr_two_step_mu, appr_two_step_Sig,mu_log_coeff,Sig_log_coeff, w_1,z_1)
                if two_step_appr_sol_val_0 < two_step_appr_sol_val_1:
                    appr_two_step_x = w_0
                    appr_two_step_y = z_0
                else:
                    appr_two_step_x = w_1
                    appr_two_step_y = z_1
            
                #List of question pairs we will enumerate on for true two-step
                prod_list_enum_two_step = product_diff_list(attr_num)

                #Do the enumeration procedure
                enumerated_solution_list_two_step = enum_two_step(true_two_step_mu,true_two_step_Sig,mu_log_coeff,
                                                             Sig_log_coeff,prod_list_enum_two_step)
                    
                opt_sol_questions_two_step = enum_two_step_opt(enumerated_solution_list_two_step[0],
                                                          enumerated_solution_list_two_step[2])
                    
                #get the first stage products for true two-step
                true_two_step_x = np.array(opt_sol_questions_two_step[1])
                true_two_step_y = np.array(opt_sol_questions_two_step[2])
                    

                gum_x = rng.gumbel(0,1,1)
                gum_y = rng.gumbel(0,1,1)

                #These temp variables will be used in the choice model below in case the user prefers y over x.
                one_step_x_temp = one_step_x
                one_step_y_temp = one_step_y
                true_two_step_x_temp = true_two_step_x
                true_two_step_y_temp = true_two_step_y
                appr_two_step_x_temp = appr_two_step_x
                appr_two_step_y_temp = appr_two_step_y

                #See preference between two products
                #set signal to noise ratio
                if (noise_par*np.dot(true_betas[b],np.array(one_step_y)) + gum_y) >= (noise_par*np.dot(true_betas[b],np.array(one_step_x))
                                                                               + gum_x):
                    one_step_x = one_step_y_temp
                    one_step_y = one_step_x_temp
                if (noise_par*np.dot(true_betas[b],np.array(true_two_step_y)) + gum_y) >= (noise_par*np.dot(true_betas[b],np.array(true_two_step_x))
                                                                               + gum_x):
                    true_two_step_x = true_two_step_y_temp
                    true_two_step_y = true_two_step_x_temp
                if (noise_par*np.dot(true_betas[b],np.array(appr_two_step_y)) + gum_y) >= (noise_par*np.dot(true_betas[b],np.array(appr_two_step_x))
                                                                               + gum_x):
                    appr_two_step_x = appr_two_step_y_temp
                    appr_two_step_y = appr_two_step_x_temp
            

                #Perform moment matching after choice is made.
                [one_step_mu, one_step_Sig] = moment_matching_update(one_step_x,one_step_y,one_step_mu,one_step_Sig)
                [true_two_step_mu, true_two_step_Sig] = moment_matching_update(true_two_step_x,true_two_step_y,true_two_step_mu,true_two_step_Sig)
                [appr_two_step_mu, appr_two_step_Sig] = moment_matching_update(appr_two_step_x,appr_two_step_y,appr_two_step_mu,appr_two_step_Sig)


                #Compute determinant and MSE after question j, save information in a list corresponding to question j.
                #This list will hold all of the MSE and determinant information for all individuals for question j.
                one_step_det[j][b].append(np.linalg.det(one_step_Sig))
                true_two_step_det[j][b].append(np.linalg.det(true_two_step_Sig))
                appr_two_step_det[j][b].append(np.linalg.det(appr_two_step_Sig))
                
                one_step_mse[j][b].append(np.square(np.subtract(true_betas[b]/np.linalg.norm(true_betas[b],ord = 2),
                                                            one_step_mu/np.linalg.norm(one_step_mu, ord = 2))).mean())
                true_two_step_mse[j][b].append(np.square(np.subtract(true_betas[b]/np.linalg.norm(true_betas[b],ord = 2),
                                                            true_two_step_mu/np.linalg.norm(true_two_step_mu, ord = 2))).mean())
                appr_two_step_mse[j][b].append(np.square(np.subtract(true_betas[b]/np.linalg.norm(true_betas[b],ord = 2),
                                                            appr_two_step_mu/np.linalg.norm(appr_two_step_mu, ord = 2))).mean())
            
    return [one_step_det,one_step_mse,true_two_step_det,true_two_step_mse,appr_two_step_det,appr_two_step_mse]


# In[38]:


#THIS NEEDS TO BE MOVED!!!
#Set up parameters for MSE_det_test
#rng = np.random.default_rng(100) 
#np.random.seed(100)
#random.seed(100)
#attr_num_test = 6
#mu_scale_test = 0.5 
#Sig_scale_test = 1.0
#num_rep_test = 50 #100
#num_beta_test = 100
#noise_par_test = 1.0


# In[39]:


#THIS NEEDS TO BE MOVED!!!
#Gather MSE and determinant information for two betas (max and min MSE) 
#MSE_det_info = MSE_det_test(attr_num_test,mu_scale_test,Sig_scale_test,num_rep_test,num_beta_test,noise_par_test,mu_log_coeff_6,
                           #Sig_log_coeff_6,use_max_min_mse_beta = True,rand_samp_par = False, question_num = 16,t=100)


# In[40]:


#THIS NEEDS TO BE MOVED!!!
#print(MSE_det_info[1])
#print(MSE_det_info[5])
#print(MSE_det_info[3])


# In[41]:


#THIS NEEDS TO BE MOVED!!!
#Place data in a csv
#zipped_MSE_det = list(zip(MSE_det_info[0], MSE_det_info[1],MSE_det_info[2],MSE_det_info[3]))
#df_MSE_det = pd.DataFrame(zipped_MSE_det,columns = ['one-step det','one-step MSE','two-step det', 'two-step MSE'])
#df_MSE_det.to_csv(r'C:\Users\wsfishe\Desktop\MSE_det_test_July13_%s_%s_%s_%s_%s_Truetwostep.csv'%(str(attr_num_test),str(num_rep_test),str(num_beta_test),
                                                                             #str(Sig_scale_test),str(mu_scale_test)),index=False,header=True)


# In[42]:


#THIS NEEDS TO BE MOVED!!!
#format data to use in sns.lineplot. Use this when there are multiple betas but only one replication for each
#beta. Plot MSE
#question_num_test = 4
#x_axis_one_mse = []
#for i in range(question_num_test + 1):
    #for j in range (0,num_beta_test):
        #x_axis_one_mse.append(i)

#y_axis_one_mse = []
    
#for j in range(question_num_test + 1):
    #for k in MSE_det_info[1][j]:
        #y_axis_one_mse.append(k)
        
#x_axis_two_mse = []
#for i in range(question_num_test + 1):
    #for j in range (0,num_beta_test):
        #x_axis_two_mse.append(i)

#y_axis_two_mse = []
    
#for j in range(question_num_test + 1):
    #for k in MSE_det_info[3][j]:
        #y_axis_two_mse.append(k)
    
#fig = sns.lineplot(x = x_axis_one_mse, y = y_axis_one_mse).set(title='MSE Sequential Comparison:July12')
#sns.lineplot(x = x_axis_two_mse, y = y_axis_two_mse).set( xlabel = "Question Number", ylabel = "MSE After Normalizing")
#plt.legend(labels=["One-Step","Two-Step"])


# In[43]:


#THIS NEEDS TO BE MOVED!!!
#format data to use in sns.lineplot. Use this when there are max and min betas but having multiple replications each
#. Plot MSE (MAX)

#question_num_max_min = 2

#max_x_axis_one_mse = []
#for i in range(question_num_max_min + 1):
    #for j in range(0,num_rep_test):
        #max_x_axis_one_mse.append(i)

#max_y_axis_one_mse = []
    
#for i in range(question_num_max_min + 1):
    #for j in MSE_det_info[1][i][1]:
        #max_y_axis_one_mse.append(j)
        
#max_x_axis_true_two_mse = []
#for i in range(question_num_max_min + 1):
    #for j in range(0,num_rep_test):
        #max_x_axis_true_two_mse.append(i)

#max_y_axis_true_two_mse = []
    
#for i in range(question_num_max_min + 1):
    #for j in MSE_det_info[3][i][1]:
        #max_y_axis_true_two_mse.append(j)
        
#max_x_axis_appr_two_mse = []
#for i in range(question_num_max_min + 1):
    #for j in range(0,num_rep_test):
        #max_x_axis_appr_two_mse.append(i)

#max_y_axis_appr_two_mse = []
    
#for i in range(question_num_max_min + 1):
    #for j in MSE_det_info[5][i][1]:
        #max_y_axis_appr_two_mse.append(j)
        
#fig = sns.lineplot(x = max_x_axis_one_mse, y = max_y_axis_one_mse).set(title='MSE Sequential Comparison: Baseline Partworth with Max MSE')
#sns.lineplot(x = max_x_axis_true_two_mse, y = max_y_axis_true_two_mse).set( xlabel = "Question Number", ylabel = "MSE After Normalizing")
#sns.lineplot(x = max_x_axis_appr_two_mse, y = max_y_axis_appr_two_mse)
#plt.legend(labels=["One-Step","True Two-Step","Approximate Two-Step"])


# In[44]:


#THIS NEEDS TO BE MOVED!!!
#format data to use in sns.lineplot. Use this when there are max and min betas but having multiple replications each
#. Plot MSE (MIN)

#question_num_max_min = 16

#min_x_axis_one_mse = []
#for i in range(question_num_max_min + 1):
    #for j in range(0,num_rep_test):
        #min_x_axis_one_mse.append(i)

#min_y_axis_one_mse = []
    
#for i in range(question_num_max_min + 1):
    #for j in MSE_det_info[1][i][0]:
        #min_y_axis_one_mse.append(j)
        
#min_x_axis_two_mse = []
#for i in range(question_num_max_min + 1):
    #for j in range(0,num_rep_test):
        #min_x_axis_two_mse.append(i)

#min_y_axis_two_mse = []
    
#for i in range(question_num_max_min + 1):
    #for j in MSE_det_info[3][i][0]:
        #min_y_axis_two_mse.append(j)
        
#fig = sns.lineplot(x = min_x_axis_one_mse, y = min_y_axis_one_mse).set(title='MSE Sequential Comparison: Baseline Partworth with Min MSE')
#sns.lineplot(x = min_x_axis_two_mse, y = min_y_axis_two_mse).set( xlabel = "Question Number", ylabel = "MSE After Normalizing")
#plt.legend(labels=["One-Step","True Two-Step"])


# In[45]:


#THIS NEEDS TO BE MOVED!!!
#format data to use in sns.lineplot. Use this when there are multiple betas but only one replication for each
#beta.Plot determinant
#question_num_test = 4
#x_axis_one_det = []
#for i in range(question_num_test + 1):
    #for j in range (0,num_beta_test):
        #x_axis_one_det.append(i)

#y_axis_one_det = []
    
#for j in range(question_num_test + 1):
    #for k in MSE_det_info[0][j]:
        #y_axis_one_det.append(k)
        
#x_axis_two_det = []
#for i in range(question_num_test + 1):
    #for j in range (0,num_beta_test):
        #x_axis_two_det.append(i)

#y_axis_two_det = []
    
#for j in range(question_num_test + 1):
    #for k in MSE_det_info[2][j]:
        #y_axis_two_det.append(k)

#print(len(x_axis_one_det))
#print(len(y_axis_one_det))
#fig = sns.lineplot(x = x_axis_one_det, y = y_axis_one_det).set(title='Determinant Sequential Comparison: KMS Prior')
#sns.lineplot(x = x_axis_two_det, y = y_axis_two_det).set( xlabel = "Question Number", ylabel = "Determinant")
#plt.legend(labels=["One-Step","Two-Step"])


# In[46]:


#THIS NEEDS TO BE MOVED!!!
#format data to use in sns.lineplot. Use this when there are max and min betas but having multiple replications each
#. Plot DET (MAX)

#question_num_max_min = 1

#max_x_axis_one_det = []
#for i in range(0,question_num_max_min + 1):
    #for j in range(0,num_rep_test):
        #max_x_axis_one_det.append(i)

#max_y_axis_one_det = []
    
#for i in range(0,question_num_max_min + 1):
    #for j in MSE_det_info[0][i][1]:
        #max_y_axis_one_det.append(j)
        
#max_x_axis_true_two_det = []
#for i in range(0,question_num_max_min + 1):
    #for j in range(0,num_rep_test):
        #max_x_axis_true_two_det.append(i)

#max_y_axis_true_two_det = []
    
#for i in range(0,question_num_max_min + 1):
    #for j in MSE_det_info[2][i][1]:
        #max_y_axis_true_two_det.append(j)
        
#max_x_axis_appr_two_det = []
#for i in range(0,question_num_max_min + 1):
    #for j in range(0,num_rep_test):
        #max_x_axis_appr_two_det.append(i)

#max_y_axis_appr_two_det = []
    
#for i in range(0,question_num_max_min + 1):
    #for j in MSE_det_info[4][i][1]:
        #max_y_axis_appr_two_det.append(j)
        
#fig = sns.lineplot(x = max_x_axis_one_det, y = max_y_axis_one_det).set(title='Determinant Sequential Comparison: Baseline Partworth with Max MSE')
#sns.lineplot(x = max_x_axis_true_two_det, y = max_y_axis_true_two_det).set( xlabel = "Question Number", ylabel = "Determinant")
#sns.lineplot(x = max_x_axis_appr_two_det, y = max_y_axis_appr_two_det)
#plt.legend(labels=["One-Step","True Two-Step","Approximate Two-Step"])


# In[47]:


#THIS NEEDS TO BE MOVED!!!
#format data to use in sns.lineplot. Use this when there are max and min betas but having multiple replications each
#. Plot DET (MIN)

#question_num_max_min = 16

#min_x_axis_one_det = []
#for i in range(8,question_num_max_min + 1):
    #for j in range(0,num_rep_test):
        #min_x_axis_one_det.append(i)

#min_y_axis_one_det = []
    
#for i in range(8,question_num_max_min + 1):
    #for j in MSE_det_info[0][i][0]:
        #min_y_axis_one_det.append(j)
        
#min_x_axis_two_det = []
#for i in range(8,question_num_max_min + 1):
    #for j in range(0,num_rep_test):
        #min_x_axis_two_det.append(i)

#min_y_axis_two_det = []
    
#for i in range(8,question_num_max_min + 1):
    #for j in MSE_det_info[2][i][0]:
        #min_y_axis_two_det.append(j)
        
#fig = sns.lineplot(x = min_x_axis_one_det, y = min_y_axis_one_det).set(title='Determinant Sequential Comparison: Baseline Partworth with Min MSE')
#sns.lineplot(x = min_x_axis_two_det, y = min_y_axis_two_det).set( xlabel = "Question Number", ylabel = "Determinant")
#plt.legend(labels=["One-Step","True Two-Step"])


# In[48]:


#MOVE IT!!!
#from scipy import stats


# In[49]:


#THIS NEEDS TO BE MOVED!!!
#Look at the determinant values after question 4
#one_step_det_q4 = MSE_det_info[0][4]
#two_step_det_q4 = MSE_det_info[2][4]
#Do a paired t-test between the two sets of determinant data for question 4
#tStat_detq4, pValue_detq4 =  stats.ttest_rel(one_step_det_q4, two_step_det_q4)
#print("P-Value:{0} T-Statistic:{1}".format(pValue_detq4,tStat_detq4)) #print the P-Value and the T-Statistic
#Print out the average
#print("One-Step Determinant Average: " +str(sum(one_step_det_q4)/len(one_step_det_q4)))
#print("Approximate Two-Step Determinant Average: " +str(sum(two_step_det_q4)/len(two_step_det_q4)))


# In[50]:


#THIS NEEDS TO BE MOVED!!!
#Look at the determinant values after question 3
#one_step_det_q3 = MSE_det_info[0][3]
#two_step_det_q3 = MSE_det_info[2][3]
#Do a paired t-test between the two sets of determinant data for question 3
#tStat_detq3, pValue_detq3 =  stats.ttest_rel(one_step_det_q3, two_step_det_q3)
#print("P-Value:{0} T-Statistic:{1}".format(pValue_detq3,tStat_detq3)) #print the P-Value and the T-Statistic
#Print out the average
#print("One-Step Determinant Average: " + str(sum(one_step_det_q3)/len(one_step_det_q3)))
#print("Approximate Two-Step Determinant Average: " + str(sum(two_step_det_q3)/len(two_step_det_q3)))


# In[51]:


#THIS NEEDS TO BE MOVED!!!
#Look at the determinant values after question 2
#one_step_det_q2 = MSE_det_info[0][2]
#two_step_det_q2 = MSE_det_info[2][2]
#Do a paired t-test between the two sets of determinant data for question 2
#tStat_detq2, pValue_detq2 =  stats.ttest_rel(one_step_det_q2, two_step_det_q2)
#print("P-Value:{0} T-Statistic:{1}".format(pValue_detq2,tStat_detq2)) #print the P-Value and the T-Statistic
#Print out the average
#print("One-Step Determinant Average: "+ str(sum(one_step_det_q2)/len(one_step_det_q2)))
#print("Approximate Two-Step Determinant Average: " + str(sum(two_step_det_q2)/len(two_step_det_q2)))


# In[52]:


#THIS NEEDS TO BE MOVED!!!
#Look at the determinant values after question 1
#one_step_det_q1 = MSE_det_info[0][1]
#two_step_det_q1 = MSE_det_info[2][1]

#Do a paired t-test between the two sets of determinant data for question 1
#tStat_detq1, pValue_detq1 =  stats.ttest_rel(one_step_det_q1, two_step_det_q1)
#print("P-Value:{0} T-Statistic:{1}".format(pValue_detq1,tStat_detq1)) #print the P-Value and the T-Statistic
#Print out the average
#print("One-Step Determinant Average: " + str(sum(one_step_det_q1)/len(one_step_det_q1)))
#print("Approximate Two-Step Determinant Average: " + str(sum(two_step_det_q1)/len(two_step_det_q1)))


# In[53]:


#THIS NEEDS TO BE MOVED!!!
#This is used to reformat the MSE and determinant data in a nicer format and save it as a csv.
#df_MSE_det_reformatted = pd.DataFrame()
#questions_and_start = 17
#max_or_min = ['MIN','MAX']
#for i in range(2):
#for j in range(questions_and_start):
        #df_MSE_det_reformatted['Question %s, Baseline Partworth %s, One-Step Det' %(str(j),max_or_min[1])] = MSE_det_info[0][j][1]
        #df_MSE_det_reformatted['Question %s, Baseline Partworth %s, One-Step MSE' %(str(j),max_or_min[1])] = MSE_det_info[1][j][1]
        #df_MSE_det_reformatted['Question %s, Baseline Partworth %s, True Two-Step Det' %(str(j),max_or_min[1])] = MSE_det_info[2][j][1]
        #df_MSE_det_reformatted['Question %s, Baseline Partworth %s, True Two-Step MSE' %(str(j),max_or_min[1])] = MSE_det_info[3][j][1]
        #df_MSE_det_reformatted['Question %s, Baseline Partworth %s, Appr Two-Step Det' %(str(j),max_or_min[1])] = MSE_det_info[4][j][1]
        #df_MSE_det_reformatted['Question %s, Baseline Partworth %s, Appr Two-Step MSE' %(str(j),max_or_min[1])] = MSE_det_info[5][j][1]


#df_MSE_det_reformatted.to_csv(r'C:\Users\wsfishe\Desktop\MSE_det_test_reformat_July29_%s_%s_%s_%s.csv'%(str(attr_num_test),str(num_rep_test),
                                                                             #str(Sig_scale_test),str(mu_scale_test)),index=False,header=True)


# In[54]:


#THIS NEEDS TO BE MOVED!!!
#Look at the 16th question in terms of Normalized MSE for Max MSE case

#one_step_MSE_maxcase_q16 = MSE_det_info[1][16][1]
#two_step_MSE_maxcase_q16 = MSE_det_info[3][16][1]
#Do a paired t-test between the two sets of MSE data for question 16 in the Max MSE case
#tStat_MSEq16maxcase, pValue_MSEq16maxcase =  stats.ttest_rel(one_step_MSE_maxcase_q16 , two_step_MSE_maxcase_q16)
#print("P-Value:{0} T-Statistic:{1}".format(pValue_MSEq16maxcase,tStat_MSEq16maxcase)) #print the P-Value and the T-Statistic
#Print out the average
#print("Max Case One-Step MSE Average: " + str(sum(one_step_MSE_maxcase_q16)/len(one_step_MSE_maxcase_q16)))
#print("Max Case True Two-Step MSE Average: " + str(sum(two_step_MSE_maxcase_q16)/len(two_step_MSE_maxcase_q16)))


# In[55]:


#THIS NEEDS TO BE MOVED!!!
#Look at the 12th question in terms of Normalized MSE for Max MSE case

#one_step_MSE_maxcase_q12 = MSE_det_info[1][12][1]
#two_step_MSE_maxcase_q12 = MSE_det_info[3][12][1]
#Do a paired t-test between the two sets of MSE data for question 16 in the Max MSE case
#tStat_MSEq12maxcase, pValue_MSEq12maxcase =  stats.ttest_rel(one_step_MSE_maxcase_q12 , two_step_MSE_maxcase_q12)
#print("P-Value:{0} T-Statistic:{1}".format(pValue_MSEq12maxcase,tStat_MSEq12maxcase)) #print the P-Value and the T-Statistic
#Print out the average
#print("Max Case One-Step MSE Average: " + str(sum(one_step_MSE_maxcase_q12)/len(one_step_MSE_maxcase_q12)))
#print("Max Case True Two-Step MSE Average: " + str(sum(two_step_MSE_maxcase_q12)/len(two_step_MSE_maxcase_q12)))


# In[56]:


#!!!THIS FUNCTION IS A REWRITE OF MSE_det_test!!!#
#This function is used to compare different methods in a sequential fashion using normalized MSE, determinant, 
#hitrate, and MSE as a measurement 
#of quality. In this function, we only use one acquisition method at a time (compare with MSE_det_test).

def new_sequential_experiment(init_mu, init_Sig, true_partworths, rep_per_partworth, num_questions,look_ahead_horizon, mu_log_coeff,
                                   Sig_log_coeff,noise_par, question_list, Method = 0,
                                   batch_size = 0, MC_budget = 0, include_one_step = False,penalty = 100,rel_gap = 0.0001):
    #init_mu: This is the initial estimate on the partworths
    #init_Sig: This is the initial covariance matrix on the partworths
    #true_partworths: These are used to make selection in the product selection stage (a list/set of partworths)
    #rep_per_partworth: This is the number of times we want to conduct a questionnaire on each partworth
    #num_questions: Length of the questionnaire
    #look_ahead_horizon: This is the number of stages that we look ahead in a rollout method.
    #mu_log_coeff; Sig_log_coeff: These are the coefficients in the optimization model for the linearized objective function
    #noise_par: This is a parameter which is used to increase the weight of the individuals' true partworth when making decision
    #hitrate_question_list: This is a list of questions which will be used to calculate the hitrate, which is the
    #proportion of times that an estimated partworth matches the product selection of a true underlying partworth.
    #between x and y. The higher this weight is, the less effect the gumbel random variable has on user choice.
    #Method: 0 - One step look ahead
    #        1 - Rollout with batch design   <------ CAN ADD MORE METHODS IF NEEDED
    #        2 - Two step look ahead via enumeration
    #        3 - Rollout using coordinate exchange
    #batch_size: Used to select batch size if we use rollout
    #        OPTIONS:
    #        batch_size = n (n is a non-negative integer): This will make it so that the batch size is constant and equal to n for every rollout iteration
    #        batch_size = -1: This will make it so that the batch size is equal to the look-ahead horizon
    #MC_budget: Used to select Monte Carlo budget if we use rollout.
    #include_one_step: This determines whether we want to include the one-step optimal question within our batch. This can
    #help ensure that rollout performs at least as well as one-step look ahead. Default value is False.
    #penalty_term: This is used to set the penalty level for orthogonality in the orthogonal batch design optimization problem. A higher penalty
    #term will lead to a more Sigma_orthogonal design, while a lower penalty term will lead to less Sigma_orthogonality in the design
    #rel_gap: This is used in rollout with coordinate exchange. This value measures the improvement of a question's rollout
    #value over the current question's rollout value. If the improvement is greater than the relative gap, we update the current
    #question
    
    
    #Construct lists for storing normalized MSE, determinant, hitrate, and MSE information. 
    #For each of the baseline partworths, we create a list holding
    #(num_questions) lists, where each of the (num_questions) lists holds the MSE and determinant values for all replications
    #after the first, second,...,nth question
    
    num_true_partworth = len(true_partworths) 
    
    hitrate_total_num_of_questions = len(question_list)
    
    
    #Construct the expected preferred product selection for each true partworth. This will be used when we calculate the probability
    #of correct selection
    x_trueselection = [[] for u in range(num_true_partworth)]
    attributes_num = len(init_mu)
    for u in range(num_true_partworth):
        x_trueselection[u] = np.array([1.0 if partworth_component>=0.0 else 0.0 for partworth_component in 
                                           true_partworths[u]])
    
    #Set up lists to hold normalized MSE, determinant, MSE, and hitrate information. For now, we will only calculate hitrate at the end
    #of the questionnaire.  Also save the mu vectors. 
    
    MSE_normalized = [[[] for j in range(num_questions)] for u in range(num_true_partworth)]
    
    DET = [[[] for j in range(num_questions)] for u in range(num_true_partworth)]
    
    HITRATE = [[] for u in range(num_true_partworth)]
    
    MSE = [[[] for j in range(num_questions)] for u in range(num_true_partworth)]
    
    PROB_CORRECT_SEL = []
    
    MU = [[[] for j in range(num_questions)] for u in range(num_true_partworth)]
    
    for u in range(num_true_partworth):
        #create a variable which will store the number of correct selections.
        correct_sel = 0
        print('true_partworth:' + str(true_partworths[u]))
        for i in range(rep_per_partworth):
            #Instantiate mu and Sig with the initial parameters init_mu and init_Sig. These act as prior parameters for all
            #the partworths
            mu = init_mu
            Sig = init_Sig
            print('init_mu: ' + str(init_mu))
            print('init_Sig: ' + str(init_Sig))
            #print('replication:'+ str(i))
            for j in range(num_questions):
                if Method == 0:
                    #get optimal question for one step
                    [x,y] = g_opt(mu,Sig,mu_log_coeff,Sig_log_coeff)[1:]
                    
                if Method == 1:
                    #get optimal question for rollout over batch design
                    #Use a rollout length equal to look_ahead_horizon - 1, until we are within look_ahead_horizon of the
                    #budget
                    
                    if j < num_questions - look_ahead_horizon:
                        rollout_len = look_ahead_horizon - 1
                    else:
                        rollout_len = num_questions - j - 1
                    
                    if batch_size == -1:
                        #When batch_size is -1, we use the rollout length + 1 as the batch size.
                        [x,y] = rollout_with_batch_design_acquisition(mu,Sig,mu_log_coeff,Sig_log_coeff,rollout_len+1,
                                                                 rollout_len,MC_budget,include_one_step,penalty_term = penalty)
                    else:
                        [x,y] = rollout_with_batch_design_acquisition(mu,Sig,mu_log_coeff,Sig_log_coeff,batch_size,
                                                                 rollout_len,MC_budget,include_one_step,penalty_term = penalty)

                    
                if Method == 2:
                    #Do the enumeration procedure for two-step look ahead
                    if j < num_questions-1:
                        enumerated_solution_list_two_step = enum_two_step(mu,Sig,mu_log_coeff,
                                                             Sig_log_coeff,question_list)
                        opt_sol_questions_two_step = enum_two_step_opt(enumerated_solution_list_two_step[0],
                                                          enumerated_solution_list_two_step[2])
                    
                        #get the first stage products for true two-step
                        x = np.array(opt_sol_questions_two_step[1])
                        y = np.array(opt_sol_questions_two_step[2])
                    #Use one-step for the last question.
                    else:
                        [x,y] = g_opt(mu,Sig,mu_log_coeff,Sig_log_coeff)[1:]
                    
                if Method == 3:
                    #Do rollout with coordinate exchange
                    if j < num_questions - look_ahead_horizon:
                        rollout_len = look_ahead_horizon - 1
                    else:
                        rollout_len = num_questions - j - 1
                        
                    [x,y] = coordinate_exchange_acq(mu,Sig,mu_log_coeff,Sig_log_coeff,batch_size,rollout_len,
                                                   MC_budget,rel_gap,include_batch = False,include_one_step = True)
                    
                #Instantiate gumbel random variables which are used in the product choice selection process.
                gum_x = rng.gumbel(0,1,1)
                gum_y = rng.gumbel(0,1,1)
                    
                #These temp variables will be used in the choice model below in case the user prefers y over x.
                x_temp = x
                y_temp = y
                    
                #See preference between two products
                #set signal to noise ratio
                if (noise_par*np.dot(true_partworths[u],np.array(y)) + gum_y) >= (noise_par*np.dot(true_partworths[u],np.array(x))
                                                                               + gum_x):
                    x = y_temp
                    y = x_temp
                
                #Perform moment matching after choice is made.
                print('Sig before:' + str(Sig))
                print('mu before: ' + str(mu))
                [mu, Sig] = moment_matching_update(x,y,mu,Sig)
                print('Sig after:' + str(Sig))
                print('mu after: ' + str(mu))
                
                #add mu to the list of mu vectors
                MU[u][j].append(mu)
                #Add the normalized MSE between the true partworth and estimator at question j to a list, and add the determinant of
                #the covariance matrix at question j into a list. Also add the regular MSE
                MSE_normalized[u][j].append(np.square(np.subtract(true_partworths[u]/np.linalg.norm(true_partworths[u],ord = 2),
                                                            mu/np.linalg.norm(mu, ord = 2))).mean())
                DET[u][j].append(np.linalg.det(Sig))
                
                MSE[u][j].append(np.square(np.subtract(true_partworths[u],
                                                            mu)).mean())
                
            #Calculate hitrate for this replication. Given a set of questions of length K, we compare how well the final 
            #estimator performs in terms of correctly selecting the preferred profile for each question. The estimator
            #makes a correct selection if its selection matches that of the true partworth (selection is absent of gumble noise)
            hits = 0
            for q in question_list:
                if np.dot(true_partworths[u],q)*np.dot(mu,q)>=0:
                    hits = hits + 1
            HITRATE[u].append(hits/hitrate_total_num_of_questions)
            
            #Calculate whether we have correct selection in this replication for estimator mu
            x_mu = np.array([1.0 if mu_component>=0.0 else 0.0 for mu_component in 
                                           mu])
            print('x_mu: ' + str(x_mu))
            print('x_true_selection: ' + str(x_trueselection[u]))
            if np.dot(x_trueselection[u]-x_mu,x_trueselection[u]-x_mu) == 0.0:
                correct_sel = correct_sel + 1
        
        #Calculate the probability of correct selection for the current true partworth
        PROB_CORRECT_SEL.append(correct_sel/rep_per_partworth)
                
        
    return[MSE_normalized,DET,HITRATE,MSE,PROB_CORRECT_SEL,true_partworths,MU]


# In[57]:


#NEEDS TO BE MOVED!!!
# 09/13/2022 Set up parameters for new_mse_det_experiment

#Set up random number seed
#rng = np.random.default_rng(100) 
#np.random.seed(100)
#random.seed(100)

#declare initial parameters mu and Sig, as well as the number of partworths to sample from N(mu,Sig).
#num_attr_nmde = 6
#samp_num_partworths = 100
#mu_initial = np.array([1.0,1.0,1.0,1.0,1.0,1.0])#np.array(num_attr_nmde*[0.1])
#Sig_initial = np.identity(num_attr_nmde)#KMS_Matrix(num_attr_nmde,0.5)#np.diag([4.0,0.1,0.1,0.1,0.1,4.0])

#partworths_nmde = []

#for i in range(0,samp_num_partworths):
        #partworths_nmde.append(rng.multivariate_normal(mu_initial,Sig_initial))

#Use only the two partworths with the max and min MSE compared to the initial mu
#init_mse_values = samp_num_partworths*[0.0]
#for i in range(0,samp_num_partworths):
    #init_mse_values[i] =  np.square(np.subtract(partworths_nmde[i]/np.linalg.norm(partworths_nmde[i],ord = 2),
                                                            #mu_initial/np.linalg.norm(mu_initial, ord = 2))).mean()
#init_mse_max_index = np.argmax(np.array(init_mse_values))
#init_mse_min_index = np.argmin(np.array(init_mse_values))
#partworths_nmde = [partworths_nmde[init_mse_min_index],partworths_nmde[init_mse_max_index]]

#Declare number of repetitions per partworth and the number of questions in the questionnaire.
#rep_per_partworth_nmde = 1
#num_questions_nmde = 4
#look_ahead_hor_nmde = 4
#noise_par_nmde = 1.0

#Construct the list of questions for the hitrate procedure.
#nmde_question_list = product_diff_list(num_attr_nmde)

#Here we decide to use rollout
#Method_nmde = 1
#batch_size_nmde = 4
#MC_budget_nmde = 40
#print(Sig_initial)


# In[58]:


#NEEDS TO BE MOVED!!!
#Set up random number seed
#rng = np.random.default_rng(100) 
#np.random.seed(100)
#random.seed(100)

#[Norm_MSE_roll,DET_roll,HITRATE_roll,MSE_roll,Prob_sel_roll] = new_sequential_experiment(mu_initial,Sig_initial,partworths_nmde,rep_per_partworth_nmde,num_questions_nmde,
                                           #look_ahead_hor_nmde,mu_log_coeff_6,Sig_log_coeff_6,noise_par_nmde,nmde_question_list,Method = 3,
                                            #batch_size = batch_size_nmde,MC_budget=MC_budget_nmde, include_one_step = True,penalty = 1.0)


# In[59]:


#NEEDS TO BE MOVED!!!
#Set up random number seed
#rng = np.random.default_rng(100) 
#np.random.seed(100)
#random.seed(100)

#[Norm_MSE_onestep,DET_onestep,HITRATE_onestep,MSE_onestep,Prob_sel_onestep] = new_mse_det_hitrate_experiment(mu_initial,Sig_initial,partworths_nmde,rep_per_partworth_nmde,num_questions_nmde,
                                           #mu_log_coeff_6,Sig_log_coeff_6,noise_par_nmde,nmde_question_list,Method = 0,
                                            #batch_size = 0,MC_budget = 0,penalty = 1.0)


# In[60]:


#MOVE IT!!!
#print(MSE_roll)
#print(DET_onestep)
#print(DET_roll)
#print(Prob_sel_roll)
#print(Prob_sel_onestep)


# In[61]:


#THIS NEEDS TO BE MOVED!!!
#Plotting MSE data for Max MSE case. Comparing one-step and rollout
#x_axis_one_step_max_mse = []

#for i in range(1,num_questions_nmde+1):
    #for j in range (0,rep_per_partworth_nmde):
        #x_axis_one_step_max_mse.append(i)
        
#y_axis_one_step_max_mse = []

#for i in range(0,num_questions_nmde):
    #for mse_data in MSE_onestep[1][i]:
        #y_axis_one_step_max_mse.append(mse_data)
        
#x_axis_rollout_max_mse = []

#for i in range(1,num_questions_nmde+1):
    #for j in range (0,rep_per_partworth_nmde):
        #x_axis_rollout_max_mse.append(i)
        
#y_axis_rollout_max_mse = []

#for i in range(0,num_questions_nmde):
    #for mse_data in MSE_roll[1][i]:
        #y_axis_rollout_max_mse.append(mse_data)
        
#fig = sns.lineplot(x = x_axis_one_step_max_mse, y = y_axis_one_step_max_mse).set(title='(Max)MSE Sequential Comparison: One-step vs Rollout')
#sns.lineplot(x = x_axis_rollout_max_mse, y = y_axis_rollout_max_mse).set( xlabel = "Question Number", ylabel = "MSE After Normalizing")
#plt.legend(labels=["One-Step","Rollout"])
#plt.xticks([1,2,3,4])


# In[62]:


#THIS NEEDS TO BE MOVED!!!
#Plotting MSE data for Min MSE case. Comparing one-step and rollout
#x_axis_one_step_min_mse = []

#for i in range(1,num_questions_nmde+1):
    #for j in range (0,rep_per_partworth_nmde):
        #x_axis_one_step_min_mse.append(i)
        
#y_axis_one_step_min_mse = []

#for i in range(0,num_questions_nmde):
    #for mse_data in MSE_onestep[0][i]:
        #y_axis_one_step_min_mse.append(mse_data)
        
#x_axis_rollout_min_mse = []

#for i in range(1,num_questions_nmde+1):
    #for j in range (0,rep_per_partworth_nmde):
        #x_axis_rollout_min_mse.append(i)
        
#y_axis_rollout_min_mse = []

#for i in range(0,num_questions_nmde):
    #for mse_data in MSE_roll[0][i]:
        #y_axis_rollout_min_mse.append(mse_data)
        
#fig = sns.lineplot(x = x_axis_one_step_min_mse, y = y_axis_one_step_min_mse).set(title='(Min)MSE Sequential Comparison: One-step vs Rollout')
#sns.lineplot(x = x_axis_rollout_min_mse, y = y_axis_rollout_min_mse).set( xlabel = "Question Number", ylabel = "MSE After Normalizing")
#plt.legend(labels=["One-Step","Rollout"])
#plt.xticks([1,2,3,4])


# In[63]:


#THIS NEEDS TO BE MOVED!!!
#Plotting determinant data for Max MSE case. Comparing one-step and rollout
#x_axis_one_step_max_det = []

#for i in range(1,num_questions_nmde+1):
    #for j in range (0,rep_per_partworth_nmde):
        #x_axis_one_step_max_det.append(i)
        
#y_axis_one_step_max_det = []

#for i in range(0,num_questions_nmde):
    #for det_data in DET_onestep[1][i]:
        #y_axis_one_step_max_det.append(det_data)
        
#x_axis_rollout_max_det = []

#for i in range(1,num_questions_nmde+1):
    #for j in range (0,rep_per_partworth_nmde):
        #x_axis_rollout_max_det.append(i)
        
#y_axis_rollout_max_det = []

#for i in range(0,num_questions_nmde):
    #for det_data in DET_roll[1][i]:
        #y_axis_rollout_max_det.append(det_data)
        
#fig = sns.lineplot(x = x_axis_one_step_max_det, y = y_axis_one_step_max_det).set(title='(Max)Determinant Sequential Comparison: One-step vs Rollout')
#sns.lineplot(x = x_axis_rollout_max_det, y = y_axis_rollout_max_det).set( xlabel = "Question Number", ylabel = "Determinant")
#plt.legend(labels=["One-Step","Rollout"])
#plt.xticks([1,2,3,4])


# In[64]:


#THIS NEEDS TO BE MOVED!!!
#Plotting determinant data for Max MSE case. Comparing one-step and rollout
#x_axis_one_step_min_det = []

#for i in range(1,num_questions_nmde+1):
    #for j in range (0,rep_per_partworth_nmde):
        #x_axis_one_step_min_det.append(i)
        
#y_axis_one_step_min_det = []

#for i in range(0,num_questions_nmde):
    #for det_data in DET_onestep[0][i]:
        #y_axis_one_step_min_det.append(det_data)
        
#x_axis_rollout_min_det = []

#for i in range(1,num_questions_nmde+1):
    #for j in range (0,rep_per_partworth_nmde):
        #x_axis_rollout_min_det.append(i)
        
#y_axis_rollout_min_det = []

#for i in range(0,num_questions_nmde):
    #for det_data in DET_roll[0][i]:
        #y_axis_rollout_min_det.append(det_data)
        
#fig = sns.lineplot(x = x_axis_one_step_min_det, y = y_axis_one_step_min_det).set(title='(Min)Determinant Sequential Comparison: One-step vs Rollout')
#sns.lineplot(x = x_axis_rollout_min_det, y = y_axis_rollout_min_det).set( xlabel = "Question Number", ylabel = "Determinant")
#plt.legend(labels=["One-Step","Rollout"])
#plt.xticks([1,2,3,4])


# In[65]:


#THIS NEEDS TO BE MOVED!!!
#Getting HITRATE Results
#Plot boxplot of HITRATE for MAX MSE case
#labels_hitrate_max = ['one-step', 'rollout']
#plt.boxplot([HITRATE_onestep[1],HITRATE_roll[1]], vert=False, patch_artist=True, labels=labels_hitrate_max) 
#plt.ylabel("Hitrate")
#plt.title("Hitrate of One-Step and Rollout (MAX MSE): n=6, rep = 50")
#plt.show()


# In[66]:


#THIS NEEDS TO BE MOVED!!!
#Getting HITRATE Results
#Plot boxplot of HITRATE for MAX MSE case
#labels_hitrate_min = ['one-step', 'rollout']
#plt.boxplot([HITRATE_onestep[0],HITRATE_roll[0]], vert=False, patch_artist=True, labels=labels_hitrate_min) 
#plt.ylabel("Hitrate")
#plt.title("Hitrate of One-Step and Rollout (MIN MSE): n=6, rep = 50")
#plt.show()


# In[67]:


#This needs to be moved!
#df_MSE_det_09272022 = pd.DataFrame()
#questions_09272022 = 4
#max_or_min_09272022 = ['MIN','MAX']
#for i in range(2):
#for j in range(questions_09272022):
        #df_MSE_det_09272022['Question %s, Baseline Partworth %s, Rollout MSE' %(str(j),max_or_min_09272022[1])] = MSE_roll[1][j]
        #df_MSE_det_09272022['Question %s, Baseline Partworth %s, Rollout DET' %(str(j),max_or_min_09272022[1])] = DET_roll[1][j]

#df_MSE_det_09272022.to_csv(r'C:\Users\wsfishe\Desktop\MSE_det_Max_rep=50_quest=4_Rollout_10052022.csv',index=False,header=True)

