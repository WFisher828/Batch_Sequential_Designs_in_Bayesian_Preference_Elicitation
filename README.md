# Batch_Sequential_Designs_in_Bayesian_Preference_Elicitation
A collection of code for the experiments ran in "Batch Sequential Designs in Bayesian Preference Elicitation with Application to Tradespace Exploration for Vehicle Concept Design".


**Description of Files.** <br />
------

<ins>Helper_Functions.py</ins> : This is a Python file which contains functions that are used in both the experiment corresponding to Section 4 "Questionnaire Simulation" in the main paper as well as the experiment corresponding to the Section S3 "Quality of MIP-Based Batch Design Solutions" found in the supplementary material document. This Python file contains the following functions:
* z_expectation_variance: This function is used for calculating the expectation and variance of the random variable $Z(m,\sigma)$ defined in Section 2.2 proposition 1 of the main paper. 
* moment_matching_update: This function is used for performing a moment matching approximation of the posterior distribution of a DM's partworth given an answer to a single query (x,y) and that the partworth's prior distribution is normal. The moment matching updating equations are given in Section 2.2 proposition 1 of the main paper.
* product_diff_list: Given the number of (binary) attributes of the design alternatives (or products), this function creates a list where each element represents the difference between two alternatives. Note that we take into account that two pairs of alternatives $(x_1,y_1)$ and $(x_2,y_2)$ can lead to the same difference vector (i.e. $(1,0,1,1) - (0,1,0,1) = (1,0,1,0) - (0,1,0,0)$ ) and so we generate all vectors from the set of trinary vectors (vectors who entries are -1, 0, or 1). We also take into account symmetry (that is, (x_1 - y_1) provides the same information as (y_1 - x_1) ) and thus we remove any vector which is the negative of another vector in order to reduce redundancy.
* question_extractor: Given a trinary vector, this function will decompose it as the difference of two binary vectors and return these binary vectors.
* norm_AO_MO_data_generation: This function is used for generating data which is to be used in fitting the linear model (for either MIP-AC or MIP-MC) discussed in Section 3.3 "Offline Learning Framework for Specifying MIP Objective Parameters". Note that capabilities for capturing the effect of dynamically changing batch size across different stages ($r_b$ in the offline learning framework) were not implemented in this function due to our numerical experiments not focusing on this aspect. 
* batch_design_AO: This function is used for setting up and solving the MIP-AC problem discussed in Section 3.2 "Mixed Integer Programming Formulations of Batch Designs". [Gurobi](https://www.gurobi.com/) will need to be installed for this function to work.
* batch_design_MO: This function is used for setting up and solving the MIP-MC problem discussed in Section 3.2 "Mixed Integer Programming Formulations of Batch Designs". [Gurobi](https://www.gurobi.com/) will need to be installed for this function to work.

<ins>MIP_formulation_vs_Enumeration_Experiment_v6.ipynb</ins>: This is a Jupyter notebook which is used in conducting the experiment corresponding to Section S3 "Quality of MIP-Based Batch Design Solutions" found in the supplementary material document. It further contains two functions: (1) KMS_Matrix(n,r) which is used for constructing a Kac-Murdock-Szego matrix of size $n$ with parameter $r$. (2) random_batch_D_error(init_mu,init_Sig,batch_size,num_random_batches,true_partworths, gumbel_error_terms) which is used for generating a number of random batch designs of some specified size and evaluate each of their D-errors under distribution N(init_mu,init_Sig).

<ins>Comparing_AO_and_MO_numerical_Experiment_ExpIII_v2.ipynb</ins>: This is a Jupyter notebook which is used in conducting the experiment corresponding to the Section 4 "Questionnaire Simulation" found in the main paper. It further contains two functions: (1) mean_CI_95(data) which is used for calculating the mean of a dataset and constructing 95% confidence intervals. (2) sequential_AO_MO_experiment(init_mu,init_Sig,true_partworths,gumbel_error_terms,rep_per_partworth,num_questions,AO_MO_batch_size,AO_alpha, AO_kappa, AO_gamma, MO_alpha, MO_kappa, MO_gamma, OS_alpha, OS_kappa, noise_par, hitrate_question_list, JMP_questionnaire, Method = 0, t=100) which is used in conducting questionnaires. 

<ins>JMP_attr_6_exp_1_cov_1_loc_025_scale_4_quest_16.csv</ins>: This is a non-adaptive Bayesian D-optimal questionnaire constructed in JMP under the balanced low signal-to-noise ratio setting discussed in Section S2 "Detailed Experiment Setup for Questionnaire Simulation" found in the supplementary material document. This csv file is used in Comparing_AO_and_MO_numerical_Experiment_ExpIII_v2.ipynb.

<ins>JMP_attr_6_exp_1_cov_1_loc_1_scale_1_quest_16.csv</ins>: This is a non-adaptive Bayesian D-optimal questionnaire constructed in JMP under the balanced medium signal-to-noise ratio setting discussed in Section S2 "Detailed Experiment Setup for Questionnaire Simulation" found in the supplementary material document. This csv file is used in Comparing_AO_and_MO_numerical_Experiment_ExpIII_v2.ipynb.

<ins>JMP_attr_6_exp_1_cov_1_loc_4_scale_025_quest_16.csv</ins>: This is a non-adaptive Bayesian D-optimal questionnaire constructed in JMP under the balanced medium signal-to-noise ratio setting discussed in Section S2 "Detailed Experiment Setup for Questionnaire Simulation" found in the supplementary material document. This csv file is used in Comparing_AO_and_MO_numerical_Experiment_ExpIII_v2.ipynb.

<ins>parameter_file_AO_MO_JMP_exp.txt</ins>: This file corresponds to the experiment in "Questionnaire Simulation". This txt file can be used if the user wishes to run all three of the low, medium, and high signal-to-noise ratio (SNR) experiment settings in a job array using a Bash script. The Jupyter notebook may be saved as a python file and then a Bash script may be written to run all three experiments in a job array, where the entries in parameter_file_AO_JMP_exp.txt will be the input. In this txt file, "1" corresponds to low SNR, "2" corresponds to medium SNR, and "3" corresponds to high SNR. (this may be an efficient choice if,for example, the user has access to a computing cluster).

<ins>parameter_file_MIPvsEnum_v6_exp.txt</ins>: This file corresponds to the experiment in "Quality of MIP-Based Batch Design Solutions". This txt file can be used if the user wishes to run all six experiment settings ( (low signal-to-noise, balanced), (medium signal-to-noise, balanced), (high signal-to-noise, balanced),(low signal-to-noise, imbalanced), (medium signal-to-noise, imbalanced), (high signal-to-noise, imbalanced) ) in a job array using a Bash script. Column one corresponds to the signal-to-noise setting ("1":low, "2": medium, "3": high) and column two corresponds to the balanced ("1") or imbalanced ("2") setting discussed in Section S3 of the supplementary materials.

**Running Experiments**: <br />
--------

<ins>Required Software</ins>: In order to run the experiments in Section 4 "Questionnaire Simulation" and Section S3 "Quality of MIP-Based Batch Design Solutions", one will need to have installed [Python](https://www.python.org/), [Jupyter Notebook](https://jupyter.org/), and [Gurobi](https://www.gurobi.com/) on their machine. 

Next, the python file "Helper_Functions.py", which contains functions used in both experiments, must be downloaded as the Jupyter Notebook files make calls to these functions.

Now, we will give directions for running each of the experiments.

<ins>Questionnaire Simulation</ins>: The notebook corresponding to this experiment is Comparing_AO_and_MO_numerical_Experiment_ExpIII_v2.ipynb. This experiment has three settings, which are the low, medium, and high signal-to-noise ratio settings. In order to run each of the three settings, one will download the Jupyter notebook and proceed to the 4th cell where they will see:

```python
#signal to noise ratio. 
#1 - LOW: multiply expectation by 0.25 and covariance by 4.0
#2 - REG: multiply expectation by 1.0 and covariance by 1.0
#3 - HIGH: multiply expectation by 4.0 and covariance by 0.25
snr = int(sys.argv[1]) 
```
Now, to run the experiment directly in the notebook, one will comment-out/delete "int(sys.argv\[1\])" and replace it with 1 (integer, not string!) for the low SNR setting, 2 for the medium SNR setting, and 3 for the high SNR setting. 

If one has access to a computing cluster, they may download the notebook as a python file and write a Bash script to run all three experiment settings as a batch job array, using parameter_file_AO_MO_JMP_exp.txt as the inputs.

<ins>Quality of MIP-Based Batch Design Solutions.</ins>: The notebook corresponding to this experiment is MIP_formulation_vs_Enumeration_Experiment_v6.ipynb. This experiment has 3x2 settings, which are the combinations of SNR and prior type: (low SNR, balanced prior), (medium SNR, balanced prior), (high SNR, balanced prior), (low SNR, imbalanced prior), (medium SNR, imbalanced prior), and (high SNR, imbalanced prior). In order to run each of the 3x2 settings, one will download the Jupyter notebook and proceed to the 4th cell where the will see:

```python
#signal to noise ratio. 
#1 - LOW: multiply expectation by 0.25 and covariance by 4.0
#2 - REG: multiply expectation by 1.0 and covariance by 1.0
#3 - HIGH: multiply expectation by 4.0 and covariance by 0.25
snr = int(sys.argv[1])


#Prior type
#1 - homogeneous expectation and identity covariance matrix
#2 - heterogeneous expectation and KMS covariance matrix
prior_type = int(sys.argv[2])
```
Now, to run the experiment directly in the notebook, one will comment-out/delete "int(sys.argv\[1\])" and "int(sys.argv\[2\])" and replace them with 
* 1 (integer) and 1 (integer) for the low SNR, balanced prior regime.
* 2 and 1 for the medium SNR, balanced prior regime.
* 3 and 1 for the high SNR, balanced prior regime.
* 1 and 2 for the low SNR, imbalanced prior regime.
* 2 and 2 for the medium SNR, imbalanced prior regime.
* 3 and 2 for the high SNR, imbalanced prior regime.

If one has access to a computing cluster, they may download the notebook as a python file and write a Bash script to run all 3x2 experiment settings as a batch job array, using parameter_file_MIPvsEnum_v6_exp.txt as the inputs.

**To Do List**: <br />
--------

Priority of tasks is denoted by the point number (1 has more priority than 2, etc.). Tasks will be updated when new problems are identified.

1. (README): Make the README document as detailed as possible to aid in the reviewers reproducing the experiment results.
2. (Helper Functions Examples): Write examples for helper functions.
3. (Reproducibility): We will check that the code works in an environment other than Jupyter Notebook. We will rerun the experiment files as well to check that everything is running correctly.
4. (Detailed Comments): Provide more detailed comments for the functions in each of the modules, describing in detail the purpose and use of the function. Comments regarding the definition of arguments in the function should include the exact data-type expected, along with any restrictions on the input (later on we may consider adding code into the functions to make sure the user is not inputting invalid arguments). Examples of functions should be provided in a separate file so that users can observe how the functions work, and what their output looks like. Write a brief users manual of the functions within the modules.
5. (Overview Files): Experiment files should not only have comments in the code myopically explaining the purpose of certain lines, but there should also be a file or introduction which describes the purpose and details of the experiment.
6. (File formatting): Data saved to files should be formatted in a neat, easy-to-read manner.
7. (Look for Bugs): After or along the way of completing points (1)-(7), we will recheck the code for any mistakes/bugs.
8. (Renaming): We will rename certain functions/variables to reflect names given in the paper.
9. (Remove sys.argv): We used sys.argv when we converted the jupyter notebook to a python file and ran it on the Palmetto cluster to facilitate the use of job arrays. Reviewers may only want to use the notebook version, and so we may want to get rid of sys.argv so that they can manually place in the argument for the experiment.

-----
