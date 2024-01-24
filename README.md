# Batch_Sequential_Designs_in_Bayesian_Preference_Elicitation
A collection of code for the paper "Batch Sequential Designs in Bayesian Preference Elicitation with Application to Tradespace Exploration for Vehicle Concept Design" 

-----
**Project Overview and Description of Files.** <br />

<ins>Helper_Functions.py</ins> : This is a Python file which contains functions that are used in both the experiment corresponding to the section "Questionnaire Simulation" in the main paper as well as the experiment corresponding to the section "Quality of MIP-Based Batch Design Solutions" found in the supplementary material document. This Python file contains the following functions:
* z_expectation_variance
* moment_matching_update
* product_diff_list
* question_extractor
* norm_AO_MO_data_generation
* batch_design_AO
* batch_design_MO

<ins>MIP_formulation_vs_Enumeration_Experiment_v6.ipynb</ins>: This is a Jupyter notebook which is used in conducting the experiment corresponding to the section "Quality of MIP-Based Batch Design Solutions" found in the supplementary material document.

<ins>Comparing_AO_and_MO_numerical_Experiment_ExpIII_v2.ipynb</ins>: This is a Jupyter notebook which is used in conducting the experiment corresponding to the section "Questionnaire Simulation" found in the main paper.

<ins>JMP_attr_6_exp_1_cov_1_loc_025_scale_4_quest_16.csv</ins>: This is a non-adaptive Bayesian D-optimal questionnaire constructed in JMP under the balanced low signal-to-noise ratio setting discussed in the section "Detailed Experiment Setup for Questionnaire Simulation" found in the supplementary material document. This csv file is used in Comparing_AO_and_MO_numerical_Experiment_ExpIII_v2.ipynb.

<ins>JMP_attr_6_exp_1_cov_1_loc_1_scale_1_quest_16.csv</ins>: This is a non-adaptive Bayesian D-optimal questionnaire constructed in JMP under the balanced medium signal-to-noise ratio setting discussed in the section "Detailed Experiment Setup for Questionnaire Simulation" found in the supplementary material document. This csv file is used in Comparing_AO_and_MO_numerical_Experiment_ExpIII_v2.ipynb.


<ins>JMP_attr_6_exp_1_cov_1_loc_4_scale_025_quest_16.csv</ins>: This is a non-adaptive Bayesian D-optimal questionnaire constructed in JMP under the balanced medium signal-to-noise ratio setting discussed in the section "Detailed Experiment Setup for Questionnaire Simulation" found in the supplementary material document. This csv file is used in Comparing_AO_and_MO_numerical_Experiment_ExpIII_v2.ipynb.

<ins>parameter_file_AO_MO_JMP_exp.txt</ins>: This txt file can be used if the user wishes to run all three of the low, medium, and high signal-to-noise ratio experiment settings in a job array. This file corresponds to the experiment in "Questionnaire Simulation".

<ins>parameter_file_MIPvsEnum_v6_exp.txt</ins>: This txt file can be used if the user wishes to run all six experiment settings ( (low signal-to-noise, balanced), (medium signal-to-noise, balanced), (high signal-to-noise, balanced),(low signal-to-noise, imbalanced), (medium signal-to-noise, imbalanced), (high signal-to-noise, imbalanced) ) in a job array. This file corresponds to the experiment in "Quality of MIP-Based Batch Design Solutions".

-----
**To Do List**: <br />

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
