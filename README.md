# Preference_Learning_Project
A collection of code for the Bayesian preference learning project 

-----
**!!!TO DO LIST!!!**: <br />
(Batch) - Task relating to the Batch-Design project. <br />
(Non-myopic) - Task relating the non-myopic portion of the project. <br />
(Batch/Non-myopic) - Tasks relating to both batch-design and non-myopic portion of the project. <br />

Priority of tasks is denoted by the point number (1 has more priority than 2, etc.). We will also put more priority into tasks relating to the batch-design project, as the 
results there seem to have a better chance of publishing. The non-myopic portion of the project seems like it will be used in the VIPR-GS report, but may not be publishable at the moment. Tasks will be updated when new problems are identified.

**1.0**(Helper Functions): Need to take helper functions and convert them into python files.

**1.1**(Batch/Non-myopic): Need to work on DEPENDENCIES between files: "Baseline_Functions_Definitions", "Questionnaire_Procedure", "Experiment_Framework", "Batch_Design_and_Rollout".
These files contain functions which are used throughout our numerical experiments. Specifically, get rid of "sys.path.append(r'C:\Users\wsfishe\Desktop\PreferenceElicitationCode')" (It seems to import your own modules in Jupyter Notebook you need to save the modules on your own device and import them in using sys.path.append) in file "Batch_Design_and_Rollout", "Experiment_Framework", "Questionnaire_Procedure".

**1.2**(Helper Functions): Remove examples from inside the Modules containing helper functions. Place those in separate files.

**2**(Batch/Non-myopic): Identify major experiments (experiments included in the paper) performed for the Batch-Design paper, "Batch Sequential Designs in Bayesian Preference Elicitation with Application to Tradespace Exploration for Vehicle Concept Design", as well as major experiments conducted in the portion of the project where we investigated non-myopic methods. The Jupyter-Notebook files for these experiments directly contain each required function from the main modules ("Baseline_Functions_Definitions", "Questionnaire_Procedure", "Experiment_Framework", "Batch_Design_and_Rollout") at the beginning of the file before the experiment begins. For readability and brevity of the experiment file, we will want to IMPORT the required functions for each experiment from the corresponding modules, rather than directly containing the required functions within the file. Include these experiments as separate files.

**3**(Batch/Non-myopic): For each of the major experiments found in point (2), check whether there are functions in these experiment files which can be placed within one of the four major modules in point (1).

**4**(Non-myopic): One experiment for the non-myopic project (sequential comparison of one-step, two-step via enumeration, rollout over batch, and rollout via coordinate exchange with 100 partworths) did not include gumbel error terms in the selection process. We need to fix this so that gumbel error terms are included.

**5**(Batch/Non-myopic): Provide more detailed comments for the functions in each of the modules, describing in detail the purpose and use of the function. Comments regarding the definition of arguments in the function should include the exact data-type expected, along with any restrictions on the input (later on we may consider adding code into the functions to make sure the user is not inputting invalid arguments). Examples of functions should be provided in a separate file so that users can observe how the functions work, and what their output looks like. Write a brief users manual of the functions within the modules.

**6**(Batch/Non-myopic): Experiment files should not only have comments in the code myopically explaining the purpose of certain lines, but there should also be a file or introduction which describes the purpose and details of the experiment.

**7**(Batch/Non-myopic): Experiment files should be separate from plotting files. We will clearly denote which plotting file corresponds to which experiment file. We will provide detailed comments on how to take data from the experiment file and use it to generate plots/results in the plotting file.

**8**(Batch/Non-myopic): After or along the way of completing points (1)-(7), we will recheck the code for any mistakes/bugs.

**9**(Batch/Non-myopic): We will check that the code works in an environment other than Jupyter Notebook. We will run examples in point (5) and also rerun the experiment files as well to check that everything is running correctly.

-----

The module "Baseline_Functions_Definitions" includes functions that the questionnaire procedure and experimental framework
are based off of. The functions include:
1. z_expectation_variance
2. g_fun
4. g_fun_linear_regression

The module 'Questionnaire_Procedure' has all of the functions needed to do one-step and 
two-step questionnaire, along with the two-step acquisition function since it depends on functions
coming from this module. It includes the following functions:
1. moment_matching_update
2. g_opt
3. g_opt_multi_lvl
4. two_stage_g_opt
5. two_step_g_acq (WE INCLUDE THIS HERE BECAUSE IT DEPENDS ON moment_matching_update AND g_opt, THEMATICALLY THIS
SHOULD NOT BE HERE)
6. multlvl_two_step_g_acq (WE INCLUDE THIS HERE BECAUSE IT DEPENDS ON moment_matching_update AND g_opt_multi_lvl, THEMATICALLY THIS
SHOULD NOT BE HERE)

The module "Experiment_Framework" has functions which are used in conducting numerical experiments. These functions include:
1. product_diff_list
2. multlvl_product_diff_list
3. question_extractor
4. multlvl_question_extractor
5. enum_two_step
6. multlvl_enum_two_step
7. enum_two_step_opt
8. quantile_test_enum_data
9. one_step_sol_two_step_quantile
10. multlvl_one_step_sol_two_step_quantile
11. MSE_det_test 
12. new_sequential_experiment (!!!THIS CODE NEEDS TO BE CHECKED IN DETAIL!!! PARTICULARLY DEALING WITH THE GUMBEL ERROR TERM)

The module "Batch_Design_and_Rollout" has functions which are used in creating a batch design of questions with certain orthogonality conditions,
as well as functions which are used in performing rollout on a question pair. Rollout is a non-myopic method often used in solving dynamic programming problems. 
These functions include:
1. orthogonal_constraint_feas
2. batch_design_delta_penalty #THIS WILL BE PREFERRED OVER batch_design_delta_refine
3. batch_design_AO
4. batch_design_MO
5. batch_design_delta_refine
6. question_selection_prob
7. rollout
8. monte_carlo_rollout
9. rollout_with_batch_design_acquisition
10. coordinate_exchange_acq

