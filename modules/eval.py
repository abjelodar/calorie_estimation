# Copyright (c) Robotic and Action Perception Lab (RPAL) at University of South Florida
# by Ahmad Babaeian Jelodar

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single
import modules.utils as utils
from torch.nn.init import xavier_uniform_
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import copy, os, time, pickle, json
from data_modules.data_aux import Path, load_calorie_dataset
from data_modules.calorie_info import CalorieInfo
import matplotlib.pyplot as plt

# =======================================
def calorie_by_separate_ingredients_calories(calorie_dataset_path, splits, indiv_calories_path, args, dataset, value_tag="non_scaled_portion", prec=2, ingr_calorie_thresh=[5], epsilon=0.0001, units_path=None, portions_results=None):
    '''
        Computes the estimated calorie for each recipe 
        based on separate estimated ingredient calories
        inputs:
            ingr_calorie_thresh: specifies to compute ingredient percentage error for ingredients where the calorie is in a range
    '''

    ingrs_use_mean = {"baking_soda", "salt", "water", "ice", "food_coloring", "coffee", 
                      "hot_sauce", "allspice", "tea", "coriander", "capers", "baking_powder", 
                      "cilantro", "tartar_cream", "thyme", "dill", "chives", "turmeric", 
                      "fish_sauce", "ginger", "cookies", "chips", "roast", "lamb", "chickpeas"}

    ingrs_use_mean = {}

    filtered_ids = dataset.get_filtered_ids()
    filtered_rids = dataset.rids

    if not os.path.exists(indiv_calories_path):
        return -1 # no error computed

    qid_to_unitportion_preds = {}
    # load units results
    predicted_units = json.load(open(units_path, 'r'))

    ingredient_calories = json.load(open(indiv_calories_path, 'r'))["results"]

    if len(ingredient_calories)==0:
        return -1 # no error computed

    if filtered_ids!=[]:
        ingredient_calories_temp = {}
        for qid in set(filtered_ids):
            if not qid in ingredient_calories.keys():
                qid = str(qid)
            ingredient_calories_temp[qid] = ingredient_calories[qid]
    ingredient_calories = ingredient_calories_temp

    if args.dataset=="recipe1m":
        ground_truth_recipe_calories, min_stat, max_stat = load_calorie_dataset(path=calorie_dataset_path)
        path = Path(args)
        rid_to_ingrs, _, _, _, _, _, _, _, _ = path.load_ques_ans(splits, value_tag=value_tag)
    elif args.dataset=="menumatch":
        ground_truth_recipe_calories = dataset.rid_to_calorie
        rid_to_ingrs = dataset.rid_to_ingrs

    calorie_errors = {"ingredients_error_percent": 0.0,
                     "ingredients_error": 0.0,
                     "recipe_error_percent": 0.0,
                     "recipe_error": 0.0,
                     "ingredients_prior_error": 0.0,
                     "ingredients_prior_error_percent": 0.0,
                     "prior_recipe_error_percent": 0.0,
                     "prior_recipe_error": 0.0}
    
    total_msep_predicted_calorie = 0.0
    total_msep_gt_calorie = 0.0
    total_msep = 0.0


    total = 0.0
    total_ingrs = 0.0
    total_thresholded_ingrs = 0.0
    # computing total recipe calories using a summation of individual ingredient calories
    for rid in rid_to_ingrs.keys():
        if rid in filtered_rids:
            predicted_recipe_calorie = 0.0
            prior_recipe_calorie = 0.0
            for qid in rid_to_ingrs[rid]:
                
                if not qid in ingredient_calories.keys():
                    qid = str(qid)

                predicted_ingr_calorie = ingredient_calories[qid]["calorie_prediction"]
                qid_to_unitportion_preds[qid] = (predicted_units[qid]["unit_prediction"], portions_results[qid]["portion_prediction"], predicted_ingr_calorie)

                target_ingr_calorie = ingredient_calories[qid]["calorie_target"]
                ingredient = ingredient_calories[qid]["ingredient"]
                #print (predicted_ingr_calorie, target_ingr_calorie)
                ingr_calorie_mean = dataset.get_ingr_calorie_mean(ingredient_calories[qid]["ingredient"])
                if args.dataset=="recipe1m":
                    dataset.ingr_prediction_stats_update(ingredient, abs(predicted_ingr_calorie-target_ingr_calorie))

                #if ingredient in ingrs_use_mean:
                #    predicted_ingr_calorie = ingr_calorie_mean

                # recipe calorie computation if using predicted and only mean ingredient calorie
                predicted_recipe_calorie += predicted_ingr_calorie
                prior_recipe_calorie += ingr_calorie_mean

                # compute current ingredient error & its percentage
                ingr_calorie_error = abs(predicted_ingr_calorie-target_ingr_calorie)
                calorie_errors["ingredients_error"] += ingr_calorie_error

                # compute ingredient caloie error if using ingredient calorie mean as predicted calorie (prior error)
                ingr_prior_calorie_error = abs(ingr_calorie_mean-target_ingr_calorie)
                calorie_errors["ingredients_prior_error"] += ingr_prior_calorie_error

                if check_in_range(target_ingr_calorie, ingr_calorie_thresh):

                    if target_ingr_calorie==0:
                        target_ingr_calorie += 0.01

                    # aggregate ingredients calorie errors (percent and absolute)
                    calorie_errors["ingredients_error_percent"] += ingr_calorie_error / target_ingr_calorie
                    calorie_errors["ingredients_prior_error_percent"] += ingr_prior_calorie_error / (target_ingr_calorie + epsilon)
                    total_thresholded_ingrs += 1.0

                total_ingrs += 1.0

            # aggregate recipe calorie errors (percent and absolute)
            cur_recipe_error = abs(ground_truth_recipe_calories[rid]-predicted_recipe_calorie)
            
            total_msep_predicted_calorie += predicted_recipe_calorie
            total_msep_gt_calorie += ground_truth_recipe_calories[rid]
            total_msep += 1.0
            
            calorie_errors["recipe_error_percent"] += cur_recipe_error / (ground_truth_recipe_calories[rid] + epsilon)
            calorie_errors["recipe_error"] += cur_recipe_error

            # aggregate recipe prior calorie errors (percent and absolute)
            cur_recipe_prior_error = abs(ground_truth_recipe_calories[rid]-prior_recipe_calorie)
            calorie_errors["prior_recipe_error_percent"] += cur_recipe_prior_error / (ground_truth_recipe_calories[rid] + epsilon)
            calorie_errors["prior_recipe_error"] += cur_recipe_prior_error

            total += 1

    total_msep_predicted_calorie /= total_msep
    total_msep_gt_calorie /= total_msep
    msep = round(abs(total_msep_predicted_calorie-total_msep_gt_calorie)/total_msep_gt_calorie, 3)

    print ("msep predicted calorie: {}".format(total_msep_predicted_calorie))
    print ("msep target calorie: {}".format(total_msep_gt_calorie))
    print ("Calorie MSE%: {}".format(msep))

    # compute recipe calorie errors (percent and absolute)
    calorie_errors["recipe_error_percent"] = round(100*calorie_errors["recipe_error_percent"]/total, prec)
    calorie_errors["recipe_error"] = round(calorie_errors["recipe_error"]/total, prec)

    # compute prior recipe calorie errors (percent and absolute)
    calorie_errors["prior_recipe_error_percent"] = round(100*calorie_errors["prior_recipe_error_percent"]/total, prec)
    calorie_errors["prior_recipe_error"] = round(calorie_errors["prior_recipe_error"]/total, prec)

    # individual ingredient error
    calorie_errors["ingredients_error_percent"] = round(100*calorie_errors["ingredients_error_percent"]/(total_thresholded_ingrs+epsilon), prec)
    calorie_errors["ingredients_error"] = round(calorie_errors["ingredients_error"]/total_ingrs, prec)

    # individual ingredient error
    calorie_errors["ingredients_prior_error_percent"] = round(100*calorie_errors["ingredients_prior_error_percent"]/(total_thresholded_ingrs+epsilon), prec)
    calorie_errors["ingredients_prior_error"] = round(calorie_errors["ingredients_prior_error"]/total_ingrs, prec)

    dataset.compute_ingr_unit_portion_error(qid_to_unitportion_preds)

    print ("Ingredients calorie errors (threshold: {}, total count: {}):".format(ingr_calorie_thresh, int(total_thresholded_ingrs)))
    print ("    predicted errors: {}% - {}".format(calorie_errors["ingredients_error_percent"], calorie_errors["ingredients_error"]))
    print ("    prior (using mean) errors: {}% - {}".format(calorie_errors["ingredients_prior_error_percent"], calorie_errors["ingredients_prior_error"]))
    print ("Recipe calorie errors based on separate calorie predictions (total count: {}):".format(total))
    print ("    predicted errors: {}%, {}".format(calorie_errors["recipe_error_percent"], calorie_errors["recipe_error"]))
    print ("    prior (using mean) errors: {}%, {}".format(calorie_errors["prior_recipe_error_percent"], calorie_errors["prior_recipe_error"]))

    #dataset.ingr_prediction_stats_finalize()

# =======================================

def calorie_by_separate_results(splits, units_path, portions_path, indiv_calories_path, dataset, args, value_tag="non_scaled_portion", prec=2):
    '''
        Computes the estimated calorie for each recipe 
        based on separate results for unit and portion predictions of each ingredient
    '''

    if args.dataset=="menumatch":
        return

    if not os.path.exists(units_path) or not os.path.exists(portions_path) or not os.path.exists(indiv_calories_path):
        return -1 # no error computed

    calorie_info = CalorieInfo()

    predicted_units = json.load(open(units_path, 'r'))
    predicted_portions = json.load(open(portions_path, 'r'))["results"]
    predicted_calories = json.load(open(indiv_calories_path, 'r'))["results"]

    if len(predicted_portions)==0 or len(predicted_units)==0 or len(predicted_calories)==0:
        return -1 # no error computed

    if args.dataset=="recipe1m":
        path = Path(args)
        rid_to_ingrs, _, _, _, _, _, _, _, _ = path.load_ques_ans(splits, value_tag=value_tag)
    elif args.dataset=="menumatch":
        rid_to_ingrs = dataset.rid_to_ingrs
    
    total = 0.0
    calorie_error = 0.0
    total_indivs = 0.0
    indiv_calorie_error = 0.0

    # computing total recipe calories using a summation of individual unit & portion estimations
    for rid in rid_to_ingrs.keys():
        if rid in dataset.rids:

            gt_recipe_calorie = dataset.iid_to_calorie[rid]
            predicted_recipe_calorie = 0.0

            for qid in rid_to_ingrs[rid]:
                if not qid in predicted_units.keys():
                    qid = str(qid)
                ingr = predicted_units[qid]["ingredient"]
                unit = predicted_units[qid]["unit_prediction"]
                portion = predicted_portions[qid]["portion_prediction"]
                indiv_calorie = predicted_calories[qid]["calorie_target"]
                ingr_for_computation = dataset.qid_to_ingr_for_comp[qid]

                gt_indiv_calorie = calorie_info.ingredient_calorie_estimation(portion, unit, ingr, ingr_for_computation)
                predicted_recipe_calorie += gt_indiv_calorie

                indiv_calorie_error += abs(gt_indiv_calorie-indiv_calorie)
                total_indivs += 1

            calorie_error += abs(gt_recipe_calorie-predicted_recipe_calorie)
            total += 1

    indiv_calorie_error = round(indiv_calorie_error/total_indivs, prec)        
    calorie_error = round(calorie_error/total, prec)

    print ("Based on separate (unit, portion) predictions: Recipe calorie error: {}, Indiv calorie error: {}".format(calorie_error, indiv_calorie_error))

# =======================================

def plot_2d_error(units_path, indiv_calories_path, dataset=None, prec=2, log_path="logs/", font_size=16):
    '''
        plots the error of (ingredient, unit) pair on a 2-D plane
            1. the radius of the plotted circle error shows the absolute error for that pair.
            2. the opacity of the plotted circle error shows the number of instances labeled with that pair.
    '''

    if not os.path.exists(units_path) or not os.path.exists(indiv_calories_path):
        return -1 # no error computed

    predicted_units = json.load(open(units_path, 'r'))
    predicted_calories = json.load(open(indiv_calories_path, 'r'))["results"]    

    if len(predicted_units)==0 or len(predicted_calories)==0:
        return -1 # no error computed

    min_count = 1000000
    max_count = 0

    ingr_unit_calorie_errors = {}
    ingr_unit_calorie_count = {}
    for qid in  predicted_units.keys():
        if dataset==None:
            ingr = predicted_units[qid]["ingredient"]
            # ground-truth items
            unit_gt = predicted_units[qid]["unit_target"]
            calorie_gt = predicted_calories[qid]["calorie_target"]
        else:
            ingr = dataset.qid_to_ingr[qid]
            # ground-truth items
            unit_gt = dataset.qid_to_unit[qid]
            calorie_gt = dataset.qid_to_calorie[qid]

        if qid in predicted_units.keys():
            # predicted items
            unit_pr = predicted_units[qid]["unit_prediction"]
            calorie_pr = predicted_calories[qid]["calorie_prediction"]

            error = abs(calorie_pr-calorie_gt)

            if ingr in ingr_unit_calorie_errors.keys() and unit_gt in ingr_unit_calorie_errors[ingr].keys():
                ingr_unit_calorie_errors[ingr][unit_gt] += error
                ingr_unit_calorie_count[ingr][unit_gt] += 1.0
            elif ingr in ingr_unit_calorie_errors.keys() and not unit_gt in ingr_unit_calorie_errors[ingr].keys():
                ingr_unit_calorie_errors[ingr][unit_gt] = error
                ingr_unit_calorie_count[ingr][unit_gt] = 1.0
            else:
                ingr_unit_calorie_errors[ingr] = {unit_gt: error}
                ingr_unit_calorie_count[ingr] = {unit_gt: 1.0}

            min_count = min(min_count, ingr_unit_calorie_count[ingr][unit_gt])
            max_count = max(max_count, ingr_unit_calorie_count[ingr][unit_gt])

    unit_set = set({})
    min_val = 10000
    max_val = 0
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(48,32))
    for ingr in sorted(ingr_unit_calorie_errors.keys()):
        for unit_gt in sorted(ingr_unit_calorie_errors[ingr].keys()):
            unit_set.add(unit_gt)
            ingr_unit_calorie_errors[ingr][unit_gt] /= ingr_unit_calorie_count[ingr][unit_gt]
            ingr_unit_calorie_errors[ingr][unit_gt] = round(ingr_unit_calorie_errors[ingr][unit_gt], prec)
            max_val = max(max_val, ingr_unit_calorie_errors[ingr][unit_gt])
            min_val = min(min_val, ingr_unit_calorie_errors[ingr][unit_gt])

    regions = 10
    boundaries = np.array(range(1,regions+1)) * (max_val-min_val)/regions
    colors = ['linen', 'peachpuff', 'lightsalmon', 'coral', 'salmon', 'tomato', 'red', 'brown', 'darkred', 'black']
    color = 'red'
    print ("number of ingredients: {}".format(len(ingr_unit_calorie_errors.keys())))
    print ("number of units: {}".format(len(unit_set)))

    ingr_labels = sorted(ingr_unit_calorie_errors.keys())
    unit_labels = sorted(list(unit_set))

    x = y = 0
    for ingr in ingr_labels:
        x += 1
        y = 0
        for unit_gt in unit_labels:
            y += 1

            # for scatter graph
            if not ingr in ingr_unit_calorie_errors.keys() or not unit_gt in ingr_unit_calorie_errors[ingr].keys():
                continue

            # scale (radius) shows the average error for the specific ingr,unit pair
            scale = 10+3*ingr_unit_calorie_errors[ingr][unit_gt]
            '''for idx, b in enumerate(boundaries):
                if scale<=b:
                    color = colors[idx]
                    break'''

            # alpha shows how many of the specific (inge,unit) pairs are there
            alpha = 0.2 + 0.8 * (ingr_unit_calorie_count[ingr][unit_gt]-min_count)/(max_count-min_count)
            ax.scatter(x, y, c=color, s=scale, label=color, alpha=round(alpha,2), edgecolors='none')

    ind_x = np.arange(1, len(ingr_labels)+1)
    ind_y = np.arange(1, len(unit_labels)+1)

    plt.yticks(ind_y, fontsize=font_size)
    plt.xticks(ind_x, rotation='vertical', fontsize=font_size)
    ax.set_xticklabels( ingr_labels )
    ax.set_yticklabels( unit_labels )

    ax.grid(True)
    #plt.show()
    fig.savefig(os.path.join(log_path, 'ingr-unit-error.png'))   # save the figure to file
    print ('error figure saved!')

# =======================================

def check_in_range(val, val_range):

    if not val_range:
        return True
    
    if val<val_range[0]:
        return False
        
    if len(val_range)>1 and val>=val_range[1]:
        return False
    
    # if len(val_range)==1 or val<val_range[1]
    return True

# =======================================
