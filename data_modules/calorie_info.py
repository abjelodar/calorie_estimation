# Copyright (c) Robotic and Action Perception Lab (RPAL) at University of South Florida
# by Ahmad Babaeian Jelodar

'''
    This file creates all calorie info for all ingredients and their units.
'''

import os
import numpy as np
import sys, json
import collections, pickle

# ------------------------------------

class CalorieInfo:

    def __init__(self, file_path="data/calories_info.txt"):

        self.main_rules = {"ounce_to_gram": 28.3495, "milliliter_to_gram": 1}
        self.None_to_gram = {"chicken": 172, "egg": 57, "ribs": 1360, "potatoes": 150, "avocado": 150, "steaks": 500,
                             "mushrooms": 125, "eggplant": 82, "sausage": 75, "bread": 25, "olive_oil": 13.3, 
                             "fish": 24, "fillets": 50, "shrimp": 85, "tortillas": 64, 
                             "rosemary": 1.12, "thyme": 0.91, "basil": 0.42, "mint": 1.88, "sage": 2.03,
                             "cooking_spray": 0.3, "margarine": 14, "cilantro": 1.83,
                             "coriander": 1.69, "noodles": 50, "syrup": 22, "oregano": 3.4, "dill": 2.36, "beef": 453,
                             "meat": 453, "rice": 200, "salad_dressing": 200, "cinnamon": 2.64, "beans": 23,
                             'ketchup': 235, 'herbs': 1.25, 'turmeric': 3.18, 'curry': 6, 'mustard': 15.8, 'cumin': 2.03,
                             'soy_sauce': 16, 'icing': 7.8, 'hot_sauce': 14, 'ham': 135,
                             'milk': 245, 'pasta': 200, 'cardamom': 1.96, 'lamb': 453, 'oil': 13.6, 'oysters': 18, 
                             'vinegar': 14.4, 'tarragon': 1.83, 'tuna': 28, 'turkey': 453, 'seasoning': 8, 'marjoram': 0.6, 
                             'worcestershire_sauce': 17, 'salsa': 250, 'splenda_sweetener': 15, "lentils": 190,
                             'crabmeat': 453, 'oats': 90, "jalapeno": 14, "sweet_peppers": 120, "lime": 44, 
                             "lemon_juice": 15, "butter": 14.2, "pork": 85, "salmon": 114, "cream": 240, "tartar_cream": 3, "celery": 50,
                             "allspice": 2, "parsley": 6, "olives": 180, "paprika": 1, "flour": 120, "whipped_topping": 60,
                             "chickpeas": 28.35, "cheese": 85}

        self.None_to_milliliter = {"pepper_chili": 70, "chili_pepper": 70, "tomatoes": 80, "carrots": 50, "orange": 120, 
            "apple": 120, "banana": 90, "zucchini": 170, "strawberries": 35, "mango": 150, "lemon": 58, "onion": 150, 
            "peach": 150, "beets": 100, "pear": 100, "cucumber": 170, "squash": 170, "potatoes_sweet": 170, "apricot": 35, 
            "fruit": 50, "pumpkin": 946, "dates": 22, "radishes": 5, "garlic": 15, "berries": 10, "grapes": 25, "peas": 20, 
            "blueberries": 25, "cranberries": 1, "raspberries": 25, "capers": 25, "cherries": 35, "bay": 1, "lettuce": 50, 
            "broccoli": 91, "spinach": 30, "cabbage": 89, "chives": 48, "greens": 67, "arugula": 20, "asparagus": 50, 
            "vegetables": 50, "cauliflower": 50, "fennel": 50, "artichoke": 50, "leek": 50, "peanuts": 30,  "nuts": 30, 
            "seeds": 10, "corn": 20, "chocolate_chips": 30, "almonds": 30, "chestnuts": 35, "cashews": 30, "pecans": 30, 
            "walnuts": 35, "raisins": 20, "nutmeg": 30, "chips": 30, "chocolate": 30, "crackers": 30, "marshmallows": 10, 
            "pie_crust": 100, "crust_pie": 100, "pie_shell": 100, "shell_pie": 100, "cookies": 50, "biscuits": 50, 
            "green_onion": 1, "soup": 30, '<pad>': 0, "rolls": 45, "honey": 30, "pineapple": 150, "vanilla": 5, "tofu": 50,
            "peppercorns": 1, "ginger": 1, "roast": 200, "sweet_potatoes": 150, "ice": 1, "bacon": 100,
            "cereal": 50, "cornstarch": 1, "cake_mix": 50, "coconut": 1, "breadcrumbs": 25, "bouillon": 30, "cocoa": 30,
            "sprouts": 30, "caramel": 8.2, "extract": 1, "wheat": 1, "sauce": 1, "barbecue_sauce": 1, 
            "juice": 5, "gelatin": 1, "beer": 30, "green_beans": 50, "baking_powder": 30, "liqueur": 1, "spaghetti_pasta": 50,
            "sake": 1, "wine": 1, "yogurt": 1, "jam": 1, "shortening": 1, "water": 50, "pudding_mix": 1, "tea": 1,
            "yeast": 1, "molasses": 1, "puree": 1, "fish_sauce": 1, "peanut_butter": 50, "coffee": 1,
            'horseradish': 35, "ice_cream": 100, "brandy_whiskey": 1, "brandy_whiskey": 1, "macaroni": 50, 
            "oatmeal": 50, "cornmeal": 50, "white_wine": 1, "apple_cider": 5, "applesauce": 5, "sour_cream": 35, 
            "preserves": 15, "mayonnaise": 5, 'baking_soda': 5, 'sugar': 15, 'salt': 1, 'food_coloring': 1}

        self.read_calorie_info_file(file_path)

    def read_calorie_info_file(self, file_path):

        self.ingredient_to_calorie = collections.defaultdict(lambda: collections.defaultdict(float))
        calorie_info = open(file_path)
        first_line = True
        for line in calorie_info:
            if first_line:
                first_line = False
                continue

            ingr, info = line.strip().split(':')
            split_info = info.split(',')
            calorie = float(split_info[-1])
            for portion_unit in split_info[:-1]:
                portion, unit = portion_unit.strip().split(" ")
                # because the calorie is given in a specific amount of a unit
                # it has to be converted to calorie per unit.
                calorie_per_unit = calorie/float(portion)
                self.ingredient_to_calorie[ingr][unit] = calorie_per_unit

    def show_calorie_ingredient_sorted(self):
        items = []
        for ingr in self.ingredient_to_calorie.keys():
            calorie_per_unit = self.ingredient_to_calorie[ingr]["gram"]
            items.append((ingr, calorie_per_unit))

        items.sort(key=lambda x: x[1], reverse=True)
        for item in items: 
            print ('{}: {}'.format(item[0], round(item[1],3)))

    def recipe_calorie_estimate(self, portions, units, ingrs, ingr_for_portion_computations):

        recipe_calorie = 0.0

        for p, u, i, ifp in zip(portions, units, ingrs, ingr_for_portion_computations):
            recipe_calorie += self.ingredient_calorie_estimation(p, u, i, ifp)

        return recipe_calorie

    def recipe_calorie_estimate_given_dict(self, ingrs_info):
        ''' Given a recipe with its info (ingredients & units) compute its gt calorie '''

        recipe_calorie = 0.0
        for qid in ingrs_info.keys():
            p, u, i, ifp = ingrs_info[qid]
            recipe_calorie += self.ingredient_calorie_estimation(p, u, i, ifp)

        return recipe_calorie

    def recipe_calorie_prior_error(self, ingrs_info, ingr_unit_portion_prior):
        ''' Given a prior (ingr,unit) portion pair compute the error of recipe calorie estimation '''

        estimated_calorie = 0.0
        gt_calorie = 0.0
        for qid in ingrs_info.keys():
            p, u, i, ifp = ingrs_info[qid]
            gt_calorie += self.ingredient_calorie_estimation(p, u, i, ifp)
            estimated_portion = ingr_unit_portion_prior[i][u]
            estimated_calorie += self.ingredient_calorie_estimation(estimated_portion, u, i, ifp)

        return abs(gt_calorie-estimated_calorie)

    def ingredient_calorie_estimation(self, portion, unit, ingr, ingr_for_portion_computation, show=False):

        if ingr in self.ingredient_to_calorie.keys():
            if unit in self.ingredient_to_calorie[ingr].keys():
                return self.ingredient_to_calorie[ingr][unit]*portion

        cp = ConvertPortions()
        new_portion, new_unit = cp.convert_portions(portion, unit, ingr_for_portion_computation)
        if new_unit == "None":
            # convert None to gram
            if ingr in self.None_to_gram.keys():
                new_portion = new_portion * self.None_to_gram[ingr]
                new_unit = 'gram'
            # convert None to milliliter
            elif ingr in self.None_to_milliliter.keys():
                new_portion = new_portion * self.None_to_milliliter[ingr]
                new_unit = 'milliliter'

        # convert it to gram
        if new_unit in {'ounce', 'milliliter'}:
            new_portion = new_portion * self.main_rules["{}_to_gram".format(new_unit)]
            new_unit = 'gram'

        if show:
            print (ingr, new_unit, new_portion, self.ingredient_to_calorie[ingr][new_unit]*new_portion)

        return self.ingredient_to_calorie[ingr][new_unit]*new_portion
    
# ------------------------------------

def open_fix_files(file_fixes_path):

    fixes = {}
    lines = open(file_fixes_path)
    for line in lines:
        if line.startswith("---"):
            fixes[r_id] = {"portion": portion, "unit": unit, "ingredient": ingredient}
            continue
        if line.startswith("portion"):
            portion = float(line.strip().split(":")[1])
        elif line.startswith("unit"):
            unit = line.strip().split(":")[1]
        elif line.startswith("ingredient"):
            ingredient = line.strip().split(":")[1]
        else:
            r_id = line.strip()

    return fixes

# ------------------------------------

def load_predicted_units_portions(portion_subsets, predicted_path="portion_classification_results/"):

    qid_to_unitportion = collections.defaultdict(lambda: collections.defaultdict(float))
    files = os.listdir(predicted_path)
    for f in files:
        print (f)
        _, _, unit_name = f.split(".")[0].split("_")

        lines = open(os.path.join(predicted_path, f))
        for line in lines:
            qid, portion_class = line.strip().split(':')
            portion_class = int(portion_class)
            portion_range = portion_subsets.class_to_portion_range(portion_class, unit_name)
            qid_to_unitportion[qid] = [unit_name, portion_range]

    return qid_to_unitportion

# ------------------------------------

def load_recipes_from_qa(a_path, q_path, recipe_to_ingrs, use_portion_class=True, id_to_main_ingrs=None, do_fix=True, processing_ingredients="all", load_predicted=False, file_fixes_path="data/fixes_train.txt", predicted_path="portion_classification_results/"):

    portion_subsets = PortionSubsets()
    if load_predicted:
        qid_to_unitportion = load_predicted_units_portions(portion_subsets, predicted_path)

    fixes = open_fix_files(file_fixes_path)
    q_file = json.load(open(q_path, 'r'))['questions']
    a_file = json.load(open(a_path, 'r'))['annotations']      

    for ann in a_file:
        r_id = ann["image_id"]
        q_id = ann["question_id"]
        
        # unit of measurement
        unit = ann["multiple_choice_answer"]

        if use_portion_class:
            
            # portion class
            portion_class = ann["portion_class"]
            # the amount of portion
            portion_range = portion_subsets.class_to_portion_range(portion_class, unit)
            # append portion range for this ingredient
            recipe_to_ingrs[r_id][q_id].append(portion_range)

        else:

            # portion value
            non_scaled_portion = ann["non_scaled_portion"]
            # append portion for this ingredient
            recipe_to_ingrs[r_id][q_id].append(non_scaled_portion)

        # unit for this ingredient
        recipe_to_ingrs[r_id][q_id].append(unit)

    ingr_to_freq = collections.defaultdict(int)
    for ann in q_file:
        r_id = ann["image_id"]
        q_id = ann["question_id"]
        
        # append ground-truth ingredient
        ingredient = ann["question"]
        ingredient_for_portion_computation = ann["question_conversion_type"]

        # fix some issues with the annotations
        if do_fix and r_id in fixes.keys():
            if ingredient==fixes[r_id]['ingredient']:
                fix_unit = fixes[r_id]['unit']
                fix_portion = fixes[r_id]['portion']
                
                # portion range fix
                portion = recipe_to_ingrs[r_id][q_id][0]
                if isinstance(portion, float):
                    recipe_to_ingrs[r_id][q_id][0] = fix_portion
                else:
                    a, _, b = portion
                    recipe_to_ingrs[r_id][q_id][0] = [a, fix_portion, b]                    

                # portion range unit
                recipe_to_ingrs[r_id][q_id][1] = fix_unit

        # the ingredient is not a main ingredient do not process it.
        if processing_ingredients=="main" and (not ingredient in id_to_main_ingrs[r_id]):
            continue 

        ingr_to_freq[ingredient] += 1
        recipe_to_ingrs[r_id][q_id].append(ingredient)
        recipe_to_ingrs[r_id][q_id].append(ingredient_for_portion_computation)

        if not load_predicted:
            continue

        # append predicted unit & portion
        if q_id in qid_to_unitportion.keys():
            predicted_unit, predicted_portion = qid_to_unitportion[q_id] 
            recipe_to_ingrs[r_id][q_id].append(predicted_unit)
            recipe_to_ingrs[r_id][q_id].append(predicted_portion)


    return recipe_to_ingrs, ingr_to_freq

# ------------------------------------

def compute_gt_calories(recipe_to_ingrs):
    ''' For a given dictionary of recipe and its info compute calorie '''

    recipe_to_calorie = {}
    calorie_info = CalorieInfo()

    for rid in recipe_to_ingrs.keys():

        recipe_to_calorie[rid] = calorie_info.recipe_calorie_estimate_given_dict(recipe_to_ingrs[rid])

    return recipe_to_calorie

# ------------------------------------

def compute_portion_errors(recipe_to_ingrs, ingr_unit_portion_prior, prec=2):
    ''' 
        For a given dictionary of recipe and its info compute portion estimation error 
            1. computes total mean portion error
            2. computes total mean calorie error for ingredients based on portion
            3. computes total mean portion error for each ingredient based on prior portion estimation
            4. computes total mean portion error for each unit based on prior portion estimation
            5. computes total mean calorie error for each ingredient based on prior portion estimation
            6. computes total mean calorie error for each unit based on prior portion estimation
    '''
    
    ingr_portion_err = collections.defaultdict(float)
    unit_portion_err = collections.defaultdict(float)
    ingr_calorie_err = collections.defaultdict(float)
    unit_calorie_err = collections.defaultdict(float)
    ingr_count = collections.defaultdict(float)
    unit_count = collections.defaultdict(float)

    portion_errors = {}
    portion_calorie_errors = {}
    calorie_info = CalorieInfo()
    total = 0
    for rid in recipe_to_ingrs.keys():
        for qid in recipe_to_ingrs[rid].keys():
            portion, u, i, ifp = recipe_to_ingrs[rid][qid]

            # ground-truth
            gt_portion = portion
            gt_calorie = calorie_info.ingredient_calorie_estimation(portion, u, i, ifp)

            # estimated
            estimated_portion = ingr_unit_portion_prior[i][u]
            estimated_calorie = calorie_info.ingredient_calorie_estimation(estimated_portion, u, i, ifp)

            cur_portion_err = abs(gt_portion-estimated_portion)
            cur_calorie_err = abs(gt_calorie-estimated_calorie)

            portion_errors[qid] = cur_portion_err
            portion_calorie_errors[qid] = cur_calorie_err

            ingr_portion_err[i] += cur_portion_err
            unit_portion_err[u] += cur_portion_err

            ingr_calorie_err[i] += cur_calorie_err
            unit_calorie_err[u] += cur_calorie_err

            ingr_count[i] += 1
            unit_count[u] += 1

            total += 1

    mean_portion_error = round(sum(portion_errors.values())/len(portion_errors.values()), prec)
    mean_calorie_error = round(sum(portion_calorie_errors.values())/len(portion_calorie_errors.values()), prec)

    # computing portion (prior) estimation error for an ingredient
    for ingr in ingr_portion_err.keys():
        ingr_portion_err[ingr] = round(ingr_portion_err[ingr] / ingr_count[ingr], prec)

    # computing portion (prior) estimation error for a unit
    for unit in unit_portion_err.keys():
        unit_portion_err[unit] = round(unit_portion_err[unit] / unit_count[unit], prec)

    # computing calorie (prior) estimation error for an ingredient
    for ingr in ingr_calorie_err.keys():
        ingr_calorie_err[ingr] = round(ingr_calorie_err[ingr] / ingr_count[ingr], prec)

    # computing calorie (prior) estimation error for a unit
    for unit in unit_calorie_err.keys():
        unit_calorie_err[unit] = round(unit_calorie_err[unit] / unit_count[unit], prec)

    return mean_portion_error, mean_calorie_error, ingr_portion_err, unit_portion_err, ingr_count, unit_count, ingr_calorie_err, unit_calorie_err, total


# ------------------------------------

def compute_calorie_errors(recipe_to_ingrs, ingr_unit_portion_prior, prec=2):
    ''' For a given dictionary of recipe and its info compute calorie estimation error '''
    
    recipe_errors = {}
    calorie_info = CalorieInfo()
    for rid in recipe_to_ingrs.keys():

        recipe_errors[rid] = calorie_info.recipe_calorie_prior_error(recipe_to_ingrs[rid], ingr_unit_portion_prior)

    mean_error = round(sum(recipe_errors.values())/len(recipe_errors.values()), prec)
    return mean_error, recipe_errors

# ------------------------------------

def load_gt_calorie(split, root="features"):
    ''' load gt calorie for a split of the dataset '''

    q_path = os.path.join(root, "recipe_{}_questions.json".format(split))
    a_path = os.path.join(root, "recipe_{}_answers.json".format(split))
    recipe_to_ingrs = collections.defaultdict(lambda: collections.defaultdict(list))
    recipe_to_ingrs, _ = load_recipes_from_qa(a_path, q_path, recipe_to_ingrs, use_portion_class=False, do_fix=True, file_fixes_path="data/fixes_train.txt")
    recipe_to_calorie = compute_gt_calories(recipe_to_ingrs)

    return recipe_to_calorie, recipe_to_ingrs

# ------------------------------------

class RecipeData:
    ''' 
        class that keeps track of paths of recipe dataset
            1. questions & answers
            2. calorie dataset
            3. dish dataset
    '''

    def __init__(self, qa_root="features", cal_root="dish_dataset", output="purified_data/filtered"):

        self.qa_root = qa_root
        self.cal_root = cal_root
        
        if not os.path.exists(output):
            os.mkdir(output)
        self.output = output

        self.init()

    def init(self):

        self.ques_paths = {}
        self.answ_paths = {}
        self.calo_paths = {}
        self.dish_paths = {}

        self.out_ques_paths = {}
        self.out_answ_paths = {}
        self.out_calo_paths = {}
        self.out_dish_paths = {}

        self.init_paths('train')
        self.init_paths('test')
        self.init_paths('val')

        self.open_files('train')
        self.open_files('test')
        self.open_files('val')

    def init_paths(self, split):
      
        self.ques_paths[split] = os.path.join(self.qa_root, "recipe_{}_questions.json".format(split))
        self.answ_paths[split] = os.path.join(self.qa_root, "recipe_{}_answers.json".format(split))
        self.calo_paths[split] = os.path.join(self.cal_root, "calorie_{}.json".format(split))
        self.dish_paths[split] = os.path.join(self.cal_root, "{}.pkl".format(split))

        self.update_output_paths(self.output)

    def update_output_paths(self, output):

        self.output = output
        self.init_out_paths('train')
        self.init_out_paths('test')
        self.init_out_paths('val')

    def init_out_paths(self, split):

        self.out_ques_paths[split] = os.path.join(self.output, "recipe_{}_questions.json".format(split))
        self.out_answ_paths[split] = os.path.join(self.output, "recipe_{}_answers.json".format(split))
        self.out_calo_paths[split] = os.path.join(self.output, "calorie_{}.json".format(split))
        self.out_dish_paths[split] = os.path.join(self.output, "{}.pkl".format(split))

    def open_files(self, split):

        self.questions =  json.load(open(self.ques_paths[split], 'r'))
        self.answers   =  json.load(open(self.answ_paths[split], 'r'))
        #self.calories = pickle.load(open(self.calo_paths[split], 'rb'))
        self.dishes   = pickle.load(open(self.dish_paths[split], 'rb'))

    def create_filtered_datasets(self, recipe_infos_train, recipe_infos_val, recipe_infos_test, output=""):

        if output!="":
            self.update_output_paths(output)

        self.create_filtered('train', recipe_infos_train)
        self.create_filtered('val', recipe_infos_val)
        self.create_filtered('test', recipe_infos_test)

    def create_filtered(self, split, recipe_infos):

        calorie_info = CalorieInfo()

        # filter questions and store
        out_questions = {}
        questions = json.load(open(self.ques_paths[split], 'r'))
        for key in questions.keys():
            if key!="questions":
                out_questions[key] = questions[key]

        qid_to_ques = {}
        filtered_ques = []
        for item in questions['questions']:
            if item['image_id'] in recipe_infos.keys():
                filtered_ques.append(item)
                qid_to_ques[item["question_id"]] = (item["question"], item["question_conversion_type"])

        out_questions['questions'] = filtered_ques

        with open(self.out_ques_paths[split], 'w') as outfile:
            json.dump(out_questions, outfile)

        # filter answers and store
        out_answers = {}
        answers = json.load(open(self.answ_paths[split], 'r'))
        for key in answers.keys():
            if key!="annotations":
                out_answers[key] = answers[key]

        filtered_annots = []
        for item in answers['annotations']:
            if item['image_id'] in recipe_infos.keys():
                filtered_annots.append(item)
                p = item["non_scaled_portion"]
                u = item["multiple_choice_answer"]
                i, ifp = qid_to_ques[item["question_id"]]
                item['calorie'] = calorie_info.ingredient_calorie_estimation(p, u, i, ifp)

        out_answers['annotations'] = filtered_annots

        with open(self.out_answ_paths[split], 'w') as outfile:
            json.dump(out_answers, outfile)

        # filter dishes and store
        dish_data = pickle.load(open(self.dish_paths[split], 'rb'))

        filtered_dishes = {}
        co = 0
        sco = 0
        for rid in dish_data.keys():
            co += 1
            if rid in recipe_infos.keys():
                filtered_dishes[rid] = dish_data[rid]
                sco += 1

        pickle.dump(filtered_dishes, open(self.out_dish_paths[split], 'wb'))

        # compute new calories and store
        recipe_to_calorie = compute_gt_calories(recipe_infos)

        with open(self.out_calo_paths[split], 'w') as outfile:
            json.dump(recipe_to_calorie, outfile)

# ------------------------------------

class PortionSubsets:

    def __init__(self):

        self.seasoning_portions = {'dash', 'pinch', 'drop'}
        self.low_frequency_portions = {'bottle', 'box', 'bowl', 'scoop', 'gallon', 'bag', 
                                       'can', 'handful', 'drop', 'package', 'pint', 'liter', 
                                       'quart', 'kilogram', 'bunch', 'milliliter', 'pinch',
                                       'dash', 'slice', 'jar', 'carton', 'packet', 'container'}

        self.all_units = {'gram', 'kilogram', 'pound', 'liter', 'gallon', 'quart', 'cup',
                          'pint', 'bowl', 'can', 'handful', 'bag', 'scoop', 'tablespoon', 
                          'teaspoon', 'box', 'package', 'dash', 'pinch', 'drop', 'bunch', 
                          'slice', 'bottle', 'milliliter', 'ounce', 'None', 'jar', 
                          'carton', 'packet', 'container'}

        self.round_ups = {  'tablespoon': {4: 0.5, 16: 1}, # up to 4 round up in 0.5 ranges, ...
                            'teaspoon': {2: 0.25, 4: 0.5}, 
                            'None': {4: 0.5, 24: 1}, 
                            'cup': {2: 0.25, 6: 0.5},
                            'ounce': {4: 0.5, 16: 1, 48: 2}, 
                            'pound': {2: 0.25, 4: 0.5, 8: 1},
                            'gram': {50: 5, 100: 10, 500: 25}}

        self.max_limits = {'tablespoon': 12,
                            'teaspoon': 5, 
                            'None': 20,
                            'cup': 7,
                            'ounce': 40,
                            'pound': 8,
                            'gram': 480}

        self.class_centroids = { 'tablespoon': np.asarray([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]),
                                 'teaspoon': np.asarray([0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]), 
                                 'None': np.asarray([0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0]), 
                                 'cup': np.asarray([0.25, 0.334, 0.5, 0.667, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]), 
                                 'ounce': np.asarray([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 14.0, 16.0, 20.0, 32.0]), 
                                 'pound': np.asarray([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]), 
                                 'gram': np.asarray([10, 20, 30, 40, 50, 60, 70, 80, 100, 125, 150, 200, 250, 300, 375, 450])}

        self.min_limits = {'tablespoon': 0.25,
                            'teaspoon': 0.1,
                            'None': 0.15,
                            'cup': 0.15,
                            'ounce': 0.25,
                            'pound': 0.15,
                            'gram': 1}

    def is_out_of_bands(self, unit, portion):
        if unit in self.seasoning_portions or unit in self.low_frequency_portions:
            return True
        if portion<self.min_limits[unit]:
            return True
        elif portion>self.max_limits[unit]:
            return True
        return False

    def round_up_or_classification(self, portion, unit, do_round_up):
        if self.is_out_of_bands(unit, portion):
            return False, -1, -1

        if do_round_up==1:
            cont, por = self.round_up(portion, unit)
            return cont, por, -1
        elif do_round_up==2:
            por_as_class = self.classification(portion, unit)
            return True, portion, int(por_as_class)

        return True, portion, -1

    def classification(self, portion, unit):
        centroids = self.class_centroids[unit]
        distances = np.abs(portion-centroids)
        class_indx = list(np.where(distances==np.min(distances))[0])[-1]
        return class_indx

    def class_to_portion_range(self, class_indx, unit):
        centroids = self.class_centroids[unit]
        st = class_indx-1
        en = class_indx+1
        if class_indx==0:
            return [0, centroids[0], (centroids[0]+centroids[1])/2]
        if class_indx==(len(centroids)-1):
            r = (centroids[class_indx]-centroids[class_indx-1])/2
            return [centroids[class_indx]-r, centroids[class_indx], centroids[class_indx]+r]
        else:
            ra = (centroids[class_indx]+centroids[class_indx-1])/2
            rb = (centroids[class_indx]+centroids[class_indx+1])/2
            return [ra, centroids[class_indx], rb]

    def round_up(self, portion, unit):
       
        round_up_threshs = self.round_ups[unit]
        sorted_keys = sorted(list(round_up_threshs.keys()))
        for thresh in sorted_keys:
            if portion<=thresh:
                divisor = (portion/round_up_threshs[thresh])
                rounded_portion = int(divisor) * round_up_threshs[thresh]
                rounded_portion += round(divisor-int(divisor)) * round_up_threshs[thresh]
                break

        if rounded_portion==0:
            rounded_portion = round_up_threshs[sorted_keys[0]]

        rounded_portion = float(rounded_portion)
        print (unit, portion, rounded_portion)

        return True, rounded_portion

# ------------------------------------

class ConvertPortions:
    def __init__(self):

        # maps units from the 27 various units to 3 gobal units (milli, ounce, count)
        self.global_unit_mapping = collections.defaultdict(lambda: set())

        self.convert = {'gram': ('ounce', 0.035274), 'kilogram': ('ounce', 35.274), 'pound': ('ounce', 16),
               'liter': ('milliliter', 1000), 'gallon': ('milliliter', 3785.41),
               'pint': ('milliliter', 473.176), 'quart': ('milliliter', 946.353),
               'cup': ('ounce', 8), 'bowl': ('ounce', 12), 'can': ('ounce', 12), 'handful': ('ounce', 1),
               'teaspoon': ('milliliter', 4.92892), 'scoop': ('milliliter', 29.5736), 
               'tablespoon': ('milliliter', 14.7868), 'bag': ('ounce', 1),
               'box': ('ounce', 3.4), 'jar': ('ounce', 8),
               'package': ('ounce', 3.4), 'dash': ('ounce', 0.03125), 
               'pinch': ('ounce', 0.01), 'drop': ('ounce', 0.0016907), 'bunch': ('ounce', 5), 
               'slice': ('ounce', 1.7), 'bottle': ('ounce', 16)}

        self.indirect_unit_translations = {'slice', 'bunch', 'carton', 'box', 'bag', 'bottle', 'package', 'cup'}
        self.units = list(self.convert.keys())
        self.units.extend(['None', 'ounce', 'milliliter'])
        self.units_length = len(self.units)

        self.main_rules = {"ounce_to_milliliter": 29.5735, "milliliter_to_ounce": 0.033814}

    def align(self, gt_portion, gt_unit, pr_portion, pr_unit):
        if not gt_unit==pr_unit and (pr_unit+"_to_"+gt_unit) in self.main_rules.keys():
            pr_portion = self.main_rules[pr_unit+"_to_"+gt_unit]*pr_portion

        return pr_portion, gt_unit

    def fill_statistics(self, min_dict, max_dict):
        
        self.min_dict = min_dict
        self.max_dict = max_dict

    def unnormalized_conversion(self, portion, unit, ingredient):
        
        dmin = self.min_dict[unit]
        dmax = self.max_dict[unit]
        unnormalized_portion = portion * (dmax-dmin) + dmin

        global_portion, global_unit = self.convert_portions(unnormalized_portion, unit, ingredient)
        return global_portion, global_unit

    def update(self, ans, global_unit):

        self.global_unit_mapping[ans].add(global_unit)

    def print(self):

        print ("Unit mappings:")
        for key in self.global_unit_mapping.keys():
            if len(self.global_unit_mapping[key])>1:
                print (colored(" {}: {}".format(key, self.global_unit_mapping[key]),'red'))
            else:
                print (" {}: {}".format(key, self.global_unit_mapping[key]))

    # given (portion,unit) ground-truth values for an ingredient (lbl) compute the global portion amount.
    def convert_portions(self, portion, unit, lbl):

        processed = False
        if unit in self.indirect_unit_translations:
            if unit=='slice':
                # slice
                new_unit, target_coef, processed = self.convert_slice(self.convert, lbl)
            elif unit=='bunch':
                # bunch
                new_unit, target_coef, processed = self.convert_bunch(self.convert, lbl)

            elif unit=='carton':
                # carton
                new_unit, target_coef, processed = self.convert_carton(self.convert, lbl)

            elif unit=='box':
                # box
                new_unit, target_coef, processed = self.convert_box(self.convert, lbl)

            elif unit=='bag':
                # bag is usually different kinds of tea-bag
                new_unit, target_coef, processed = self.convert_bag(self.convert, lbl)

            elif unit=='bottle':
                # bottle
                new_unit, target_coef, processed = self.convert_bottle(self.convert, lbl)

            elif unit=='package':
                # package
                if 'cake_mix' in lbl:
                    new_unit = 'ounce'
                    target_coef = 15
                    processed = True

            elif unit=='cup':
                if 'marshmallow' in lbl:
                    new_unit = 'gram'
                    target_coef = 50
                    processed = True

        if unit in self.convert.keys():
            if not processed:
                new_unit, target_coef = self.convert[unit]
            portion = round(target_coef*portion,2)
        else:
            portion = round(portion,2)
            new_unit = unit

        # just to find all kinfs of mapping between variety of units & global units.
        self.update(portion, new_unit)
                
        return portion, new_unit

    def convert_slice(self, convert, lbl):
        
        if 'bacon_bits' in lbl:
            return 'ounce', 0.45, True
        elif 'kamaboko' in lbl or 'taleggio' in lbl or 'poundcake_mix' in lbl or 'raw_zucchini' in lbl or 'eggplant' in lbl or 'white_cake_mix' in lbl or 'fontina' in lbl or "medium_cheddar" in lbl or '_free_italian_salad_dressing' in lbl or 'american_cheese' in lbl or 'cucumber' in lbl or 'lime' in lbl or 'smoked_salmon' in lbl or 'butter' in lbl:
            return 'ounce', 1, True
        # these are all ham, beef, turkey slices of meat.
        elif "<pad>" in lbl or "mortadella" in lbl or 'pastrami' in lbl or "pork_tenderloin" in lbl or "lean_meat" in lbl or "lean_ground_beef" in lbl or "capicola" in lbl or 'pork_chop' in lbl or 'salami' in lbl or 'deli_ham' in lbl or 'turkey' in lbl or 'prosciutto' in lbl or "pancetta" in lbl or 'light_bologna' in lbl:
            return 'ounce', 2, True
        elif 'tomato' in lbl or 'avocado' in lbl or 'peach_slices' in lbl or 'anjou_pear' in lbl:
            return 'ounce', 0.7, True
        elif 'onion' in lbl or 'carrot' in lbl or 'lime_pickle' in lbl or 'potato' in lbl:
            return 'ounce', 0.6, True
        elif 'pepperoni' in lbl or 'jalapeno' in lbl or 'sweet_italian_sausage' in lbl or 'banana' in lbl:
            return 'ounce', 0.42, True
        elif 'olive_oil_flavored_cooking_spray' in lbl or 'ground_ginger' in lbl or "gingerroot" in lbl or 'galangal' in lbl or 'fresh_garlic' in lbl:
            return 'ounce', 0.26, True
        elif 'french_baguette' in lbl:
            return 'ounce', 1.5, True
        elif 'leaf_lettuce' in lbl or 'shiso_leaves' in lbl or "green_bell_pepper" in lbl or "strawberries" in lbl or 'chili_powder' in lbl:
            return 'ounce', 0.5, True
        elif 'chicken_thighs' in lbl or 'chicken_breast_fillets' in lbl:
            return 'ounce', 4, True
        elif 'watermelon' in lbl:
            return 'ounce', 4.85, True
        elif 'steak' in lbl or 'loin_lamb' in lbl:
            return 'ounce', 24, True

        return 'ounce', 1, False

    def convert_bunch(self, convert, lbl):

        if 'asparagus' in lbl or 'broccolini' in lbl or 'frozen_french_-_cut_green_beans' in lbl:
            return 'ounce', 17.5, True
        elif 'spinach' in lbl:
            return 'ounce', 12.35, True
        elif 'mitsuba' in lbl or 'cilantro' in lbl or 'mizuna' in lbl or 'coriander' in lbl or 'dry_dill_weed' in lbl or 'watercress' in lbl or 'dried_fenugreek_leaves' in lbl or 'dried_marjoram' in lbl:
            return 'ounce', 2.8, True
        elif 'dried_parsley' in lbl or 'fresh_chives' in lbl:
            return 'ounce', 2, True
        elif 'escarole' in lbl or 'frisee' in lbl or 'chard_leaves' in lbl or 'black_kale' in lbl or "leaf_lettuce" in lbl or 'swiss_chard' in lbl or 'mixed_salad_greens' in lbl: # only considering leaf & not the stem (with stem is ~ 6.5)
            return 'ounce', 4.0, True
        elif 'frozen_chopped_broccoli' in lbl:
            return 'ounce', 9, True
        elif 'rocket' in lbl or 'arugula' in lbl:
            return 'ounce', 7.5, True
        elif 'scallion' in lbl or 'red_onion' in lbl or 'ramps' in lbl:
            return 'ounce', 3.8, True
        elif 'fresh_basil' in lbl:
            return 'ounce', 2.5, True
        elif 'of_fresh_mint' in lbl:
            return 'ounce', 1.7, True
        elif 'thyme' in lbl or 'dried_rosemary' in lbl or 'dried_sage' in lbl or 'oregano' in lbl or 'fresh_tarragon' in lbl:
            return 'ounce', 1, True
        elif 'celery' in lbl or 'carrot' in lbl or 'bok_choy' in lbl:
            return 'ounce', 16, True
        elif 'radish' in lbl:
            return 'ounce', 10, True
        elif 'leek' in lbl:
            return 'ounce', 27, True
        elif 'portabella_mushroom_caps' in lbl or 'dried_herbs' in lbl or 'dried_chamomile' in lbl or 'lavender' in lbl:
            return 'ounce', 1.2, True
        elif 'fettuccine_pasta' in lbl or 'spaghetti_sauce' in lbl or 'grapes' in lbl or 'tomato_sauce' in lbl:
            return 'ounce', 8, True

        return 'ounce', 1, False

    def convert_carton(self, convert, lbl):
        # carton is usually milk, water, sour cream, etc
        if 'berry' in lbl or 'berries' in lbl:
            return 'ounce', 6, True
        elif 'oven_cooking' in lbl or 'zip' in lbl: # the number just counts
            return 'None', 1, True
        elif 'marshmallow' in lbl:
            return 'ounce', 6, True

        return 'ounce', 1, False

    def convert_box(self, convert, lbl):

        if 'berry' in lbl or 'berries' in lbl:
            return 'ounce', 6, True
        if 'macaroni' in lbl or 'pasta' in lbl or 'penne' in lbl or 'noodle' in lbl or 'spaghetti' in lbl or 'manicotti' in lbl or 'linguine' in lbl or 'ditalini' in lbl:
            return 'ounce', 12, True
        elif 'cheese' in lbl:
            return 'ounce', 3, True
        elif 'cracker' in lbl or 'pretzels' in lbl:
            return 'ounce', 16, True
        elif 'puff_pastry' in lbl:
            return 'ounce', 17, True
        elif 'pie_crusts' in lbl:
            return 'ounce', 14, True
        elif 'spinach' in lbl or 'greens' in lbl or 'lettuce' in lbl:
            return 'ounce', 10, True
        elif 'pectin' in lbl:
            return 'ounce', 1.75, True
        elif 'stock' in lbl or 'broth' in lbl:
            return 'ounce', 32, True
        elif 'corn_flakes' in lbl or 'cereal' in lbl:
            return 'ounce', 12, True
        elif 'roux' in lbl:
            return 'ounce', 10, True
        elif 'egg' in lbl:
            return 'ounce', 16, True
        elif 'rice' in lbl:
            return 'ounce', 7.4, True
        elif 'breadstick' in lbl:
            return 'ounce', 4.4, True
        elif 'sugar' in lbl or 'water' in lbl or 'ice_cube' in lbl: # as needed
            return 'ounce', 0, True

        return 'ounce', 1, False

    def convert_bag(self, convert, lbl):

        if 'oven_cooking' in lbl or 'zip' in lbl: # the number just counts
            return 'None', 1, True
        elif 'marshmallow' in lbl:
            return 'ounce', 6, True
        if 'macaroni' in lbl or 'pasta' in lbl or 'penne' in lbl or 'noodle' in lbl or 'spaghetti' in lbl or 'manicotti' in lbl or 'linguine' in lbl or 'ditalini' in lbl:
            return 'ounce', 12, True
        elif 'cheese' in lbl:
            return 'ounce', 3, True
        elif 'spinach' in lbl or 'greens' in lbl or 'lettuce' in lbl:
            return 'ounce', 10, True
        elif 'rice' in lbl:
            return 'ounce', 7.4, True
        elif 'sugar' in lbl or 'water' in lbl or 'ice_cube' in lbl: # as needed
            return 'ounce', 0, True

        return 'ounce', 1, False

    def convert_bottle(self, convert, lbl):

        if 'wine' in lbl or 'champagne' in lbl or 'vodka' in lbl or 'prosecco' in lbl or 'rum' in lbl:
            return 'ounce', 25.4, True
        elif 'beer' in lbl:
            return 'ounce', 12, True
        elif 'club_soda' in lbl or 'ginger_ale' in lbl:
            return 'ounce', 12, True
        elif 'barbecue_sauce' in lbl:
            return 'ounce', 16, True
        elif 'sauce' in lbl or 'ketchup' in lbl:
            return 'ounce', 20, True
        elif 'carbonated_lemon_' in lbl:
            return 'ounce', 16.9, True
        elif 'dressing' in lbl:
            return 'ounce', 16, True
        elif 'spray_bottle' in lbl:
            return 'ounce', 1, True
        elif 'cola' in lbl:
            return 'ounce', 16, True
        else:
            return 'ounce', 16, True

        return 'ounce', 1, False

# ------------------------------------
