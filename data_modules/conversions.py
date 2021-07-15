# Copyright (c) Robotic and Action Perception Lab (RPAL) at University of South Florida
# by Ahmad Babaeian Jelodar

'''
    This file creates accurate conversions of calories for ingredients using 
    different unit and portion values.
'''

import os
import numpy as np
import sys, json
import collections, pickle

# ------------------------------------

class Conversions:

    def __init__(self):

        self.unavailables = set()
        self.initialize()

    def check_ingr_avail(self, ingr):

        if ingr in self.None_to_gram.keys():
            return True

        if ingr in self.None_to_milliliter.keys():
            return True

        self.unavailables.add(ingr)
        return False

    def initialize(self):

        # what is the precision of the portions needed in each unit type
        self.precisions = {'ounce': 6, 'pound': 8, 'teaspoon': 4, 'tablespoon': 4, 'none': 6, 'cup': 6}

        self.conversions = {'ounce': {"pound": 0.0625, "ounce": 1.0, "gram": 28.3495, "milliliter": 29.5735},
                            'pound': {"ounce": 16.0, "pound": 1.0},
                            'teaspoon': {"tablespoon": 0.333333, "teaspoon": 1.0},
                            'tablespoon': {"teaspoon": 3.0, "tablespoon": 1.0},
                            'gram': {"ounce": 0.035274}, 'milliliter': {"ounce": 0.033814},
                            'none': {"none": 1.0}, 'cup': {"cup": 1.0}}

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

        self.cup_to_ounce = {"chicken": 6, "egg": 7, "ribs": 7, "potatoes": 5.3, "avocado": 5.25, "steaks": 5.75,
            "mushrooms": 4.4, "eggplant": 2.9, "sausage": 3.95, "bread": 3.2, "olive_oil": 4.6,
            "fish": 8, "fillets": 8, "shrimp": 10, "tortillas": 2.2, 
            "rosemary": 1.86, "thyme": 1.5, "basil": 0.7, "mint": 1.6, "sage": 2.0,
            "cooking_spray": 8, "margarine": 7.6, "cilantro": 0.56,
            "coriander": 0.56, "noodles": 8, "syrup": 12, "oregano": 1.8, "dill": 1.8, "beef": 5.3,
            "meat": 10, "rice": 7, "salad_dressing": 8, "cinnamon": 4.7, "beans": 6.4,
            'ketchup': 8, 'herbs': 0.75, 'turmeric': 5.6, 'curry': 5.6, 'mustard': 8, 'cumin': 3.4,
            'soy_sauce': 6, 'icing': 4.4, 'hot_sauce': 8, 'ham': 6,
            'milk': 8, 'pasta': 8, 'cardamom': 3, 'lamb': 5.3, 'oil': 8.1, 'oysters': 8, 
            'vinegar': 8, 'tarragon': 1.1, 'tuna': 8, 'turkey': 5, 'seasoning': 4.7, 'marjoram': 1, 
            'worcestershire_sauce': 6, 'salsa': 6, 'splenda_sweetener': 8, "lentils": 7.1,
            'crabmeat': 5.3, 'oats': 3.6, "jalapeno": 3, "sweet_peppers": 3, "lime": 8.5, 
            "lemon_juice": 8, "butter": 8, "pork": 8, "salmon": 8, "cream": 8, "celery": 3.56,
            "allspice": 3.4, "parsley": 2, "olives": 6.35, "paprika": 1, "flour": 8, "whipped_topping": 4.2,
            "chickpeas": 6, "cheese": 6, "pepper_chili": 3.95, "chili_pepper": 3.95, "tomatoes": 7, 
            "carrots": 3.9, "orange": 8.5, "apple": 3.85, "banana": 8.5, "zucchini": 4.4, "strawberries": 7.1, 
            "mango": 5.8, "lemon": 8, "onion": 4, "peach": 8, "beets": 6, "pear": 5.8,  "cucumber": 4.23, 
            "squash": 4.3, "potatoes_sweet": 4.7, "apricot": 6.7, "fruit": 6, 
            "pumpkin": 8, "dates": 6.2, "radishes": 4.1, "garlic": 4.8, "berries": 5.86, 
            "grapes": 6, "peas": 5,  "blueberries": 5.86, "cranberries": 4.23, "raspberries": 6, 
            "capers": 5.36, "cherries": 5, "bay": 1.07, "lettuce": 2.6, "broccoli": 6.2, "spinach": 0.95, 
            "cabbage": 3.14, "chives": 1.7, "greens": 2.6, "arugula": 0.71, "asparagus": 6.18,
            "cauliflower": 2.26, "fennel": 3, "artichoke": 5.93, "leek": 3.14, "peanuts": 5.29,  "nuts": 5.29, 
            "seeds": 4.7, "corn": 5.82, "chocolate_chips": 6, "almonds": 5.2, "chestnuts": 5.47, 
            "cashews": 5.3, "pecans": 4, "walnuts": 4.1, "raisins": 5.3, "nutmeg": 4.17, 
            "chocolate": 6, "crackers": 4, "marshmallows": 1.76, "honey": 12, "pineapple": 6,
            "cookies": 3.5, "biscuits": 3.5, "green_onion": 8, "soup": 8, '<pad>': 0, 
            "vanilla": 5.3, "tofu": 8.9, "peppercorns": 4.74, "ginger": 4, 
            "sweet_potatoes": 4.7, "ice": 5.29, "bacon": 8, "cereal": 3.6, "bouillon": 8.75,
            "cornstarch": 4.52, "cake_mix": 6, "coconut": 3.35,  "breadcrumbs": 3.81, "cocoa": 3.5,
            "sprouts": 3.67, "caramel": 8, "extract": 8, "wheat": 3.6, "sauce": 7, "barbecue_sauce": 7,
            "juice": 8, "gelatin": 5.22, "beer": 8, "green_beans": 5.3, "baking_powder": 8, "liqueur": 8, 
            "spaghetti_pasta": 6, "sake": 8, "wine": 8, "yogurt": 8, "jam": 8, "shortening": 7.23, "water": 8,
            "pudding_mix": 6, "tea": 8,  "yeast": 4.8, "molasses": 8, "puree": 9, "fish_sauce": 8, 
            "peanut_butter": 8, "coffee": 8, 'horseradish': 8.47, "ice_cream": 5.29, 
            "brandy_whiskey": 8, "macaroni": 8, "sour_cream": 8.6, "mayonnaise": 8.11,
            "oatmeal": 3.17, "cornmeal": 4.9, "white_wine": 8, "apple_cider": 8.5, "applesauce": 8, 
            'salt': 9.6, 'food_coloring': 8, "preserves": 11, 'baking_soda': 8.1, 'sugar': 7.1, 
            "pie_crust": 4.4, "crust_pie": 4.4, "pie_shell": 4.4, "shell_pie": 4.4, "tartar_cream": 5.7,
            "rolls": 3.5, "roast": 8, "vegetables": 8, "chips": 0.92}

        # ------------------------------
        # ounce to teaspoon or tablespoon
        # set1
        self.set0 = {"rolls", "bread", "olives", "tortillas", "chips", "cinnamon", "beans", "curry", "paprika", "seasoning", "allspice", "celery", "flour", "apple", "strawberries", "pumpkin", "radishes", "peas", "capers", "asparagus", "seeds", "crackers", "biscuits", "cornstarch", "breadcrumbs", "cocoa", "yeast", "cornmeal"}
        self.set1 = {"potatoes", "avocado", "turmeric", "cumin", "mango", "peach", "beets", "pear", "squash", "potatoes_sweet", "sweet_potatoes", "apricot", "fruit", "peanuts", "nuts", "chocolate_chips", "chestnuts" "cashews", "pecans", "walnuts", "raisins", "chocolate", "gelatin", "ice_cream"}
        self.set2 = {"orange", "carrots", "eggplant", "cardamom", "jalapeno", "sweet_peppers", "pepper_chili", "chili_pepper", "cabbage", "oatmeal"}
        self.set3 = {"herbs", "bay", "parsley", "marjoram", "tarragon", "rosemary", "thyme", "basil", "mint", "sage", "coriander", "oregano", "dill", "onion", "lettuce", "spinach", "greens", "chives", "arugula", "cauliflower", "marshmallows", "green_onion", "sprouts"}
        self.set4 = {"noodles", "almonds", "broccoli"}
        self.set5 = {"syrup"}
        self.set6 = {"rice", "spaghetti_pasta", "pasta", "tomatoes", "lentils", "splenda_sweetener", "icing", "sugar", "shortening"}
        self.set7 = {"soy_sauce", "worcestershire_sauce", "sauce", "barbecue_sauce", "salsa", "artichoke", "tofu", "apple_cider", "applesauce", "salt"}
        self.set8 = {"whipped_topping", "sour_cream", "cream", "chickpeas", "dates", "berries", "blueberries", "cranberries", "raspberries", "leek", "corn", "pineapple", "tartar_cream"}
        self.set9 = {"cucumber", "garlic", "grapes", "cherries", "fennel", "nutmeg","peppercorns", "ginger", "bacon", "cereal", "oats", "wheat", "bouillon"}
        self.set10 = {"honey", "caramel", "jam", "molasses", "preserves"}

        self.teaspoon_to_ounce = {}
        self.tablespoon_to_ounce = {}
        for ingr in self.cup_to_ounce.keys():
            # default teaspoon-to-ounce & tablespoon-to-ounce values.
            self.teaspoon_to_ounce[ingr] = 0.166667
            self.tablespoon_to_ounce[ingr] = 0.5
            if ingr in self.set0:
                self.teaspoon_to_ounce[ingr] = 0.09333333333
                self.tablespoon_to_ounce[ingr] = 0.28
            elif ingr in self.set1:
                self.teaspoon_to_ounce[ingr] = 0.11111111111
                self.tablespoon_to_ounce[ingr] = 0.33333333333
            elif ingr in self.set2:
                self.teaspoon_to_ounce[ingr] = 0.06
                self.tablespoon_to_ounce[ingr] = 0.18
            elif ingr in self.set3:
                self.teaspoon_to_ounce[ingr] = 0.044
                self.tablespoon_to_ounce[ingr] = 0.132
            elif ingr in self.set4:
                self.teaspoon_to_ounce[ingr] = 0.055
                self.tablespoon_to_ounce[ingr] = 0.165
            elif ingr in self.set5:
                self.teaspoon_to_ounce[ingr] = 0.25
                self.tablespoon_to_ounce[ingr] = 0.75
            elif ingr in self.set6:
                self.teaspoon_to_ounce[ingr] = 0.1467
                self.tablespoon_to_ounce[ingr] = 0.44
            elif ingr in self.set7:
                self.teaspoon_to_ounce[ingr] = 0.1867
                self.tablespoon_to_ounce[ingr] = 0.56
            elif ingr in self.set8:
                self.teaspoon_to_ounce[ingr] = 0.13
                self.tablespoon_to_ounce[ingr] = 0.39
            elif ingr in self.set9:
                self.teaspoon_to_ounce[ingr] = 0.07667
                self.tablespoon_to_ounce[ingr] = 0.23
            elif ingr in self.set10:
                self.teaspoon_to_ounce[ingr] = 0.2467
                self.tablespoon_to_ounce[ingr] = 0.74

            self.cup_to_ounce[ingr] = 8 * self.tablespoon_to_ounce[ingr]

    def convert(self, portion, ingr, src_unit, tgt_unit):

        ingr = ingr.lower()
        src_unit = src_unit.lower()
        tgt_unit = tgt_unit.lower()

        if not self.check_ingr_avail(ingr):
            print ("ingredient {} is unavailable!".format(ingr))
            return -1

        # if src_unit is {"pound", "ounce", "teaspoon", "tablespoon"} do nothing here just return
        if src_unit in self.conversions.keys():
            if tgt_unit in self.conversions[src_unit].keys():
                return portion * self.conversions[src_unit][tgt_unit]

        coef = 1.0
        # check to correct source units
        if src_unit=="none":

            if ingr in self.None_to_gram.keys():
                src_unit = "gram"
                coef = self.None_to_gram[ingr]
            elif ingr in self.None_to_milliliter.keys():
                src_unit = "milliliter"
                coef = self.None_to_milliliter[ingr]

            coef = coef * self.conversions[src_unit]["ounce"]
            # source unit changed to ounce
            src_unit = "ounce"

        elif src_unit=="cup":

            if ingr in self.cup_to_ounce.keys():
                src_unit = "ounce"
                coef = coef * self.cup_to_ounce[ingr]

        elif src_unit=="teaspoon":

            if ingr in self.teaspoon_to_ounce.keys():
                src_unit = "ounce"
                coef = coef * self.teaspoon_to_ounce[ingr]

        elif src_unit=="tablespoon":

            if ingr in self.tablespoon_to_ounce.keys():
                src_unit = "ounce"
                coef = coef * self.tablespoon_to_ounce[ingr]

        # check to correct target units
        if tgt_unit=="none":

            if ingr in self.None_to_gram.keys():
                tgt_unit = "gram"
                coef = coef * 1.0/self.None_to_gram[ingr]
            elif ingr in self.None_to_milliliter.keys():
                tgt_unit = "milliliter"
                coef = coef * 1.0/self.None_to_milliliter[ingr]

            coef = coef * self.conversions["ounce"][tgt_unit]
            # target unit changed to ounce
            tgt_unit = "ounce"

        elif tgt_unit=="cup":

            if ingr in self.cup_to_ounce.keys():
                tgt_unit = "ounce"
                coef = coef * 1.0/self.cup_to_ounce[ingr]

        elif tgt_unit=="teaspoon":

            if ingr in self.teaspoon_to_ounce.keys():
                tgt_unit = "ounce"
                coef = coef * 1.0/self.teaspoon_to_ounce[ingr]

        elif tgt_unit=="tablespoon":

            if ingr in self.tablespoon_to_ounce.keys():
                tgt_unit = "ounce"
                coef = coef * 1.0/self.tablespoon_to_ounce[ingr]

        return round(portion * coef * self.conversions[src_unit][tgt_unit], self.precisions[tgt_unit])

    def compute_all_conversions(self, qid_to_ingr, qid_to_unit, qid_to_portion, unit_to_idx):

        qid_to_portion_distrib = {}
        target_units = self.precisions.keys()
        for qid in qid_to_ingr.keys():
            ingr = qid_to_ingr[qid]
            portion = qid_to_portion[qid]
            src_unit = qid_to_unit[qid]
            portion_distrib = [0 for i in range(len(target_units))]
            for tgt_unit in target_units:
                converted_portion = self.convert(portion, ingr, src_unit, tgt_unit)
                portion_distrib[unit_to_idx[tgt_unit]] = converted_portion

            qid_to_portion_distrib[qid] = portion_distrib

        return qid_to_portion_distrib

    def portion_to_calorie(self, qid_to_unitportioningr, ingr_to_idx, idx_to_ingr, unit_to_idx, idx_to_unit):

        unit_size = len(unit_to_idx)
        ingr_size = len(ingr_to_idx)

        ingr_to_coefs = collections.defaultdict(lambda: collections.defaultdict(list))
        flattened_weights = [0 for i in range(unit_size*len(ingr_to_idx))]

        for qid in qid_to_unitportioningr:
            ingr, unit, idx, portion, calorie, state = qid_to_unitportioningr[qid]
            ingr_to_coefs[ingr][unit].append(calorie/portion)
            flattened_weights[ingr_to_idx[ingr]*unit_size+idx] = calorie/portion

            for tgt in self.precisions:
                if unit!=tgt:
                    converted_portion = self.convert(portion, ingr, unit, tgt)
                    #print (portion, ingr, unit, tgt, converted_portion, calorie)
                    ingr_to_coefs[ingr][tgt].append(calorie/converted_portion)

        for u in unit_to_idx:
            ingr_to_coefs['<pad>'][u] = [1]

        for idx in range(len(flattened_weights)):
            if flattened_weights[idx]==0:
                ingr_idx = idx // unit_size
                unit_idx = idx %  unit_size
                flattened_weights[idx] = np.mean(ingr_to_coefs[idx_to_ingr[ingr_idx]][idx_to_unit[unit_idx]])

        return flattened_weights

# ------------------------------------

def check_unavailable_ingredients(split, qa_root="/media/mass-data/recipe1m/from_inverse_cooking/features"):

    conversions = Conversions()

    ques_paths = os.path.join(qa_root, "recipe_{}_questions.json".format(split))

    questions =  json.load(open(ques_paths, 'r'))["questions"]
    for ques in questions:
        conversions.check_ingr_avail(ques["question"])

    print (conversions.unavailables, len(conversions.unavailables))

# ------------------------------------
