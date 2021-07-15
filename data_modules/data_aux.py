# Copyright (c) Robotic and Action Perception Lab (RPAL) at University of South Florida
# by Ahmad Babaeian Jelodar

'''
    This file is used for loading all data and creating paths to datasets, json files, and pickle files.
'''

import json, pickle, glob
import os, collections
import random
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torch.utils.data as data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------------------

class Path:

    def __init__(self, args):

        # Bottom up features root path
        self.FEATURE_PATH = '/data/ajelodar/mcan-vqa-master/datasets/recipe_extract/'
        self.UNIT_PORTION_ROOT = args.question_answer_path

        self.IMG_FEAT_PAD_SIZE = 20
        self.init_path()

    def init_path(self):

        # Bottom up features path for different splits
        self.IMG_FEAT_PATH = {
            'train': self.FEATURE_PATH + 'train/',
            'val': self.FEATURE_PATH + 'val/',
            'test': self.FEATURE_PATH + 'test/',
        }

        # Answer path
        self.PORTION_ANS_PATH = {
            'train': self.UNIT_PORTION_ROOT + "recipe_train_answers.json",
            'test': self.UNIT_PORTION_ROOT + "recipe_test_answers.json",
            'val': self.UNIT_PORTION_ROOT + "recipe_val_answers.json"
        }

        # Question path
        self.PORTION_QUES_PATH = {
            'train': self.UNIT_PORTION_ROOT + "recipe_train_questions.json",
            'test': self.UNIT_PORTION_ROOT + "recipe_test_questions.json",
            'val': self.UNIT_PORTION_ROOT + "recipe_val_questions.json"
        }

        # States path
        self.STATES_PATH = {
            'train': self.UNIT_PORTION_ROOT + "recipe1m_train.pkl",
            'test': self.UNIT_PORTION_ROOT + "recipe1m_test.pkl",
            'val': self.UNIT_PORTION_ROOT + "recipe1m_val.pkl"
        }

    def load_states(self, splits=['train', 'test', 'val']):

        unique_states = set([])
        id_to_ingrstate = collections.defaultdict(lambda: collections.defaultdict(str))
        for split in splits:
            pkl_file = pickle.load(open(self.STATES_PATH[split], 'rb'))
            for item in pkl_file:
                iid = item['id']
                for key in item:
                    states = item['states']
                    ingrs = item['ingrs']
                    for idx, state in enumerate(states):
                        id_to_ingrstate[iid][ingrs[idx]] = state
                        unique_states.add(state)

        state_to_idx, idx_to_state = create_word_to_idx_and_reverse(unique_states)

        return id_to_ingrstate, state_to_idx, idx_to_state

    def load_all_ques_ans(self):
        """ Loading question & answer files (ingredeint file & unit/portion file) """

        # Loading question word list
        ques_list_all = \
            json.load(open(self.PORTION_QUES_PATH['train'], 'r'))['questions'] + \
            json.load(open(self.PORTION_QUES_PATH['val'], 'r'))['questions'] + \
            json.load(open(self.PORTION_QUES_PATH['test'], 'r'))['questions']

        # Loading answer word list
        ans_list_all = \
            json.load(open(self.PORTION_ANS_PATH['train'], 'r'))['annotations'] + \
            json.load(open(self.PORTION_ANS_PATH['val'], 'r'))['annotations'] + \
            json.load(open(self.PORTION_ANS_PATH['test'], 'r'))['annotations']

        return ques_list_all, ans_list_all

    def load_predicted_ingrs(self, rid_to_ingrs, path="/home/rpal/ahmad/personalized_unit_recognition/data/predictions.pkl"):
        # loads predicted ingrediens from a previous model

        keys = set(list(rid_to_ingrs.keys()))

        new_predictions = {}
        predictions = pickle.load(open(path, 'rb'))
        for item in predictions:
            if item['id'] in keys:
                new_predictions[item['id']] = item['ingrs']

        return new_predictions

    def load_ques_ans(self, idx_to_ingrstate, splits=["train", "val"], idx_tag="multiple_choice_answer", value_tag="portion", calorie_class_width=100, scale_values_path="data/scales.json"):
        """
            Loading question & answer files (ingredeint file & unit/portion file) 
            into recipe id sets
        """

        qid_to_ingr = collections.defaultdict(str)
        qid_to_ingr_for_comp = collections.defaultdict(str)

        # Loading question word list
        rid_to_ingrs = collections.defaultdict(lambda: collections.defaultdict(str))
        for split in splits:
            items = json.load(open(self.PORTION_QUES_PATH[split], 'r'))['questions']
            for item in items:
                item["split"] = split
                rid_to_ingrs[item["image_id"]][item["question_id"]] = item["question"].lower()
                qid_to_ingr[item["question_id"]] = item["question"].lower()
                qid_to_ingr_for_comp[item["question_id"]] = item["question_conversion_type"].lower()

        # recipe-id to a list of tuples containing (ingredient,unit) pairs
        rid_to_data = collections.defaultdict(list)

        calorie_classes = set({})
        qid_to_calorie_class = collections.defaultdict(int)
        qid_to_calorie = collections.defaultdict(int)

        qid_to_rid = {}
        qid_to_state = {}

        scale_dict = {"min_calorie": 1e10, "max_calorie": 0, 
                      "min_ns_portion": 1e10, "max_ns_portion": 0,
                      "min_portion": 1e10, "max_portion": 0}
        # Loading answer word list
        for split in splits:
            items = json.load(open(self.PORTION_ANS_PATH[split], 'r'))['annotations']
            for item in items:
                qid = item["question_id"]
                rid = item["image_id"]

                state = get_state(rid, qid_to_ingr[qid], idx_to_ingrstate)
                qid_to_state[qid] = state

                qid_to_rid[qid] = rid

                ingr = rid_to_ingrs[rid][qid]
                unit = item[idx_tag].lower()
                portion = item[value_tag]
                scale_dict['min_ns_portion'] = min(item["non_scaled_portion"], scale_dict['min_ns_portion'])
                scale_dict['max_ns_portion'] = max(item["non_scaled_portion"], scale_dict['max_ns_portion'])

                scale_dict['min_portion'] = min(item["portion"], scale_dict['min_portion'])
                scale_dict['max_portion'] = max(item["portion"], scale_dict['max_portion'])

                calorie = item["calorie"]

                scale_dict['min_calorie'] = min(calorie, scale_dict['min_calorie'])
                scale_dict['max_calorie'] = max(calorie, scale_dict['max_calorie'])

                qid_to_calorie_class[qid] = int(calorie//calorie_class_width)
                calorie_classes.add( qid_to_calorie_class[qid] )

                qid_to_calorie[qid] = calorie

                if len(rid_to_data[rid])==0:
                    rid_to_data[rid].append(split)

                rid_to_data[rid].append((ingr, unit, portion, calorie, qid, state))

        # use scaled data & scale all un-scaled data to 0-1
        if value_tag=="portion":
            if "train" in splits:
                with open( scale_values_path, 'w') as outfile:
                    json.dump(scale_dict, outfile)
            else:
                # load scale values
                with open(os.path.join(scale_values_path)) as infile:
                    scale_dict = json.load(infile)
                    max_calorie = scale_dict["max_calorie"]
                min_calorie = scale_dict["min_calorie"]

            for rid in rid_to_data:
                items = rid_to_data[rid]
                new_items = [items[0]]
                for item in items[1:]:
                    scaled_calorie = scale_item(item[3], scale_dict["max_calorie"], scale_dict["min_calorie"])
                    new_items.append((item[0], item[1], item[2], round(scaled_calorie, 5), item[4]))

                rid_to_data[rid] = new_items

        return rid_to_ingrs, rid_to_data, qid_to_calorie_class, qid_to_calorie, qid_to_ingr, qid_to_ingr_for_comp, qid_to_state, calorie_classes, qid_to_rid, scale_dict

    def load_ques_ans_indiv(self, splits=["train", "val"], idx_tag="multiple_choice_answer", value_tag="portion", calorie_class_width=100):
        """ 
            Loading question & answer files (ingredeint file & unit/portion file) 
            into question-annswer pairs
        """
        count = 0
        # Loading question word list
        qid_to_ingr = collections.defaultdict(str)
        qid_to_ingr_for_comp = collections.defaultdict(str)
        qid_to_rid = collections.defaultdict(list)
        for split in splits:
            items = json.load(open(self.PORTION_QUES_PATH[split], 'r'))['questions']
            for item in items:
                qid_to_ingr[item["question_id"]] = item["question"].lower()
                qid_to_rid[item["question_id"]] = (item["image_id"], split)
                qid_to_ingr_for_comp[item["question_id"]] = item["question_conversion_type"].lower()
                if qid_to_ingr[item["question_id"]]=="celery":
                    count += 1

        # recipe-id to a list of tuples containing (ingredient,unit) pairs
        qid_to_unit = collections.defaultdict(str)
        qid_to_calorie = collections.defaultdict(str)
        qid_to_portion = collections.defaultdict(float)
        qid_to_calorie_class = collections.defaultdict(int)

        # Loading answer word list
        for split in splits:
            items = json.load(open(self.PORTION_ANS_PATH[split], 'r'))['annotations']
            for item in items:
                qid_to_unit[item["question_id"]] = item[idx_tag].lower()
                qid_to_portion[item["question_id"]] = item[value_tag]
                qid_to_calorie[item["question_id"]] = item["calorie"]
                qid_to_calorie_class[item["question_id"]] = int(item["calorie"]//calorie_class_width)

        return qid_to_ingr, qid_to_unit, qid_to_portion, qid_to_rid, qid_to_calorie, qid_to_ingr_for_comp, qid_to_calorie_class

# -------------------------------------------

def get_state(iid, ingr, id_to_ingrstate):

    if ingr in id_to_ingrstate[iid]:
        return id_to_ingrstate[iid][ingr]

    ingr_new = "nothing"
    if ingr=="tartar_cream":
        ingr_new = "cream"

    elif ingr=="celery":
        ingr_new = "ribs"

    elif ingr=="shell_pie":
        ingr_new = "crust_pie"

    elif ingr=="chicken" or ingr=="crabmeat":
        ingr_new = "meat"

    if ingr_new in id_to_ingrstate[iid]:
        return id_to_ingrstate[iid][ingr_new]

    words = ingr.split('_')
    if len(words)==2:
        ingr_reversed = words[1] + "_" + words[0]
        if ingr_reversed in id_to_ingrstate[iid]:
            return id_to_ingrstate[iid][ingr_reversed]
        else:
            assert 1 != 0, "len 2: ingredient length is 2 but not found!"
    else:
        assert 1 != 0, "NOT len 2: ingredient length is not 2 and not found"

# -------------------------------------------

def scale_item(unscaled_val, smax, smin):
    return (unscaled_val-smin) / (smax-smin)

def unscale_batch(scaled_batch, smin, smax):

    coef = torch.tensor(smax-smin).expand_as(scaled_batch).to(device)
    minb = torch.tensor(smin).expand_as(scaled_batch).to(device)

    return scaled_batch*coef + minb
    
# -------------------------------------------

class IngrCalorieStats:

    def __init__(self, qid_to_calorie, qid_to_ingr, prec=2):

        self.ingr_to_instances = {}
        self.ingr_to_calorie_errors = {}
        self.ingr_to_mean = {}
        self.ingr_to_std = {}
        for qid in qid_to_calorie:
            ingr = qid_to_ingr[qid]
            calorie_value = qid_to_calorie[qid]
            if not ingr in self.ingr_to_instances:
                self.ingr_to_instances[ingr] = [calorie_value]
            else:
                self.ingr_to_instances[ingr].append(calorie_value)
        
        self.compute_stats()

    def compute_stats(self ):

        for ingr in self.ingr_to_instances:
            self.ingr_to_mean[ingr] = sum(self.ingr_to_instances[ingr])/len(self.ingr_to_instances[ingr])

        for ingr in self.ingr_to_instances:
            self.ingr_to_std[ingr] = np.std(np.array(self.ingr_to_instances[ingr]))

    def compute_ingr_based_percentage_error(self, ingr, error, prec=3):
        
        std = self.ingr_to_std[ingr]
        return round(error/(2*std), prec)

    def get_mean(self, ingr):
        
        return self.ingr_to_mean[ingr]

    def update_errors(self, ingr, err):

        if not ingr in self.ingr_to_calorie_errors:
            self.ingr_to_calorie_errors[ingr] = [err]
        else:
            self.ingr_to_calorie_errors[ingr].append(err)

    def finalize_errors(self):

        self.ingr_to_calorie_error = {}
        for ingr in self.ingr_to_calorie_errors:
            self.ingr_to_calorie_error[ingr] = sum(self.ingr_to_calorie_errors[ingr]) / len(self.ingr_to_calorie_errors[ingr])

    def augment_ids(self, ids, extended_ids, qid_to_calorie, qid_to_ingr, thresh=500):

        print ('********** extended:', len(ids))
        for qid in extended_ids:
            ingr = qid_to_ingr[qid]
            calorie = qid_to_calorie[qid]
            if len(self.ingr_to_instances[ingr])<thresh:
                self.ingr_to_instances[ingr].append(calorie)
                ids.append(qid)

        self.compute_stats()
        self.print_stats(with_errors=False)

    def print_stats(self, with_errors=True, color='red', alpha=0.95, scale=50, font_size=48, log_path="logs/", max_instances=2000):

        num_to_err = []
        if with_errors:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(48,32))

        items = []
        for ingr in self.ingr_to_mean:
            if with_errors:
                error = self.ingr_to_calorie_error[ingr]
                x, y = len(self.ingr_to_instances[ingr]), error/self.ingr_to_std[ingr]
                ax.scatter(x, min(y,4), c=color, s=scale, label=color, alpha=round(alpha,2), edgecolors='none')
            else:
                error = 0
            items.append((ingr, len(self.ingr_to_instances[ingr]), self.ingr_to_mean[ingr], self.ingr_to_std[ingr], error))

        if with_errors:

            ind_x = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
            plt.xticks(ind_x, rotation='vertical', fontsize=font_size)
            ax.set_xticklabels( ind_x )
            ax.grid(True)
            fig.savefig(os.path.join(log_path, 'num_to_error.png'))   # save the figure to file

        items.sort(key = lambda x: x[3])
        items.sort(key = lambda x: x[2])
        items.sort(key = lambda x: x[1])

        for item in items:
            print (item[0], item[1], item[2], item[3], item[4])

# -------------------------------------------'

def transformation_functions(args, run_mode="train"):
    '''
        apply transfomrations to input images is images are used as input.
        the type of transformation (augmentations) used is different at train and eval time.
    '''
    transforms_list = [transforms.Resize((args.image_size))]

    if run_mode == 'train':
        # Image preprocessing, normalization for the pretrained resnet
        transforms_list.append(transforms.RandomHorizontalFlip())
        transforms_list.append(transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)))
        transforms_list.append(transforms.RandomCrop(args.crop_size))
    else:
        transforms_list.append(transforms.CenterCrop(args.crop_size))
    transforms_list.append(transforms.ToTensor())
    transforms_list.append(transforms.Normalize((0.485, 0.456, 0.406),
                                                (0.229, 0.224, 0.225)))

    return transforms.Compose(transforms_list) 

# -------------------------------------------

def load_calorie_dataset(path, scaled_output=False, calorie_range = [100, 1900]):
    '''
        loads all three splits (train, test, val) of calorie dataset into one dictionary
    '''
    iid_to_calorie = {}
    iid_to_calorie, stats = load_calorie_split(iid_to_calorie, "test", normalize=scaled_output, min_val=calorie_range[0], max_val=calorie_range[1], path=path)
    iid_to_calorie, stats = load_calorie_split(iid_to_calorie, "val", normalize=scaled_output, min_val=calorie_range[0], max_val=calorie_range[1], path=path)
    iid_to_calorie, stats = load_calorie_split(iid_to_calorie, "train", normalize=scaled_output, min_val=calorie_range[0], max_val=calorie_range[1], path=path)
    min_stat, max_stat = stats

    mi = 1e10
    mx = 0    
    for iid in iid_to_calorie:
        mi = min(iid_to_calorie[iid], mi)
        mx = max(iid_to_calorie[iid], mx)

    return iid_to_calorie, min_stat, max_stat

# -----

def load_calorie_split(iid_to_calorie, split, path, normalize=True, min_val=100, max_val=1900):
    '''
        loads one split of the calorie dataset
    '''
    path = '{}/calorie_{}.json'.format(path, split)
    calorie_dataset = json.load(open(path, 'r'))
    for id in calorie_dataset:

        calorie = calorie_dataset[id]
        if isinstance(calorie, float):
            calorie_value = calorie
        else:
            calorie_value = calorie[1]
        if normalize:
            calorie_value = (calorie_value-min_val)/(max_val-min_val)

        iid_to_calorie[id] = calorie_value

    return iid_to_calorie, (min_val, max_val)

# -------------------------------------------

def load_dish_dataset(path):
    '''
        loads all three splits (train, test, val) of dish dataset into one dictionary,
        assigns a unique index to each dish name
    '''
    iid_to_dishname = {}
    iid_to_dishname = load_dish_split(iid_to_dishname, "train", path=path)
    iid_to_dishname = load_dish_split(iid_to_dishname, "test", path=path)
    iid_to_dishname = load_dish_split(iid_to_dishname, "val", path=path)
    
    dish_to_idx, idx_to_dish = create_word_to_idx_and_reverse(list(set(iid_to_dishname.values())))
    
    return iid_to_dishname, dish_to_idx, idx_to_dish

# -----

def load_dish_split(iid_to_dishname, split, path):
    '''
        loads one split of the dish dataset
    '''
    # load dishes
    path = os.path.join(path, '{}.pkl'.format(split))
    dish_data = pickle.load(open(path, 'rb'))

    co = 0
    for key in dish_data:
        iid_to_dishname[key] = dish_data[key]["class"]
        co += 1

    return iid_to_dishname

# -------------------------------------------

def create_mappings(item_list, removing_ingredients={}, tag="question", padding=True):
    ''' 
        This function is used to create mappings for unit & ingredient classes
        creates a token to index and index to token hash (dictionary) based on:
            1. a given dictionary (item_list) and a tag to show which item of the dictionary would be used for the process.
            2. a list of removing tokens if specified.
            3. if padding add a pad token to the hash.
    '''
    
    # create a list of tokens to create a mapping from
    tokens = []
    for item in item_list:
        tokens.append( item[tag].lower() )

    token_to_idx, idx_to_token = create_word_to_idx_and_reverse(tokens, removing_ingredients)

    if padding:
        pad_idx = len(token_to_idx)
        token_to_idx['<pad>'] = pad_idx
        idx_to_token[pad_idx] = '<pad>'
    
    return token_to_idx, idx_to_token

# -------------------------------------------

def create_word_to_idx_and_reverse(words, removing_words={}):
    ''' 
        This function is used to create one-to-one indice for word-to-index.
        creates a token to index and index to token hash (dictionary) based on:
            1. a given set of words.
            2. it does not index for words in removing_words.
    '''
    word_to_idx = {}
    idx_to_word = {}
    idx = 0
    for word in sorted(list(words)):
        if not word in word_to_idx and not word in removing_words:
            word_to_idx[word] = idx
            idx_to_word[idx] = word
            idx += 1

    return word_to_idx, idx_to_word

# -------------------------------------------

def prepare_image_features(splits, aux_data_dir, data_dir, suff, args, rids, run_mode, root, bottom_up_feat_size=2048):
    '''
        Load images or pre-processed features of images to either
            1. use pre-computed Resnet features.
            2. use pre-computed bottom-up features.
            3. use images themselves as input to the model.
    '''
    image_encoder_type = args.image_encoder_type
    maxnumims = args.maxnumims

    iid_to_img_feat_path = None
    iid_to_img_feat = None
    img_feat_path_list = []
    iid_to_impaths = collections.defaultdict(list)
    iid_to_image = collections.defaultdict(list)

    # just to resize the image
    resize_transform = transforms.Compose([transforms.Resize((args.image_size))])

    # laod image paths & images
    for split in splits:
        dataset = pickle.load(open(os.path.join(aux_data_dir, suff + 'recipe1m_'+split+'.pkl'), 'rb'))
        for item in dataset:
            iid = item["id"]
            if iid in rids:
                iid_to_impaths[iid] = item['images'][0:maxnumims]
                
                if image_encoder_type=="resnet":
                    for path in iid_to_impaths[iid]:
                        image = Image.open(os.path.join(root, split, path[0], path[1], path[2], path[3], path)).convert('RGB')
                        iid_to_image[iid].append(resize_transform(image))

    iid_to_image_features = None
    if image_encoder_type=="resnet_features":
        # load image resnet featurs 
        print ("Using Resnet features computed before.")
        iid_to_image_features = {}
        for split in splits:
            features_path = os.path.join(args.features_dir, '{}_features.pkl'.format(split))
            if not os.path.exists(features_path):
                features_path = os.path.join(args.features_dir, 'train_features1.pkl')
                iid_to_image_features.update( pickle.load(open(features_path, 'rb')) )
                features_path = os.path.join(args.features_dir, 'train_features2.pkl')
            
            iid_to_image_features.update( pickle.load(open(features_path, 'rb')) )

    elif image_encoder_type=="resnet":
        
        print ("Using actual images thorugh Resnet.")
        
    elif image_encoder_type=="bottom-up":
        # load bottom-up image features
        print ("Using Bottom-up features.")
        for split in splits:
            if split in ['train', 'val', 'test']:
                img_feat_path_list += glob.glob(self.paths.IMG_FEAT_PATH[split] + '*.npz')

        feat_size = -1
        # image id to image features or paths
        if args.pre_load:
            print ("   pre-loading features.")
            iid_to_img_feat, feat_size = img_feat_load(img_feat_path_list, rids)
        else:
            iid_to_img_feat_path = img_feat_path_load(img_feat_path_list)
        
        if feat_size>0:
            bottom_up_feat_size = feat_size

    return bottom_up_feat_size, iid_to_img_feat_path, iid_to_img_feat, iid_to_impaths, iid_to_image, iid_to_image_features # img_feat_path_list

# -------------------------------------------

def load_removing_ingredients(args):
    '''
        This function creates a subset of ingredients to be removed for experiments.
        Removes either 
            1. the N ingredients with highest error in previous experiments. or
            2. all but a specified subset (pick_subset_ingrs).
    '''
    ingr_to_error = json.load(open(args.ingr_list_errors, 'r'))["errors"]

    additional_fake_errors = {"shell_pie": 0, "tartar_cream": 0}
    removing_ingredients  = set({})
    for term in additional_fake_errors:
        ingr_to_error[term] = additional_fake_errors[term]

    if args.remove_highest_errors:
        sorted_ingr_to_error = sorted(ingr_to_error.items(), key=lambda x: x[1], reverse=True)    
        for item, error in sorted_ingr_to_error[0:args.N_error]:
            removing_ingredients.add(item)
    elif args.pick_subset_ingrs:
        subset = args.ingr_subset
        for ingr in ingr_to_error:
            if not ingr in subset:
                removing_ingredients.add(ingr)

    return removing_ingredients

# -------------------------------------------

def create_cross_validation(samples=641, N=10, cross_val_path="cross_val_indice.txt"):

    test_size = samples//N
    cross_val_indice = open(cross_val_path, 'w')
    samples = [i for i in range(samples)]
    random.shuffle(samples)
    for cv_idx in range(N):
        train_set_indice = samples[0:cv_idx*test_size]
        test_set_indice = samples[cv_idx*test_size:(cv_idx+1)*test_size]
        train_set_indice.extend( samples[(cv_idx+1)*test_size:] )
        print (len(train_set_indice), len(test_set_indice))
        cross_val_indice.write("\ncross_val-{}\n  train-set: \n    ".format(cv_idx))
        for indx in train_set_indice[:-1]:
            cross_val_indice.write(str(indx)+", ")
        cross_val_indice.write(str(train_set_indice[-1]))

        cross_val_indice.write("\n  eval-set: \n    ")
        for indx in test_set_indice[:-1]:
            cross_val_indice.write(str(indx)+", ")
        cross_val_indice.write(str(test_set_indice[-1]))

def read_cross_val_indice(cross_split=0, run_mode="train", rootdir="data/menu_match_dataset", cross_val_path="cross_val_indice.txt"):
    
    cv_idx = -1
    cross_val_indice = open(os.path.join(rootdir, cross_val_path))
    read = False
    for line in cross_val_indice:
        if "cross_val" in line:
            _, cv_idx = line.strip().split("-")
            cv_idx = int(cv_idx)
        elif run_mode in line and cv_idx==cross_split:
            read = True
        elif read:
            indice = [int(x) for x in line.strip().split(",")]
            read = False

    return indice

def load_menumatch_dataset(transforms, run_mode="train", cross_split=0, normalize=False, rootdir="data/menu_match_dataset"):

    data_indice = read_cross_val_indice(cross_split, run_mode, rootdir)
    
    min_stat = 10000
    max_stat = -1

    # ingredient to calorie
    ingr_to_calorie = {}
    items_info = open(os.path.join(rootdir, "items_info.txt"))
    for line in items_info:
        _, lbl, calorie, dish_type = line.strip().split(";")
        lbl = lbl.strip().lower()
        calorie = calorie.strip().lower()
        dish_type = dish_type.strip().lower()
        if dish_type!="restaurant":
            ingr_to_calorie[lbl] = float(calorie)        

            max_stat = max(max_stat, ingr_to_calorie[lbl])
            min_stat = min(min_stat, ingr_to_calorie[lbl])

    if normalize:
        for lbl in ingr_to_calorie:
            ingr_to_calorie[lbl] = (ingr_to_calorie[lbl]-min_stat)/(max_stat-min_stat)

    name_to_impaths = {}
    name_to_image = {}
    labels = open(os.path.join(rootdir, "labels.txt"))

    rid_to_name = {}
    rid_to_calorie = {}
    rid_to_ingrs = collections.defaultdict(list)
    iid_to_rid = {}
    iid_to_name = {}
    iid_to_ingr = {}
    iid_to_calorie = {}

    rid = 0
    iid = 0
    # image to a list of ingredients
    idx = 0    
    for line in labels:
        parts = line.strip().split(";")
        img_name = parts[0]
        # load image paths and images & apply transformations (if applicable) & save them in dict
        img_path = os.path.join(rootdir, "foodimages", img_name)
        image = Image.open(img_path).convert('RGB')
        if transforms is not None:
            image = transforms[run_mode](image)
        name_to_image[img_name] = image
        name_to_impaths[img_name] = img_path

        # create ingredient id to ingredient dict, recipe id to ingredient list dict, ...
        lbls = [lbl.strip().lower() for lbl in parts[1:]][:-1]
        if idx in data_indice:
            rid_to_calorie[rid] = 0
            rid_to_name[rid] = img_name
            for lbl in lbls:
                iid_to_name[iid] = img_name
                iid_to_ingr[iid] = lbl
                iid_to_calorie[iid] = ingr_to_calorie[lbl]
                rid_to_calorie[rid] += ingr_to_calorie[lbl]
                iid_to_rid[iid] = rid
                rid_to_ingrs[rid].append(iid)
                iid += 1
            rid += 1
        idx += 1

    return iid_to_calorie, iid_to_ingr, iid_to_name, iid_to_rid, rid_to_calorie, rid_to_name, rid_to_ingrs, name_to_image, ingr_to_calorie, min_stat, max_stat

# -------------------------------------------

import numpy as np
import en_vectors_web_lg, re, json, collections
import en_trf_bertbaseuncased_lg

# https://spacy.io/models/en-starters#en_vectors_web_lg
# https://spacy.io/models/en-starters#en_trf_bertbaseuncased_lg

def word_to_embedding(idx_to_word, embedding_type="Bert"):

    spacy_tool = None

    if embedding_type=='Glove':
        spacy_tool = en_vectors_web_lg.load()
    elif embedding_type=='Bert':
        spacy_tool = en_trf_bertbaseuncased_lg.load()

    pretrained_emb = []
    for idx in range(len(idx_to_word)):
        word_vector = spacy_tool(idx_to_word[idx]).vector
        pretrained_emb.append(word_vector)

    pretrained_emb = np.array(pretrained_emb)

    return pretrained_emb

# -------------------------------------------

def keep_specified_ids(dict, ids):

    new_dict = {}
    for id in ids:
        new_dict[id] = dict[id]
    return new_dict

