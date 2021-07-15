# Copyright (c) Robotic and Action Perception Lab (RPAL) at University of South Florida
# by Ahmad Babaeian Jelodar

'''
    This file creates a dataset loader for when the input is a single ingredient & an image and
    the output is either a portion, a unit, or a calorie for the ingredient in the image.
'''

import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os, pickle, json
import numpy as np
import random, collections
from modules.utils import img_feat_load, img_feat_path_load, proc_img_feat
from data_modules.data_aux import transformation_functions, load_calorie_dataset, load_dish_dataset, Path, create_mappings, prepare_image_features, load_removing_ingredients, IngrCalorieStats, word_to_embedding, load_menumatch_dataset, create_word_to_idx_and_reverse
from data_modules.conversions import Conversions

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------------------

class Ingr1MDataset(data.Dataset):
    '''
        To generate dataset individually for each (ingr, image) pair in an instance of the data
    '''

    def __init__(self, data_dir, aux_data_dir, splits, run_mode, transforms=None, suff='', args=None):

        # this function loads or/and computes a subset of ingredients that
        # the model would not be trained or evaluated on.
        self.removing_ingredients = load_removing_ingredients(args)

        self.paths = Path(args)
        
        self.maxnumingrs = args.maxnumingrs
        self.run_mode = run_mode
        self.indiv_calorie = False
        min_calorie_thresh = args.calorie_range[0]
        max_calorie_thresh = args.calorie_range[1]

        # if indiv_calorie is true the model would learn to predict ingredient calorie
        # if indiv_calorie is false the model would learn to predict recipe calorie
        if args.model_type in {"simple_ml_calorie1", "simple_ml_calorie1_dish", "simple_ml_calorie2", "simple_ml_calorie2_dish", "simple_ingrtocalorie", "simple_ingrtocalorie_dish", "simple_ingrtojointcalorie", "simple_ingrtojointcalorie_dish", "simple_heirarchitocalorie", "simple_heirarchitocalorie_dish", "simple_heirarchitoportion", "simple_heirarchitoportion_dish"}:
            self.indiv_calorie = True
        
        # if using a subset of ingredients compute individual ingredient calories
        if args.pick_subset_ingrs:
            self.indiv_calorie = True

        self.iid_to_calorie, self.min_stat, self.max_stat = load_calorie_dataset(path=args.calorie_dataset_path, scaled_output=args.scaled_output, calorie_range=args.calorie_range)

        self.iid_to_dishname, self.dish_to_idx, self.idx_to_dish = load_dish_dataset(path=args.calorie_dataset_path)

        # load all questions & answrs for creating vocabs
        self.ques_list_all, self.ans_list_all = self.paths.load_all_ques_ans()

        # input mappings (ingredient tokens)
        if args.just_eval_subset_ingrs and run_mode=="eval":
            # only instances with the the specified ingredients (removing_ingredients) would be removed
            # do not change the dictionary according to the removed ingredients if subset ingredients at evaluation time and the model is at eval time 
            self.ingr_to_idx, self.idx_to_ingr = create_mappings(self.ques_list_all, {}, tag="question")
        else:
            self.ingr_to_idx, self.idx_to_ingr = create_mappings(self.ques_list_all, self.removing_ingredients, tag="question")
        self.pad_value = self.ingr_to_idx['<pad>']
        
        self.pretrained_emb = word_to_embedding(self.idx_to_ingr)

        # output mappings (unit tokens)
        self.unit_to_idx, self.idx_to_unit = create_mappings(self.ans_list_all, tag="multiple_choice_answer", padding=False)
        self.unit_pad_value = len(self.unit_to_idx)
      
        print (" ---------- data info ---------- ")
        # loading questions & answers for training/testing
        if args.scaled_output:
            value_tag = "portion"
            print ("Data is scaled.")
        else:
            value_tag = "non_scaled_portion"
            print ("Data is Not scaled.")

        self.qid_to_ingr, self.qid_to_unit, self.qid_to_portion, self.qid_to_rid, self.qid_to_calorie, self.qid_to_ingr_for_comp, self.qid_to_calorie_class = self.paths.load_ques_ans_indiv(splits, value_tag=value_tag, calorie_class_width=args.calorie_class_width)


        self.use_distrib_loss = args.use_distrib_loss
        conversions = Conversions()
        if self.use_distrib_loss:
            self.qid_to_portion_distrib = conversions.compute_all_conversions(self.qid_to_ingr, self.qid_to_unit, self.qid_to_portion, self.unit_to_idx)

        # filter recipe-ids based on two criteria
        #   1. if the whole recipe has an ingredient in the self.removing_ingredients set.
        #   2. if the whole recipe calorie is not in the specified range.
        removing_rids = set({})
        self.rids = set()
        for key in self.qid_to_rid.keys():
            rid = self.qid_to_rid[key][0]
            # ingredient filtering
            if not args.pick_subset_ingrs and self.qid_to_ingr[key] in self.removing_ingredients:
                removing_rids.add(rid)
            # calorie filtering
            if self.iid_to_calorie[rid]>=min_calorie_thresh and self.iid_to_calorie[rid]<=max_calorie_thresh:
                self.rids.add( rid )

        self.rids = set(self.rids) - removing_rids
        
        # set transformation operator for image transformation
        self.transforms = transforms
        self.image_encoder_type = args.image_encoder_type
        self.pre_load = args.pre_load
        self.img_pad_size = args.img_pad_size

        # create root dir for image paths
        self.root = os.path.join(data_dir, 'images')

        self.bottom_up_feat_size, self.iid_to_img_feat_path, self.iid_to_img_feat, self.iid_to_impaths, self.iid_to_image, self.iid_to_image_features = prepare_image_features(splits, aux_data_dir, data_dir, suff, args, self.rids, self.run_mode, self.root, self.transforms)

        if not self.iid_to_image_features:
            self.iid_to_image_features = self.iid_to_image

        self.calorie_classes = set({})
        # filter ingredient-ids based on three criteria
        #   1. if the instance has an ingredient in the self.removing_ingredients set.
        #   2. if the whole recipe calorie is not in the specified range.
        self.ids = []
        extended_ids = []
        recipes_covered = set({})
        for qid in list(self.qid_to_ingr.keys()):
            iid = self.qid_to_rid[qid][0]
            if iid in self.iid_to_image_features.keys():
                # a) not subset processing or 
                # b) subset processing and the ingredients is in the subset.
                if not args.pick_subset_ingrs or (args.pick_subset_ingrs and not self.qid_to_ingr[qid] in self.removing_ingredients):
                    if iid in self.rids: # self.iid_to_calorie[iid]>=min_calorie_thresh and self.iid_to_calorie[iid]<=max_calorie_thresh:
                        self.ids.append(qid)
                        recipes_covered.add(iid)
                        self.calorie_classes.add( self.qid_to_calorie_class[qid] )
                    else:
                        extended_ids.append(qid)    

        self.ingr_calorie_stats = IngrCalorieStats(self.qid_to_calorie, self.qid_to_ingr)

        # put instances with the same class name (same ingredient) in the same set for paired batch processing.
        if self.use_distrib_loss:
            self.create_class_wise_set()

        self.shuffle()
        self.data_size = len(self.ids)

        print ("Data loader in {} mode!\n   experiments on: {}".format(self.run_mode, splits))
        print ("Number of dish tokens: {}.".format(self.get_dish_vocab_size()))
        print ("Number of input tokens: {}.".format(self.get_ingrs_vocab_size()))
        print ("Number of output tokens: {}.".format(self.get_unit_vocab_size()))
        print ("Number of recipes covered: {}.".format(len(recipes_covered)))
        if self.indiv_calorie:
            print ("Calorie estimation of ingredients.")
        else:
            print ("Calorie estimation of recipes.")
        print ("Data size: {}.".format(self.data_size))
        print (" ---------- --------- ---------- \n")

    def compute_stats(self, ingr, error, prec=3):
        mean = self.ingr_calorie_stats.get_mean(ingr)
        percentage_error = self.ingr_calorie_stats.compute_ingr_based_percentage_error(ingr, error, prec)
        return percentage_error, mean

    def get_ingr_calorie_mean(self, ingr):
        return self.ingr_calorie_stats.get_mean(ingr)

    def ingr_prediction_stats_update(self, ingr, calorie_prediction_error):
        self.ingr_calorie_stats.update_errors(ingr, calorie_prediction_error)

    def ingr_prediction_stats_finalize(self):
        self.ingr_calorie_stats.finalize_errors()
        self.ingr_calorie_stats.print_stats()

    def unscale_item(self, unscaled_val):
        return unscaled_val * (self.max_stat-self.min_stat) + self.min_stat

    def unscale_batch(self, unscaled_batch):

        coef = torch.tensor(self.max_stat-self.min_stat).expand_as(unscaled_batch)
        minb = torch.tensor(self.min_stat).expand_as(unscaled_batch)

        return unscaled_batch*coef + minb

    def shuffle(self):
        random.shuffle(self.ids)

    def get_ingrs_vocab_size(self):
        return len(self.ingr_to_idx)

    def get_unit_vocab_size(self):
        return len(self.unit_to_idx)

    def get_dish_vocab_size(self):
        return len(self.dish_to_idx)

    def get_calorie_classes_num(self):
        return len(self.calorie_classes)

    def get_filtered_ids(self):
        return self.ids

    def create_class_wise_set(self):
        '''
            creates a hashmap that maps an ingredient name to a set of question-ids
        '''
        qid_set = set(self.ids)
        self.classwise_qids = {}
        for qid in self.qid_to_ingr.keys():
            if qid in qid_set:
                ingr_class = self.qid_to_ingr[qid]
                if ingr_class in self.classwise_qids.keys():
                    self.classwise_qids[ingr_class].append(qid)
                else:
                    self.classwise_qids[ingr_class] = [qid]

    def get_pairs(self, batch_qids):
        '''
            creates a list of qids 
                1. where each qid in the list is picked randomly &
                2. has the same class as a qid in the original batch
        '''

        if not self.use_distrib_loss:
            return None, None, None, None

        # create the list of randomly picked (paired) qids
        self.pair_qids = []
        for qid in batch_qids:
            ingr_class = self.qid_to_ingr[qid]
            self.pair_qids.append(random.choice(self.classwise_qids[ingr_class]))

        batch = []
        # create a batch using the created list
        for idx, _ in enumerate(self.pair_qids):
            ingr_idx, unit_idx, portion_gt, calorie_gt, portion_distrib_gt, dish_idx, image_input, iid, qid, path = self.__getitem__(-idx-1)

            # create lists of tuples (from output)
            batch.append((ingr_idx, unit_idx, portion_gt, calorie_gt, portion_distrib_gt, dish_idx, image_input, iid, qid, path))

        ingr_idxs, unit_idxs, _, _, _, _, dish_idxs, image_input, _, _, _ = collate_fn(batch)

        return ingr_idxs, unit_idxs, dish_idxs, image_input

    def __getitem__(self, idx):
        """
            Returns image, 1 ingredient, 1 unit, 1 portion.
        """

        if idx<0:
            # create batchs using loading of the paired samples for kl-divergence loss of portion distributions
            qid = self.pair_qids[-idx-1]
        else:    
            # create original batch samples
            qid = self.ids[idx]

        ingr = self.qid_to_ingr[qid]
        unit = self.qid_to_unit[qid]
        calorie_class_idx = self.qid_to_calorie_class[qid]
        
        portion = self.qid_to_portion[qid]
        portion_distrib = np.zeros(6)
        if self.use_distrib_loss:
            portion_distrib = np.asarray(self.qid_to_portion_distrib[qid])
        iid, split = self.qid_to_rid[qid]

        dish_gt = self.iid_to_dishname[iid]
        
        # individual ingredient calorie
        ingredient_calorie = self.qid_to_calorie[qid]
        # recipe calorie
        recipe_calorie = self.iid_to_calorie[iid]

        dish_idx = self.dish_to_idx[dish_gt]
        ingr_idx = self.ingr_to_idx[ingr]
        unit_idx = self.unit_to_idx[unit]
        portion_gt = float(portion)

        ingredient_calorie_gt = float(ingredient_calorie)
        recipe_calorie_gt = float(recipe_calorie)
        mean_prior_calories_gt = float(self.get_ingr_calorie_mean(ingr))

        paths = self.iid_to_impaths[iid]

        if self.run_mode == 'train':
            img_idx = np.random.randint(0, len(paths))
        else:
            img_idx = 0

        path = ''
        if self.image_encoder_type=="resnet_features":    
            img_feat_iter = self.iid_to_image_features[iid]
            image_input = torch.from_numpy(img_feat_iter).float()

        elif self.image_encoder_type=="resnet":
            path = paths[img_idx]
            image_input = self.iid_to_image[iid][img_idx]
            if self.transforms is not None:
                image_input = self.transforms[self.run_mode](image_input)

        elif self.image_encoder_type=="bottom-up":
            # Process image feature from (.npz) file
            if self.pre_load:
                img_feat_x = self.iid_to_img_feat[iid]
            else:
                img_feat = np.load(self.iid_to_img_feat_path[iid])

            img_feat_iter = proc_img_feat(img_feat_x, self.img_pad_size)
            image_input = torch.from_numpy(img_feat_iter).float()

        portion_distrib_gt = torch.from_numpy(portion_distrib).float()

        return ingr_idx, unit_idx, portion_gt, ingredient_calorie_gt, recipe_calorie_gt, mean_prior_calories_gt, portion_distrib_gt, calorie_class_idx, dish_idx, image_input, iid, qid, path

    def __len__(self):
        return len(self.ids)

# -------------------------------------------

def collate_fn(batch):
    '''
        Collate function for the pytorch data-loader class (stacks given data from the get-item function).
    '''

    # Sort a data list by caption length (descending order).
    ingr_idx, unit_idx, portion_gt, ingredient_calorie_gt, recipe_calorie_gt, mean_prior_calories_gt, portion_distrib_gt, calorie_class_idx, dish_idx, image_input, img_id, qids, path = zip(*batch)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    image_input = torch.stack(image_input, 0)

    portion_gt = torch.tensor(portion_gt)
    ingredient_calorie_gt = torch.tensor(ingredient_calorie_gt)
    recipe_calorie_gt = torch.tensor(recipe_calorie_gt)
    mean_prior_calories_gt = torch.tensor(mean_prior_calories_gt)
    dish_idxs = torch.tensor(dish_idx)
    ingr_idxs = torch.tensor(ingr_idx)
    unit_idxs = torch.tensor(unit_idx)
    portion_distrib_gt = torch.stack(portion_distrib_gt, 0)
    calorie_class_idxs = torch.tensor(calorie_class_idx)

    return ingr_idxs, unit_idxs, portion_gt, recipe_calorie_gt, ingredient_calorie_gt, mean_prior_calories_gt, portion_distrib_gt, calorie_class_idxs, dish_idxs, image_input, img_id, qids, path

# -------------------------------------------

def get_ingr_loader(data_dir, aux_data_dir, splits, run_mode, transforms, batch_size,
               shuffle, num_workers, drop_last=False,
               suff='',
               args=None):

    '''
        Creates the dataset and the data loader for the train & 
        sample phases to load the instances of the dataset from.
    '''

    if args.dataset=="recipe1m":
        dataset = Ingr1MDataset(data_dir=data_dir, aux_data_dir=aux_data_dir, splits=splits,
                                run_mode=run_mode,
                                transforms=transforms,
                                suff=suff,
                                args=args)
        collate_function = collate_fn
    elif args.dataset=="menumatch":
        dataset = IngrMenuMatchDataset(run_mode=run_mode, transforms=transforms, 
                                       suff=suff, args=args)
                        
        collate_function = collate_menumatch_fn

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                              drop_last=drop_last, collate_fn=collate_function, pin_memory=True)
    return data_loader, dataset

# -------------------------------------------

def collate_menumatch_fn(batch):
    '''
        Collate function for the pytorch data-loader class for the menu-match dataset.
    '''

    # Sort a data list by caption length (descending order).
    ingr_idx, ingr_calorie, recipe_calorie, image_input, qid = zip(*batch)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    image_input = torch.stack(image_input, 0)
    ingr_calorie = torch.tensor(ingr_calorie)
    recipe_calorie = torch.tensor(recipe_calorie)
    ingr_idxs = torch.tensor(ingr_idx)

    return ingr_idxs, None, None, recipe_calorie, ingr_calorie, ingr_calorie, None, None, image_input, None, qid, None

# -------------------------------------------

class IngrMenuMatchDataset(data.Dataset):
    '''
        To generate dataset individually for each (ingr, image) pair in an instance of the data
    '''

    def __init__(self, run_mode, transforms=None, suff='', args=None):

        self.run_mode = run_mode
        self.iid_to_calorie, self.iid_to_ingr, self.iid_to_name, self.iid_to_rid, self.rid_to_calorie, self.rid_to_name, self.rid_to_ingrs, self.name_to_image, self.ingr_to_calorie, self.min_stat, self.max_stat = load_menumatch_dataset(transforms, run_mode, args.cross_valid_idx, args.scaled_output)
        
        self.ids = list(self.iid_to_ingr.keys())
        self.data_size = len(self.ids)

        self.rids = list(self.rid_to_calorie.keys())
        self.ingr_to_idx, self.idx_to_ingr = create_word_to_idx_and_reverse(list(self.ingr_to_calorie.keys()))
        self.pad_value = len(self.ingr_to_idx)
        self.ingr_to_idx['<pad>'] = self.pad_value
        self.idx_to_ingr[self.pad_value] = '<pad>'

        self.dish_to_idx, self.idx_to_dish = create_word_to_idx_and_reverse(["asian", "italian", "soup"])
        
        self.unit_pad_value = 0
        self.bottom_up_feat_size = -1
        self.pretrained_emb = None
        self.unit_vocab_size = -1

        print ("Data loader in {} mode!".format(run_mode))
        print ("Number of input tokens: {}.".format(self.get_ingrs_vocab_size()))
        print ("Number of recipes covered: {}.".format(len(self.rid_to_calorie)))
        print ("Calorie estimation of ingredients.")
        print ("Data size: {}.".format(self.data_size))
        print (" ---------- --------- ---------- \n")

    def get_ingrs_vocab_size(self):
        return len(self.ingr_to_idx)
    
    def get_dish_vocab_size(self):
        return len(self.dish_to_idx)

    def get_unit_vocab_size(self):
        return self.unit_vocab_size

    def get_ingr_calorie_mean(self, ingr):
        return self.ingr_to_calorie[ingr]

    def __getitem__(self, idx):
        """
            Returns image, 1 ingredient
        """

        iid = self.ids[idx]
        rid = self.iid_to_rid[iid]

        ingr = self.iid_to_ingr[iid]
        ingr_idx = self.ingr_to_idx[ingr]
        ingr_calorie = float(self.iid_to_calorie[iid])
        recipe_calorie = float(self.rid_to_calorie[rid])
        

        img_name = self.iid_to_name[iid]
        image_input = self.name_to_image[img_name]

        return ingr_idx, ingr_calorie, recipe_calorie, image_input, iid

    def shuffle(self):
        random.shuffle(self.ids)

    def __len__(self):
        return len(self.ids)

    def unscale_batch(self, unscaled_batch):

        coef = torch.tensor(self.max_stat-self.min_stat).expand_as(unscaled_batch)
        minb = torch.tensor(self.min_stat).expand_as(unscaled_batch)

        coef = coef.to(device)
        minb = minb.to(device)

        return unscaled_batch*coef + minb

# -------------------------------------------