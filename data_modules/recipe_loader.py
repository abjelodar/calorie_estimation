# Copyright (c) Robotic and Action Perception Lab (RPAL) at University of South Florida
# by Ahmad Babaeian Jelodar

'''
    This file creates a dataset loader for when the input is all ingredients & an image and
    the output is a calorie estimated for the image and intermediate outputs of portions,
    units, and calories per ingredient.
'''

import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os, pickle, json
import numpy as np
import random, collections
from modules.utils import img_feat_load, img_feat_path_load, proc_img_feat
from data_modules.data_aux import transformation_functions, load_calorie_dataset, load_dish_dataset, Path, create_mappings, prepare_image_features, scale_item, unscale_batch, IngrCalorieStats, keep_specified_ids
from data_modules.conversions import Conversions
from PIL import Image
from torchvision import transforms
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------------------

class Recipe1MDataset(data.Dataset):
    '''
        To generate dataset for image and all associated ingredients in one instance of the data
    '''

    def __init__(self, data_dir, aux_data_dir, splits, run_mode, transforms=None, suff='', args=None):

        # if set to true, the model gets predicted ingredients from previous model as input else it uses ground-truth ingredients
        self.use_predicted_ingrs = args.use_predicted
        self.evaluate_given_images = args.evaluate_given_images

        self.one_pass_counter = 0

        self.pretrained_emb = None
        self.batch_size = args.batch_size

        self.paths = Path(args)
        self.maxnumingrs = args.maxnumingrs

        self.run_mode = run_mode

        self.iid_to_calorie, self.min_stat, self.max_stat = load_calorie_dataset(path=args.calorie_dataset_path, scaled_output=args.scaled_output, calorie_range=args.calorie_range)

        self.iid_to_dishname, self.dish_to_idx, self.idx_to_dish = load_dish_dataset(path=args.calorie_dataset_path)

        # load all questions & answrs for creating vocabs
        self.ques_list_all, self.ans_list_all = self.paths.load_all_ques_ans()

        # input mappings
        self.ingr_to_idx, self.idx_to_ingr = create_mappings(self.ques_list_all, tag="question")
        self.pad_value = self.ingr_to_idx['<pad>']

        # output mappings
        self.unit_to_idx, self.idx_to_unit = create_mappings(self.ans_list_all, tag="multiple_choice_answer", padding=False)
        self.unit_pad_value = len(self.unit_to_idx)

        # states mappings
        self.idx_to_ingrstate, self.state_to_idx, self.idx_to_state = self.paths.load_states()
        self.state_pad_value = len(self.state_to_idx)

        print (" ---------- data info ---------- ")
        # loading questions & answers for training/testing
        if args.scaled_output:
            value_tag = "portion"
            print ("Data is scaled.")
        else:
            value_tag = "non_scaled_portion"
            print ("Data is Not scaled.")

        self.rid_to_ingrs, self.rid_to_data, self.qid_to_calorie_class, self.qid_to_calorie, self.qid_to_ingr, self.qid_to_ingr_for_comp, self.qid_to_state, self.calorie_classes, self.qid_to_rid, self.scale_dict = self.paths.load_ques_ans(self.idx_to_ingrstate, splits, value_tag=value_tag, calorie_class_width=args.calorie_class_width)

        self.qid_to_unitportion = {}

        # filter out iids that are not in the calorie dataset
        self.ids = []
        for iid in self.rid_to_data:
            if iid in self.iid_to_calorie and self.iid_to_calorie[iid]>=args.calorie_range[0] and self.iid_to_calorie[iid]<=args.calorie_range[1]:
                self.ids.append(iid)
                items = self.rid_to_data[iid][1:]
                for item in items:
                    ingr, unit, portion, calorie, qid, state = item
                    self.qid_to_unitportion[qid] = (ingr, unit, self.unit_to_idx[unit], portion, calorie, state)

        # take all predicted ingredients from file
        if self.use_predicted_ingrs:
            self.predicted_ingrs = self.paths.load_predicted_ingrs(self.rid_to_ingrs)
            self.process_predicted_ingredients()

        self.compute_prior_ingr_unit_portion(do_print=('test' in splits))
        self.create_weight_matrix()

        self.qids = []
        self.rids = self.ids
        for qid in self.qid_to_rid:
            if self.qid_to_rid[qid] in self.rids:
                self.qids.append(qid)

        conversions = Conversions()
        self.flattened_weights = conversions.portion_to_calorie(self.qid_to_unitportion, self.ingr_to_idx, self.idx_to_ingr, self.unit_to_idx, self.idx_to_unit)

        self.ingr_calorie_stats = IngrCalorieStats(self.qid_to_calorie, self.qid_to_ingr)

        # set transformation operator for image transformation
        self.transforms = transforms
        self.image_encoder_type = args.image_encoder_type
        self.pre_load = args.pre_load
        self.img_pad_size = args.img_pad_size
        # create root dir for image paths
        self.root = os.path.join(data_dir, 'images')

        self.bottom_up_feat_size, self.iid_to_img_feat_path, self.iid_to_img_feat, self.iid_to_impaths, self.iid_to_image, self.iid_to_image_features = prepare_image_features(splits, aux_data_dir, data_dir, suff, args, self.rid_to_data.keys(), self.run_mode, self.root, self.transforms)

        self.shuffle()
        self.data_size = len(self.ids)

        if self.evaluate_given_images:
            self.make_images_dataset(args.images_folder, args.image_size)

        print ("Data loader in {} mode!".format(self.run_mode))
        print ("Number of dish tokens: {}.".format(self.get_dish_vocab_size()))
        print ("Number of input tokens: {}.".format(self.get_ingrs_vocab_size()))
        print ("Number of output tokens: {}.".format(self.get_unit_vocab_size()))
        print ("Data size: {}.".format(self.data_size))
        print (" ---------- --------- ---------- \n")

    def make_images_dataset(self, images_folder, image_size, images_preds = "data/images_predictions.pkl", fixed_ingrs_path="data/iid_to_ingredients_foon.json"):
        '''
            Make a dataset based on the given predicted ingredients dataset (subset of the origianl Recipe1M dataset)
        '''
        fixed_ingrs = json.load(open(fixed_ingrs_path, 'r'))

        predictions = pickle.load(open(images_preds, 'rb'))
        new_predictions = {}
        for item in predictions:
            new_predictions[item['id']] = item['ingrs']
            if item['id'] in fixed_ingrs:
                new_predictions[item['id']] = fixed_ingrs[item['id']]

        self.predicted_ingrs = {}

        image_paths = os.listdir(images_folder)
        # select the first N id to images for the current images in the folder
        ids = list(self.iid_to_image.keys())[:len(image_paths)]

        # just to resize the image
        resize_transform = transforms.Compose([transforms.Resize((image_size))])

        self.iid_to_imgnameid = {}
        count = 0
        for img in image_paths:
            image = Image.open(os.path.join(images_folder, img)).convert('RGB')
            self.iid_to_image[ids[count]] = [resize_transform(image)]
            self.predicted_ingrs[ids[count]] = new_predictions[img]
            self.iid_to_imgnameid[ids[count]] = img
            count += 1

        # take all predicted ingredients from file
        if self.use_predicted_ingrs:
            self.process_predicted_ingredients()

        # remove all but the ids for current images
        self.iid_to_image = keep_specified_ids(self.iid_to_image, ids)
        self.iid_to_impaths = keep_specified_ids(self.iid_to_impaths, ids)
        self.rid_to_data = keep_specified_ids(self.rid_to_data, ids)

    def process_predicted_ingredients(self):

        for iid in self.predicted_ingrs:
            for ingr in self.predicted_ingrs[iid]:
                if not ingr in self.ingr_to_idx:
                    words = ingr.split('_')
                    ingr = ""
                    words.reverse()
                    for w in words:
                        ingr += w+"_"
                    ingr = ingr.strip('_')

                idx = self.ingr_to_idx[ingr]
                ingr_id = '{}_id_{}'.format(iid, idx)
                self.qid_to_rid[ingr_id] = iid

        self.ids2 = []
        for iid in self.ids:
            if iid in self.predicted_ingrs:
                self.ids2.append(iid)

        self.ids = self.ids2

    def unscale_batch(self, scaled_batch, input_type="calorie"):

        if input_type=="calorie":
            return unscale_batch(scaled_batch, self.min_stat, self.max_stat)
        elif input_type=="calories":
            return unscale_batch(scaled_batch, self.scale_dict["min_calorie"], self.scale_dict["max_calorie"])
        else: # if portions
            return unscale_batch(scaled_batch, self.scale_dict["min_ns_portion"], self.scale_dict["max_ns_portion"])

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

    def get_state_vocab_size(self):
        return len(self.state_to_idx)

    def get_filtered_ids(self):
        return self.qids

    def compute_ingr_unit_portion_error(self, qid_to_unitportion_preds, prec=2):
        '''
            Compute the error of calorie and portion estimation
        '''
        gt_portion_maes = collections.defaultdict(float)
        gt_calorie_maes = collections.defaultdict(float)

        portion_maes = collections.defaultdict(float)
        calorie_maes = collections.defaultdict(float)
        totals = collections.defaultdict(float)

        for qid in qid_to_unitportion_preds:
            pred_unit, pred_portion, pred_calorie = qid_to_unitportion_preds[qid]

            ingr, gt_unit, unit_idx, gt_portion, gt_calorie = self.qid_to_unitportion[qid]

            if pred_unit==gt_unit:

                portion_maes[gt_unit] += abs(gt_portion-pred_portion)
                portion_maes["total"] += abs(gt_portion-pred_portion)

                calorie_maes[gt_unit] += abs(gt_calorie-pred_calorie)
                calorie_maes["total"] += abs(gt_calorie-pred_calorie)

                totals[gt_unit] += 1
                totals["total"] += 1

                # mean ground-truth errors
                gt_portion_maes[gt_unit] += abs(self.ingrunit_to_meanportion[ingr][gt_unit]-pred_portion)
                gt_calorie_maes[gt_unit] += abs(self.ingrunit_to_meancalorie[ingr][gt_unit]-pred_calorie)

        print ('-----------------')
        print ("portion errors: ")
        for key in portion_maes:
            print ("{} {}: pred: {} vs gt: {}".format(totals[key], key, round(portion_maes[key]/totals[key], prec), round(gt_portion_maes[key]/totals[key], prec)))
        print ("calorie errors: ")
        for key in calorie_maes:
            print ("{} {}: pred: {} vs gt: {}".format(totals[key], key, round(calorie_maes[key]/totals[key], prec), round(gt_calorie_maes[key]/totals[key], prec)))
        print ('-----------------')

    def compute_prior_ingr_unit_portion(self, do_print=False, prec=3):
        '''
            Computes the prior ingredient portion and calorie per ingredient-unit tuple.
        '''

        self.unit_to_meancalorie = collections.defaultdict(float)
        self.unit_to_meanportion = collections.defaultdict(float)
        self.unit_to_count = collections.defaultdict(float)

        self.ingrunit_to_minportion = collections.defaultdict(lambda: collections.defaultdict(float))
        self.ingrunit_to_maxportion = collections.defaultdict(lambda: collections.defaultdict(float))

        self.ingrunit_to_meancalorie = collections.defaultdict(lambda: collections.defaultdict(float))
        self.ingrunit_to_meanportion = collections.defaultdict(lambda: collections.defaultdict(float))
        self.ingrunit_to_count = collections.defaultdict(lambda: collections.defaultdict(float))

        # pile up portions and calories per unit and per (unit-ingr)
        for qid in self.qid_to_unitportion:

            ingr, unit, unit_idx, portion, calorie, state = self.qid_to_unitportion[qid]

            self.ingrunit_to_minportion[ingr][unit] = min(portion, self.ingrunit_to_minportion[ingr][unit])
            self.ingrunit_to_maxportion[ingr][unit] = max(portion, self.ingrunit_to_maxportion[ingr][unit])

            self.ingrunit_to_meanportion[ingr][unit] += portion
            self.unit_to_meanportion[unit] += portion

            self.ingrunit_to_meancalorie[ingr][unit] += calorie
            self.unit_to_meancalorie[unit] += calorie

            self.ingrunit_to_count[ingr][unit] += 1.0
            self.unit_to_count[unit] += 1.0

        # compute mean portion and mean calorie per (unit-ingr)
        for ingr in self.ingrunit_to_meanportion:
            for unit in self.ingrunit_to_meanportion[ingr]:

                self.ingrunit_to_meanportion[ingr][unit] /= self.ingrunit_to_count[ingr][unit]
                self.ingrunit_to_meanportion[ingr][unit] = round(self.ingrunit_to_meanportion[ingr][unit], prec)

                self.ingrunit_to_meancalorie[ingr][unit] /= self.ingrunit_to_count[ingr][unit]
                self.ingrunit_to_meancalorie[ingr][unit] = round(self.ingrunit_to_meancalorie[ingr][unit], prec)

        # compute mean portion and mean calorie per unit
        for unit in self.unit_to_meancalorie:

            self.unit_to_meancalorie[unit] = round(self.unit_to_meancalorie[unit]/self.unit_to_count[unit], prec)
            self.unit_to_meanportion[unit] = round(self.unit_to_meanportion[unit]/self.unit_to_count[unit], prec)

        self.error_ingrunit_to_meancalorie = collections.defaultdict(float)
        self.error_ingrunit_to_meanportion = collections.defaultdict(float)

        self.error_unit_to_meancalorie = collections.defaultdict(float)
        self.error_unit_to_meanportion = collections.defaultdict(float)

        self.error_total_portion_ingrunit = 0.0
        self.error_total_portion_unit = 0
        self.error_total_calorie_ingrunit = 0.0
        self.error_total_calorie_unit = 0

        total = len(self.qid_to_unitportion)
        # compute mean error per unit and per (unit-ingr)
        for qid in self.qid_to_unitportion:

            ingr, unit, unit_idx, portion, calorie, state = self.qid_to_unitportion[qid]
            self.error_ingrunit_to_meancalorie[unit] += abs(self.ingrunit_to_meancalorie[ingr][unit]-calorie)/self.unit_to_count[unit]
            self.error_ingrunit_to_meanportion[unit] += abs(self.ingrunit_to_meanportion[ingr][unit]-portion)/self.unit_to_count[unit]
            self.error_total_calorie_ingrunit += abs(self.ingrunit_to_meancalorie[ingr][unit]-calorie)/total
            self.error_total_portion_ingrunit += abs(self.ingrunit_to_meanportion[ingr][unit]-portion)/total
            
            self.error_unit_to_meancalorie[unit] += abs(self.unit_to_meancalorie[unit]-calorie)/self.unit_to_count[unit]
            self.error_unit_to_meanportion[unit] += abs(self.unit_to_meanportion[unit]-portion)/self.unit_to_count[unit]
            self.error_total_calorie_unit += abs(self.unit_to_meancalorie[unit]-calorie)/total
            self.error_total_portion_unit += abs(self.unit_to_meanportion[unit]-portion)/total
            
        if do_print:
            print (' =========== unit - calorie portion priors =========== ')
            print ("Prior calorie errors of ingr-units:")
            for unit in self.error_ingrunit_to_meancalorie:
                print ("{}: {}".format(unit, round(self.error_ingrunit_to_meancalorie[unit],prec)))
            print ("Prior calorie errors of units:")
            for unit in self.error_unit_to_meancalorie:
                print ("{}: {}".format(unit, round(self.error_unit_to_meancalorie[unit],prec)))
            print ("Prior portion errors of ingr-units:")
            for unit in self.error_ingrunit_to_meanportion:
                print ("{}: {}".format(unit, round(self.error_ingrunit_to_meanportion[unit],prec)))
            print ("Prior portion errors of units:")
            for unit in self.error_unit_to_meanportion:
                print ("{}: {}".format(unit, round(self.error_unit_to_meanportion[unit],prec)))

            print ("totals:")
            print ("for ingr-unit: calorie error: {}, portion error: {}".format(round(self.error_total_calorie_ingrunit, prec), round(self.error_total_portion_ingrunit, prec)))
            print ("for unit: calorie error: {}, portion error: {}".format(round(self.error_total_calorie_unit, prec), round(self.error_total_portion_unit, prec)))
            print (self.unit_to_count)
            for unit in self.unit_to_count:
                print (unit, self.unit_to_idx[unit])
            print (' =========== ==== = ======= ====== =========== ')

    def create_weight_matrix(self):
        '''
            Compute weights for cross-entropy loss based on frequency of classes in the dataset.
        '''

        weights = np.ones(len(self.unit_to_count))
        total = 1.2*max(self.unit_to_count.values())
        for unit in self.unit_to_count:
            weights[self.unit_to_idx[unit]] = total/self.unit_to_count[unit]

        weights = 10.0 * weights / max(weights)
        self.weights_tensor = torch.from_numpy(weights).float().to(device)

        print ("The weights for cross-entropy loss is: {}".format(self.weights_tensor))

    def get_ingr_calorie_mean(self, ingr):
        return self.ingr_calorie_stats.get_mean(ingr)

    def ingr_prediction_stats_update(self, ingr, calorie_prediction_error):
        self.ingr_calorie_stats.update_errors(ingr, calorie_prediction_error)

    def is_pad_token(self, idx):
        if idx==self.ingr_to_idx['<pad>']:
            return True
        return False

    def __getitem__(self, idx):
        """
            Returns one image, ingredients, units, portions associated with that image
        """

        iid = self.ids[idx]

        split = self.rid_to_data[iid][0]

        # Load the run data from list
        pairs = self.rid_to_data[iid][1:]

        ingr_labels_gt = np.ones(self.maxnumingrs) * self.ingr_to_idx['<pad>']
        unit_labels_gt = np.ones(self.maxnumingrs) * self.unit_pad_value
        state_labels_gt = np.ones(self.maxnumingrs) * self.state_pad_value

        portion_labels_gt = np.zeros(self.maxnumingrs)
        calorie_labels_gt = np.zeros(self.maxnumingrs)
        calorie_gt = float(self.iid_to_calorie[iid])
        dish_gt = self.iid_to_dishname[iid]
        dish_idx = self.dish_to_idx[dish_gt]

        pos = 0
        qids = []
        for pair in pairs:
            ingr_labels_gt[pos] = self.ingr_to_idx[pair[0]]
            unit_labels_gt[pos] = self.unit_to_idx[pair[1]]
            portion_labels_gt[pos] = float(pair[2])
            calorie_labels_gt[pos] = float(pair[3])
            qids.append(pair[4])
            state_labels_gt[pos] = self.state_to_idx[pair[5]]
            pos += 1

        # if using predicted ingredients as input instead of ground-truth ingredients
        if self.use_predicted_ingrs:
            pos = 0
            ingr_labels_gt = np.ones(self.maxnumingrs) * self.ingr_to_idx['<pad>']
            unit_labels_gt = np.ones(self.maxnumingrs) * self.unit_pad_value
            state_labels_gt = np.ones(self.maxnumingrs) * self.state_pad_value
            portion_labels_gt = np.zeros(self.maxnumingrs)
            calorie_labels_gt = np.zeros(self.maxnumingrs)

            qids = []
            for ingr in self.predicted_ingrs[iid]:
                if not ingr in self.ingr_to_idx:
                    words = ingr.split('_')
                    ingr = ""
                    words.reverse()
                    for w in words:
                        ingr += w+"_"
                    ingr = ingr.strip('_')

                ingr_idx = self.ingr_to_idx[ingr]
                ingr_labels_gt[pos] = ingr_idx

                unit_labels_gt[pos] = self.unit_to_idx['ounce'] # default unit
                portion_labels_gt[pos] = float(2) # default portion
                calorie_labels_gt[pos] = float(50) # default calorie

                qids.append('{}_id_{}'.format(iid, ingr_idx))

                self.one_pass_counter += 1
                state_labels_gt[pos] = self.state_to_idx['whole'] # default state

                pos += 1

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

        ingr_labels_gt = torch.from_numpy(ingr_labels_gt).long()
        unit_labels_gt = torch.from_numpy(unit_labels_gt).long()
        portion_labels_gt = torch.from_numpy(portion_labels_gt).float()
        calorie_labels_gt = torch.from_numpy(calorie_labels_gt).float()
        state_labels_gt = torch.from_numpy(state_labels_gt).long()

        return ingr_labels_gt, unit_labels_gt, portion_labels_gt, calorie_labels_gt, state_labels_gt, calorie_gt, dish_idx, image_input, iid, qids, path

    def __len__(self):
        return len(self.ids)

# -------------------------------------------

def collate_fn(data):
    '''
        Collate function for the pytorch data-loader class (stacks given data from the get-item function).
    '''

    # Sort a data list by caption length (descending order).
    ingr_labels_gt, unit_labels_gt, portion_labels_gt, calorie_labels_gt, state_labels_gt, calorie_gt, dish_idx, image_input, img_id, qids, path = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    image_input = torch.stack(image_input, 0)

    ingr_labels_gt = torch.stack(ingr_labels_gt, 0)
    unit_labels_gt = torch.stack(unit_labels_gt, 0)
    state_labels_gt = torch.stack(state_labels_gt, 0)
    portion_labels_gt = torch.stack(portion_labels_gt, 0)
    calorie_labels_gt = torch.stack(calorie_labels_gt, 0)
    calorie_gt = torch.tensor(calorie_gt)
    dish_idxs = torch.tensor(dish_idx)

    return ingr_labels_gt, unit_labels_gt, portion_labels_gt, calorie_labels_gt, state_labels_gt, calorie_gt, None, None, None, dish_idxs, image_input, img_id, qids, path

# -------------------------------------------

def get_recipe_loader(data_dir, aux_data_dir, splits, run_mode, transforms, batch_size,
               shuffle, num_workers, drop_last=False,
               suff='',
               args=None):

    '''
        Creates the dataset and the data loader for the train & 
        sample phases to load the instances of the dataset from.
    '''

    dataset = Recipe1MDataset(data_dir=data_dir, aux_data_dir=aux_data_dir, splits=splits,
                              run_mode=run_mode,
                              transforms=transforms,
                              suff=suff,
                              args=args)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                              drop_last=drop_last, collate_fn=collate_fn, pin_memory=True)
    return data_loader, dataset

# -------------------------------------------
