# Copyright (c) Robotic and Action Perception Lab (RPAL) at University of South Florida
# by Ahmad Babaeian Jelodar

'''
    Runs the model over the test set to evaluate the model.
'''

import torch
import numpy as np
from args import get_parser
import pickle
import os
from torchvision import transforms
from model import get_model, label2onehot
import json
import sys
import random
from modules.utils import LogResults, Accuracy, store_features, maintain_cnn_features, print_arg_info, unit_test_portions_units
from data_modules.recipe_loader import get_recipe_loader
from data_modules.ingr_loader import get_ingr_loader
from data_modules.data_aux import transformation_functions

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
map_loc = None if torch.cuda.is_available() else 'cpu'

# -------------------------------------------

def main(args):

    checkpoints_dir = args.save_dir
    args.use_predicted = True

    vars_to_replace = ['batch_size', 'eval_split', 'save_dir', 'aux_data_dir',
                       'recipe1m_dir', 'use_lmdb', 'ckpt_epoch', 'subsequent_calorie',
                       'use_predicted', 'evaluate_given_images', 'images_folder']

    store_dict = {}
    for var in vars_to_replace:
        store_dict[var] = getattr(args, var)

    if not args.store_cnn_feat:
        do_calorie_scale = args.do_calorie_scale
        splits = args.splits
        model_output = args.model_output
        subsequent_calorie = args.subsequent_calorie
        args = pickle.load(open(os.path.join(checkpoints_dir, 'args.pkl'), 'rb'))
        args.splits = splits
        args.do_calorie_scale = do_calorie_scale
        args.model_output = model_output
        args.subsequent_calorie = subsequent_calorie

    for var in vars_to_replace:
        setattr(args, var, store_dict[var])

    model_path = os.path.join(checkpoints_dir, 'model{}.ckpt'.format(args.ckpt_epoch))

    torch.set_printoptions(profile="full")

    data_dir = args.recipe1m_dir

    train_transform = transformation_functions(args, run_mode="train")
    eval_transform = transformation_functions(args, run_mode="eval")
    transforms = {"train": train_transform, "eval": eval_transform}

    if "simple" in args.model_type:
        is_simple = True
        data_loader, dataset = get_ingr_loader(data_dir, args.aux_data_dir, args.splits,
                                      run_mode='eval', transforms=transforms, 
                                      batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                      drop_last=False, suff=args.suff, args=args)
    else:
        is_simple = False
        data_loader, dataset = get_recipe_loader(data_dir, args.aux_data_dir, args.splits, 
                                      run_mode='eval', transforms=transforms, 
                                      batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                      drop_last=False, suff=args.suff, args=args)


    ingr_vocab_size = dataset.get_ingrs_vocab_size()
    if args.dataset=="recipe1m":
        unit_vocab_size = dataset.get_unit_vocab_size()
    else:
        unit_vocab_size = -1
    dish_vocab_size = dataset.get_dish_vocab_size()
    states_vocab_size = dataset.get_state_vocab_size()
    calorie_classes_num = dataset.get_calorie_classes_num()

    # Build the model
    model = get_model(args, ingr_vocab_size, unit_vocab_size, states_vocab_size, calorie_classes_num, dish_vocab_size, dataset.bottom_up_feat_size, dataset.pretrained_emb, loss_weights_matrix=dataset.weights_tensor, flattened_weights=dataset.flattened_weights)

    if not args.store_cnn_feat:
        # Load the trained model parameters
        model.load_state_dict(torch.load(model_path, map_location=map_loc))
        print ("loaded epoch {} checkpoint of the model.".format(args.ckpt_epoch))

    log_results = LogResults(dataset, is_simple, dataset.run_mode, args.evaluate_given_images)

    # prints settings of the current run
    print_arg_info(args)

    if args.just_plot_error:
        log_results.plot_errors()
        exit()

    model = model.to(device)
    model.eval()

    # dictionary for storing cnn features for each instance in the dataset
    if args.store_cnn_feat:
        imgid_to_features = {}
        stored_imgids = set({})
        store_batch_num = 1

    accuracy_log = Accuracy(args, dataset.data_size, is_simple, dataset.run_mode, dataset.unscale_batch)
    for step, (ques_idx, tar_idx, tar_values, tar_calories, tar_states, tar_calorie, mean_prior_calories_gt, portion_distrib, calorie_class_idx, dish_idxs, image_input, img_id, qids, path) in enumerate(data_loader):

        if args.store_cnn_feat:
            update_batch_num = store_features(args.features_dir, dataset.run_mode, imgid_to_features, store_batch_num)
            if update_batch_num!=store_batch_num:
                store_batch_num = update_batch_num
                imgid_to_features = {}

        # ---------- move all data loaded from dataloader to gpu
        # input indice referring to ingredients
        if not ques_idx is None:
            ques_idx = ques_idx.to(device)
        # input indice referring to a dish name
        if not dish_idxs is None:
            dish_idxs = dish_idxs.to(device)
        # target state idx
        if not tar_states is None:
            tar_states = tar_states.to(device)
        # input image or image features
        if not image_input is None:
            image_input = image_input.to(device)
        # target indice referring to unit classes (N=6)
        if not tar_idx is None:
            tar_idx = tar_idx.to(device)
        # target portion values per input (ingredient)
        if not tar_values is None:
            tar_values = tar_values.to(device)
        # target portion outputs referring to each unit type (N=6)
        if not portion_distrib is None:
            portion_distrib = portion_distrib.to(device)
        # target calorie class associated with the calorie value.
        if not calorie_class_idx is None:
            calorie_class_idx = calorie_class_idx.to(device)
        # target per recipe calorie
        if not tar_calorie is None:
            tar_calorie = tar_calorie.to(device)
        # individual calories referring to ingredients
        if not tar_calories is None:
            tar_calories = tar_calories.to(device)
        # ----------
        if not mean_prior_calories_gt is None:
            mean_prior_calories_gt = mean_prior_calories_gt.to(device)

        if "simple" in args.model_type:
            ingr_calorie_vec_mask = label2onehot(ques_idx, dataset.pad_value)
            ingr_calorie_vec = label2onehot(ques_idx, dataset.pad_value, tar_calorie.squeeze())
        else:
            ingr_calorie_vec, ingr_calorie_vec_mask = None, None

        # create a mask for image featuers if the number of image feature vectors used is variable
        if args.image_encoder_type=="bottom-up":
            image_mask = (torch.sum(torch.abs(image_input), dim=-1) == 0)
        else:
            image_mask = None

        # create mask for ingredients
        ques_mask = (ques_idx == dataset.pad_value)

        # feed-forward data in the model
        losses, outputs = model(ques_idx, ques_mask, None, tar_idx, tar_values, tar_calorie, tar_calories, tar_states,
                                mean_prior_calories_gt, portion_distrib, dish_idxs, image_input, image_mask,
                                ingr_calorie_vec_mask, ingr_calorie_vec, calorie_class_idx=calorie_class_idx)

        if args.store_cnn_feat:
            maintain_cnn_features(img_id, outputs, imgid_to_features, stored_imgids)
            continue

        # logging acuracy
        pred_idx = accuracy_log.update(outputs, tar_idx, tar_values, tar_calorie, tar_calories, tar_states, ques_mask)

        log_results.update(outputs, pred_idx, tar_idx, tar_values, tar_calorie, tar_calories, tar_states, qids, img_id, ques_idx, is_simple)

    if args.store_cnn_feat:
        store_features(args.features_dir, dataset.run_mode, imgid_to_features, store_batch_num, last_batch=True)

    log_results.finalize(args, is_simple)

    # mean accuracy over all testing data
    accuracy_log.epoch_finalize()

    print ('******************* Testing Ended *******************')

if __name__ == '__main__':
    args = get_parser()
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)
    main(args)
