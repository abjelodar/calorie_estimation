# Copyright (c) Robotic and Action Perception Lab (RPAL) at University of South Florida
# by Ahmad Babaeian Jelodar

'''
    Runs the model over the train set to train the model.
'''

from args import get_parser
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import os, random
import pickle, json
from model import get_model, label2onehot
import torch.backends.cudnn as cudnn
from modules.utils import LogResults, Accuracy, Optim, print_arg_info, count_parameters, save_model, maintain_cnn_features, store_features, create_ingr_portion_mask, unit_test_portions_units
from data_modules.data_aux import transformation_functions
from data_modules.recipe_loader import get_recipe_loader
from data_modules.ingr_loader import get_ingr_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
map_loc = None if torch.cuda.is_available() else 'cpu'

# -------------------------------------------

def main(args):
    
    torch.set_printoptions(profile="full")

    drop_last = False
    # dictionary for storing cnn features for each instance in the dataset
    if args.store_cnn_feat:
        drop_last = False
        imgid_to_features = {}
        store_batch_num = 1

    checkpoints_dir = args.save_dir
    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)

    # initialize coefficients for loss trade-off
    loss_lambda = args.loss_lambda

    # Build data loader
    data_dir = args.recipe1m_dir

    train_transform = transformation_functions(args, run_mode="train")
    eval_transform = transformation_functions(args, run_mode="eval")
    transforms = {"train": train_transform, "eval": eval_transform}

    if "simple" in args.model_type:
        is_simple = True
        data_loader, dataset = get_ingr_loader(data_dir, args.aux_data_dir, args.splits, 
                                      run_mode='train', transforms=transforms, 
                                      batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                      drop_last=drop_last, suff=args.suff, args=args)
    else:
        is_simple = False
        data_loader, dataset = get_recipe_loader(data_dir, args.aux_data_dir, args.splits, 
                                      run_mode='train', transforms=transforms, 
                                      batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                      drop_last=drop_last, suff=args.suff, args=args)

    ingr_vocab_size = dataset.get_ingrs_vocab_size()
    unit_vocab_size = dataset.get_unit_vocab_size()
    dish_vocab_size = dataset.get_dish_vocab_size()
    states_vocab_size = dataset.get_state_vocab_size()
    calorie_classes_num = dataset.get_calorie_classes_num()

    # prints settings of the current run
    print_arg_info(args)

    # Build the model
    model = get_model(args, ingr_vocab_size, unit_vocab_size, states_vocab_size, calorie_classes_num, dish_vocab_size, dataset.bottom_up_feat_size, dataset.pretrained_emb, loss_weights_matrix=dataset.weights_tensor, flattened_weights=dataset.flattened_weights)

    print ("Number of model parameters is: {}  /  {}.".format(*count_parameters(model)))

    # create optimizer
    optim = Optim(args, model, dataset.data_size, is_simple, decay_factor=1.0)

    print ("cnn gradients: {}.".format(args.keep_cnn_gradients))

    if device != 'cpu' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    model.train()

    print ("model created & starting training ...")

    # create class for accuracy and loss computation
    accuracy_log = Accuracy(args, dataset.data_size, is_simple, dataset.run_mode, dataset.unscale_batch)
    # create class for logging
    log_results = LogResults(dataset, is_simple, dataset.run_mode, args.evaluate_given_images) 

    start_epoch = 0
    # Training script
    for epoch in range(start_epoch, args.num_epochs):

        # shuffle training dataset
        dataset.shuffle()

        # updates only if warm-up is not used
        optim.update_optimizer_epoch(epoch)

        # initialize variables for loss and accuracy reporting
        accuracy_log.initialize(epoch)

        # step loop
        for step, (ques_idx, tar_idx, tar_values, tar_calories, tar_states, tar_calorie, mean_prior_calories_gt, portion_distrib, calorie_class_idx, dish_idxs, image_input, img_id, qids, path) in enumerate(data_loader):

            # update optimizer (lr) only if warm-up is used
            optim.update_optimizer_step(step, epoch)

            # ---------- move all data loaded from dataloader to gpu
            # input indice referring to ingredients
            if not ques_idx is None:
                ques_idx = ques_idx.to(device)
            # input indice referring to a dish name
            if not dish_idxs is None:
                dish_idxs = dish_idxs.to(device)
            # input image or image features
            if not image_input is None:
                image_input = image_input.to(device)
            # target indice referring to unit classes (N=6)
            if not tar_idx is None:
                tar_idx = tar_idx.to(device)
            # target state idx
            if not tar_states is None:
                tar_states = tar_states.to(device)
            # target portion values per input (ingredient)
            if not tar_values is None:
                tar_values = tar_values.to(device)
            # target portion outputs referring to each unit type (N=6)
            if not portion_distrib is None:
                portion_distrib = portion_distrib.to(device)
            # target calorie class associated with the calorie value.
            if not calorie_class_idx is None:
                calorie_class_idx = calorie_class_idx.to(device)
            # target per recipe calorie (or ingredient calorie when in simple mode)
            if not tar_calorie is None:
                tar_calorie = tar_calorie.to(device)
            # individual calories referring to ingredients (or recipe caloire when in simple mode)
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
            ingr_portion_mask = create_ingr_portion_mask(ingr_vocab_size, unit_vocab_size, ques_idx, args.batch_size, args.maxnumingrs)

            if args.use_distrib_loss:
                # load an associated pair for each item in the original batch if doing kl-divergence loss
                paired_ques_idx, paired_tar_idx, paired_dish_idxs, paired_image_input = dataset.get_pairs(qids)

                # move the paired data to gpu
                paired_ques_idx = paired_ques_idx.to(device)
                paired_tar_idx = paired_tar_idx.to(device)
                paired_dish_idxs = paired_dish_idxs.to(device)
                paired_image_input = paired_image_input.to(device)

            else:
                paired_ques_idx, paired_tar_idx, paired_dish_idxs, paired_image_input = None, None, None, None

            # feed-forward data in the model
            losses, outputs = model(ques_idx, ques_mask, ingr_portion_mask, tar_idx, tar_values, tar_calorie, tar_calories, tar_states, mean_prior_calories_gt,
                                    portion_distrib, dish_idxs, image_input, image_mask, ingr_calorie_vec_mask, ingr_calorie_vec, calorie_class_idx,
                                    args.keep_cnn_gradients, paired_ques_idx, paired_tar_idx, paired_dish_idxs, paired_image_input)

            if args.store_cnn_feat:
                maintain_cnn_features(img_id, outputs, imgid_to_features)
                continue

            loss_names = []
            # compute losses
            loss_array = []
            if "unit_loss" in losses.keys(): 
                losses["unit_loss"] = losses["unit_loss"].mean()
                loss_array.append(losses["unit_loss"])
                loss_names.append("unit_loss")
            if "portion_loss" in losses.keys():
                losses["portion_loss"] = losses["portion_loss"].mean()
                loss_array.append(losses["portion_loss"])
                loss_names.append("portion_loss")
            if "calorie_loss" in losses.keys():
                losses["calorie_loss"] = losses["calorie_loss"].mean()
                loss_array.append(losses["calorie_loss"])
                loss_names.append("calorie_loss")
            if "recipe_calorie_loss" in losses.keys():
                losses["recipe_calorie_loss"] = losses["recipe_calorie_loss"].mean()
                loss_array.append(losses["recipe_calorie_loss"])
                loss_names.append("recipe_calorie_loss")
            if "prior_calorie_loss" in losses.keys():
                losses["prior_calorie_loss"] = losses["prior_calorie_loss"].mean()
                loss_array.append(losses["prior_calorie_loss"])
                loss_names.append("prior_calorie_loss")
            if "calories_loss" in losses.keys():
                losses["calories_loss"] = losses["calories_loss"].mean()
                loss_array.append(losses["calories_loss"])
                loss_names.append("calories_loss")
            if "calories_alignment_loss" in losses.keys():
                losses["calories_alignment_loss"] = losses["calories_alignment_loss"].mean()
                loss_array.append(losses["calories_alignment_loss"])
                loss_names.append("calories_alignment_loss")
            if "calorie_alignment_loss" in losses.keys():
                losses["calorie_alignment_loss"] = losses["calorie_alignment_loss"].mean()
                loss_array.append(losses["calorie_alignment_loss"])
                loss_names.append("calorie_alignment_loss")
            if "calorie_alignment2_loss" in losses.keys():
                losses["calorie_alignment2_loss"] = losses["calorie_alignment2_loss"].mean()
                loss_array.append(losses["calorie_alignment2_loss"])
                loss_names.append("calorie_alignment2_loss")
            if "ingr_portion_loss" in losses.keys():
                losses["ingr_portion_loss"] = losses["ingr_portion_loss"].mean()
                loss_array.append(losses["ingr_portion_loss"])
                loss_names.append("ingr_portion_loss")
            if "portion_distrib_loss" in losses.keys():
                losses["portion_distrib_loss"] = losses["portion_distrib_loss"].mean()
                loss_array.append(losses["portion_distrib_loss"])
                loss_names.append("portion_distrib_loss")
            if "distrib_kl_loss" in losses.keys():
                losses["distrib_kl_loss"] = losses["distrib_kl_loss"].mean()
                loss_array.append(losses["distrib_kl_loss"])
                loss_names.append("distrib_kl_loss")
            if "calorie_class_loss" in losses.keys():
                losses["calorie_class_loss"] = losses["calorie_class_loss"].mean()
                loss_array.append(losses["calorie_class_loss"])
                loss_names.append("calorie_class_loss")
            if "state_loss" in losses.keys(): 
                losses["state_loss"] = losses["state_loss"].mean()
                loss_array.append(losses["state_loss"])
                loss_names.append("state_loss")

            loss = 0
            for i in range(len(loss_array)):
                loss += loss_lambda[i] * loss_array[i]

            step_loss = loss.item()

            # back-propagate the loss in the model & optimize
            model.zero_grad()
            loss.backward()
            optim.step()

            # logging acuracy & loss
            pred_idx = accuracy_log.update(outputs, tar_idx, tar_values, tar_calorie, tar_calories, tar_states, ques_mask, args.batch_size, losses, step_loss)

            log_results.update(outputs, pred_idx, tar_idx, tar_values, tar_calorie, tar_calories, tar_states, qids, img_id, ques_idx, is_simple)

        if args.store_cnn_feat:
            store_features(args.features_dir, dataset.run_mode, imgid_to_features, store_batch_num, last_batch=True)
            break

        # finished executing all steps of this epoch

        # logging actual results into file
        if epoch+1 in args.saving_epochs:
            save_model(model, optim.optimizer, args, checkpoints_dir, suff=str(epoch+1))

        log_results.finalize(args, is_simple)

        # mean accuracy over all training data
        accuracy_log.epoch_finalize(optim.learning_rate)

    save_model(model, optim.optimizer, args, checkpoints_dir, suff='last')
    print ('******************* Training Ended *******************')

if __name__ == '__main__':
    args = get_parser()
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)
    main(args)
