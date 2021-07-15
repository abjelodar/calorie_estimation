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
import copy, os, time, pickle, json, collections
from modules.eval import calorie_by_separate_ingredients_calories, calorie_by_separate_results, plot_2d_error

# =======================================

class Accuracy:

    def __init__(self, args, data_size, is_simple, run_mode, unscale_batch=None, root="logs"):

        self.args = args
        self.unscale_batch = unscale_batch

        if args.scaled_output:
            self.scaled = True
        else:
            self.scaled = False

        self.run_mode = run_mode

        self.data_size = data_size
        self.is_simple = is_simple
        self.initialize(0)

    def initialize(self, epoch):

        self.epoch = epoch
        self.total_correct = 0.0
        self.total_ingredients = 0.0
        self.calorie_mse = 0.0
        self.calories_mse = 0.0
        self.portion_mse = 0.0
        self.total = 0.0
        if self.run_mode=="train":
            self.epoch_loss = 0.0
            self.unit_loss = 0.0
            self.portion_loss = 0.0
            self.calorie_loss = 0.0
            self.portion_distrib_loss = 0.0
        self.start_time = time.time()

    def compute_summed_mse(self, out_calorie, tar_calorie, input_type="calorie"):

        tar_calorie = torch.reshape(tar_calorie, out_calorie.size())
        abs_error = torch.abs( torch.sub(out_calorie, tar_calorie) )
        if self.scaled:
            abs_error = self.unscale_batch(abs_error, input_type)
            out = self.unscale_batch(out_calorie, input_type)
            tar = self.unscale_batch(tar_calorie, input_type)

        # if not scaled output
        return torch.sum( abs_error ).item()

    def update(self, outputs, tar_idx, tar_values, tar_calorie, tar_calories, tar_states, ques_mask, batch_size=32, losses=None, step_loss=None):

        # total instances in this batch
        self.total += batch_size

        if self.run_mode=="train":
            self.epoch_loss += step_loss
            if "portion_loss" in losses.keys():
                self.portion_loss += losses["portion_loss"].item()
            if "unit_loss" in losses.keys():
                self.unit_loss += losses["unit_loss"].item()
            if "calorie_loss" in losses.keys():
                self.calorie_loss += losses["calorie_loss"].item()
            if "portion_distrib_loss" in losses.keys():
                self.portion_distrib_loss += losses["portion_distrib_loss"].item()

        total_batch_ingredients = (~ques_mask).float().sum().item()
        self.total_ingredients += total_batch_ingredients

        if "calorie_output" in outputs.keys():
            self.calorie_mse += self.compute_summed_mse(outputs["calorie_output"], tar_calorie, input_type="calorie")

        if "calories_output" in outputs.keys():
            self.calories_mse += self.compute_summed_mse(outputs["calories_output"], tar_calories, input_type="calories")

        if "portion_output" in outputs.keys():
            self.portion_mse += self.compute_summed_mse(outputs["portion_output"], tar_values, input_type="portion")

        if "unit_output" in outputs:
            _, pred_idx = torch.max(outputs["unit_output"], dim=1)

            total_batch_corrects = torch.sum(pred_idx==tar_idx).item()
            self.total_correct += total_batch_corrects
            return pred_idx

        if "state_output" in outputs:
            _, pred_idx = torch.max(outputs["state_output"], dim=1)
            total_batch_corrects = torch.sum(pred_idx==tar_states).item()
            self.total_correct += total_batch_corrects
            return pred_idx

        calories_mse = round(self.calories_mse/self.total_ingredients, 3)

    def epoch_finalize(self, learning_rate=0.0, prec=2):

        elapsed_time = round(time.time()-self.start_time, prec)
        epoch_acc = round( 100*self.total_correct / self.total_ingredients, prec)
        calorie_mse = round(self.calorie_mse/self.total, prec)
        if self.is_simple:
            portion_mse = round(self.portion_mse/self.total, prec)
        else:
            portion_mse = round(self.portion_mse/self.total_ingredients, prec)
            calories_mse = round(self.calories_mse/self.total_ingredients, prec)

        if self.run_mode=="train":
            epoch_loss = round( self.epoch_loss / (self.data_size/self.args.batch_size), prec)
            unit_loss = round( self.unit_loss / (self.data_size/self.args.batch_size), prec)
            portion_loss = round( self.portion_loss / (self.data_size/self.args.batch_size), prec)
            calorie_loss = round( self.calorie_loss / (self.data_size/self.args.batch_size), prec)
            portion_distrib_loss = round( self.portion_distrib_loss / (self.data_size/self.args.batch_size), prec)

            output = [self.epoch+1, elapsed_time, learning_rate, epoch_loss, unit_loss, portion_loss, calorie_loss, portion_distrib_loss, calorie_mse, epoch_acc]
            if self.is_simple:
                output.append("")
            else:
                output.append(", calories-mae: {}".format(calories_mse))
            print ("Results @ epoch {} are: time: {}, lr: {}, [losses: {} (unit: {}, portion: {}, calorie: {}, portion-distrib: {})], calorie-mae: {},  accuracy: {}%{}".format(*output))

        else:

            output = [elapsed_time, epoch_acc, portion_mse, calorie_mse]
            if self.is_simple:
                output.append("")
            else:
                output.append(", calories-mae: {}".format(calories_mse))
            print ("Results on test set is time: {}, accuracy: {}%,  portion-mae: {}, calorie-mae: {}{}".format(*output))

# =======================================

class LogResults:

    def __init__(self, dataset, is_simple, run_mode, evaluate_given_images, root="logs"):

        self.id_to_count = collections.defaultdict(int)
        self.evaluate_given_images = evaluate_given_images

        if not os.path.exists(root):
            os.mkdir(root)

        self.state_results_file_path = os.path.join(root, "state_results_{}.json".format(run_mode))
        self.portionunit_results_file_path = os.path.join(root, "portionunit_results_{}.json".format(run_mode))

        self.unit_results_file_path = os.path.join(root, "unit_results_{}.json".format(run_mode))
        self.portion_results_file_path = os.path.join(root, "portion_results_{}.json".format(run_mode))
        self.calorie_ingr_file_path = os.path.join(root, "report_qid_to_calorie_{}.json".format(run_mode))

        self.dataset = dataset

        self.state_softmax = nn.Softmax(dim=2)

        self.calories_results = {}
        self.calorie_results = {}
        self.unit_results = {}
        self.state_results = collections.defaultdict(lambda: collections.defaultdict(list))
        self.portionunit_results = collections.defaultdict(lambda: collections.defaultdict(list))
        self.portion_results = {}
        self.is_simple = is_simple

        self.run_mode = run_mode

        self.ingr_portion_error = collections.defaultdict(float)
        self.ingr_portion_count = collections.defaultdict(float)
        self.ingr_portion_error_file_path = os.path.join(root, "ingr_error_portion_{}.txt".format(run_mode))

        self.ingr_calorie_error = collections.defaultdict(float)
        self.ingr_calorie_count = collections.defaultdict(float)
        self.ingr_calorie_error_file_path = os.path.join(root, "ingr_error_calorie_{}.txt".format(run_mode))

        self.calorie_error = collections.defaultdict(float)
        self.calorie_count = collections.defaultdict(float)
        self.calorie_error_file_path = os.path.join(root, "error_calorie_{}.txt".format(run_mode))

    def update(self, outputs, max_pred_idx, max_tar_idx, tar_values, tar_calorie, tar_calories, max_state_idx, qids, img_id, ques_idx, is_simple, prec=3):

        predicted_classes = []
        predicted_state_classes = []
        gt_classes = []
        gt_state_classes = []

        ques_idx = ques_idx.squeeze().detach().cpu().numpy()
        ques_idx = flatten_array(ques_idx, unchanged_if_flat=True)

        if "unit_output" in outputs:
            max_pred_idx = list(max_pred_idx.squeeze().detach().cpu().numpy())
            max_tar_idx = list(max_tar_idx.squeeze().detach().cpu().numpy())

            if any(isinstance(i, list) for i in qids):
                max_pred_idx = [item for sublist in max_pred_idx for item in sublist]
                max_tar_idx = [item for sublist in max_tar_idx for item in sublist]
                qids = [item for sublist in qids for item in sublist]

            idx = 0
            for item in max_tar_idx:
                # get class of unit for ground-truth item
                classidx = max_tar_idx[idx]
                if classidx!=self.dataset.unit_pad_value:
                    gt_classes.append(classidx)

                    # get class of unit for predicted item
                    predicted_classes.append(max_pred_idx[idx])

                idx += 1

            self.store_predicted_classes(self.unit_results, predicted_classes, gt_classes, ques_idx, qids)

        if "state_output" in outputs:
            max_pred_idx = list(max_pred_idx.squeeze().detach().cpu().numpy())
            max_state_idx = list(max_state_idx.squeeze().detach().cpu().numpy())

            state_outputs, _ = torch.max(self.state_softmax(outputs['state_output']), dim=1)
            state_outputs = list(state_outputs.squeeze().detach().cpu().numpy())

            if any(isinstance(i, list) for i in qids):
                max_pred_idx = [item for sublist in max_pred_idx for item in sublist]
                max_state_idx = [item for sublist in max_state_idx for item in sublist]
                qids = [item for sublist in qids for item in sublist]
                state_probs = [item for sublist in state_outputs for item in sublist]

            idx = 0
            for item in max_state_idx:
                # get class of state for ground-truth item
                classidx = max_state_idx[idx]
                if classidx!=self.dataset.state_pad_value:
                    gt_state_classes.append(classidx)

                    # get class of state for predicted item
                    predicted_state_classes.append(max_pred_idx[idx])

                idx += 1

            self.store_predicted_classes(self.state_results, predicted_state_classes, gt_state_classes, ques_idx, qids, tag="state", probs=state_probs)

        if "calorie_output" in outputs:

            pred_calorie = outputs["calorie_output"].squeeze().detach().cpu().numpy()
            tar_calorie = tar_calorie.squeeze().detach().cpu().numpy()

            if any(isinstance(i, list) for i in qids):
                qids = [item for sublist in qids for item in sublist]

            pred_calorie = flatten_array(pred_calorie)
            tar_calorie = flatten_array(tar_calorie)

            if is_simple:
                self.store_predicted_values(self.calories_results, self.ingr_calorie_error, self.ingr_calorie_count, pred_calorie, tar_calorie, ques_idx, qids, prec=1, tag="calorie")
            else:
                self.store_predicted_values(self.calorie_results, self.calorie_error, self.calorie_count, pred_calorie, tar_calorie, ques_idx, qids, img_id, prec=1, tag="calorie")

        if "portion_output" in outputs:

            pred_portions = outputs["portion_output"].squeeze().detach().cpu().numpy()
            tar_values = tar_values.squeeze().detach().cpu().numpy()

            if any(isinstance(i, list) for i in qids):
                qids = [item for sublist in qids for item in sublist]

            pred_portions = flatten_array(pred_portions)
            tar_values = flatten_array(tar_values)

            self.store_predicted_values(self.portion_results, self.ingr_portion_error, self.ingr_portion_count, pred_portions, tar_values, ques_idx, qids)

        if "calories_output" in outputs:

            calories_output = outputs["calories_output"].squeeze().detach().cpu().numpy()
            tar_calories = tar_calories.squeeze().detach().cpu().numpy()

            if any(isinstance(i, list) for i in qids):
                qids = [item for sublist in qids for item in sublist]

            calories_output = flatten_array(calories_output)
            tar_calories = flatten_array(tar_calories)

            self.store_predicted_values(self.calories_results, self.ingr_calorie_error, self.ingr_calorie_count, calories_output, tar_calories, ques_idx, qids, prec=1, tag="calorie")

    def save_generated_portionunits(self):
        # save portion, unit results for each recipe

        for qid in self.portion_results:
            rid = self.dataset.qid_to_rid[qid]

            self.id_to_count[rid] += 1

            unit = self.unit_results[qid]["unit_prediction"]
            portion = self.portion_results[qid]["portion_prediction"]
            ingr_class = self.portion_results[qid]["ingredient"]

            # check unit and portion simultaneusly to see if it is a valid combination
            unit, portion = check_unit_portions_and_fix(unit, portion, ingr_class, self.dataset)

            if self.evaluate_given_images:
                rid = self.dataset.iid_to_imgnameid[rid]

            self.portionunit_results[rid]['ingrs'].append(ingr_class)
            self.portionunit_results[rid]['units'].append(unit)
            self.portionunit_results[rid]['portions'].append(portion)

    def update_deprecated(self, max_pred_idx, max_tar_idx, outputs, tar_values, qids):

        predicted_classes = []
        gt_classes = []

        max_pred_idx = max_pred_idx.numpy()
        max_tar_idx = max_tar_idx.numpy()
        idx = 0
        for item in qids:

            pred_idx = max_pred_idx[idx]
            tar_idx = max_tar_idx[idx]

            jdx = 0
            for qid in item:

                classidx = pred_idx[jdx]
                predicted_classes.append(classidx)

                classidx = tar_idx[jdx]
                gt_classes.append(classidx)

                jdx += 1

            idx += 1

        qids_flat = [item for sublist in qids for item in sublist]
        self.lookup_and_store(predicted_classes, gt_classes, qids_flat)

    def store_predicted_classes(self, class_results, predicted_classes, gt_classes, ques_idx, qids, tag="unit", probs=[]):

        qid_counter = 0
        for i in range(len(ques_idx)):

            if not self.dataset.is_pad_token(ques_idx[i]):
                ingr_class = self.dataset.idx_to_ingr[ques_idx[i]]

                pidx = predicted_classes[qid_counter]
                if tag=="unit":
                    _class_pred = self.dataset.idx_to_unit[pidx]
                if tag=="state":
                    _class_pred = self.dataset.idx_to_state[pidx]

                gidx = gt_classes[qid_counter]

                if tag=="unit":
                    _class_gt = self.dataset.idx_to_unit[pidx]
                    class_results[qids[qid_counter]] = {"ingredient": ingr_class, "{}_prediction".format(tag): _class_pred, "{}_target".format(tag): _class_gt}

                if tag=="state":
                    _class_gt = self.dataset.idx_to_state[gidx]
                    rid = self.dataset.qid_to_rid[qids[qid_counter]]

                    if self.evaluate_given_images:
                        rid = self.dataset.iid_to_imgnameid[rid]

                    class_results[rid]['ingrs'].append(ingr_class)
                    class_results[rid]['states'].append(_class_pred)
                    class_results[rid]['states_probs'].append(str(probs[qid_counter]))
                    class_results[rid]['states_gt'].append(_class_gt)

                qid_counter += 1

    def store_predicted_values(self, value_results, ingr_error, ingr_count, predicted_values, gt_values, ques_idx, qids, img_id=None, prec=3, tag="portion"):

        qid_counter = 0
        if img_id is None:
            for i in range(len(ques_idx)):

                if not self.dataset.is_pad_token(ques_idx[i]):
                    ingr_class = self.dataset.idx_to_ingr[ques_idx[i]]
                    pvalue = round(predicted_values[i], prec)
                    gvalue = round(gt_values[i], prec)

                    value_results[qids[qid_counter]] = {"ingredient": ingr_class, "{}_prediction".format(tag): pvalue, "{}_target".format(tag): gvalue}
                    qid_counter += 1

                    ingr_error[ingr_class] += abs(round(gt_values[i]-predicted_values[i], prec+5))
                    ingr_count[ingr_class] += 1.0

        else:
            for i in range(len(predicted_values)):

                pvalue = round(predicted_values[i], prec)
                gvalue = round(gt_values[i], prec)

                value_results[img_id[i]] = {"{}_prediction".format(tag): pvalue, "{}_target".format(tag): gvalue}

    def finalize_ingr_errors(self, ingr_error, ingr_count, ingr_error_path, prec=3):

        for ingr in ingr_error.keys():
            ingr_error[ingr] = round(ingr_error[ingr]/ingr_count[ingr], prec)

        ingr_error_sorted = sorted(ingr_error.items(), key=lambda x: x[1], reverse=True)
        ingr_error_file = open(ingr_error_path, 'w')
        for item in ingr_error_sorted:
            ingr_error_file.write("{}: {}\n".format(*item))

    def plot_errors(self, use_dataset=False):

        if use_dataset:
            plot_2d_error(self.unit_results_file_path, self.calorie_ingr_file_path, self.dataset)
        else:
            plot_2d_error(self.unit_results_file_path, self.calorie_ingr_file_path)

    def finalize(self, args, is_simple):

        calorie_dataset_path = args.calorie_dataset_path 
        splits = args.splits
        subsequent_calorie = args.subsequent_calorie

        if len(self.unit_results)>0:
            with open(self.unit_results_file_path, 'w') as outfile:
                json.dump(self.unit_results, outfile)

        if len(self.state_results)>0:
            with open(self.state_results_file_path, 'w') as outfile:
                json.dump(self.state_results, outfile)

        if len(self.unit_results)>0 and len(self.portion_results)>0:
            self.save_generated_portionunits()
            with open(self.portionunit_results_file_path, 'w') as outfile:
                json.dump(self.portionunit_results, outfile)


        self.finalize_ingr_errors(self.ingr_calorie_error, self.ingr_calorie_count, self.ingr_calorie_error_file_path, prec=2)
        self.finalize_ingr_errors(self.ingr_portion_error, self.ingr_portion_count, self.ingr_portion_error_file_path)

        if len(self.portion_results)>0:
            with open(self.portion_results_file_path, 'w') as outfile:
                json.dump({"results": self.portion_results, "errors": self.ingr_portion_error}, outfile)

        recipe_calorie_error = 0
        if subsequent_calorie and ((is_simple and len(self.calorie_results)>0) or (not is_simple and len(self.calories_results)>0)):
            if is_simple:
                with open(self.calorie_ingr_file_path, 'w') as outfile:
                    json.dump({"results": self.calorie_results, "errors": self.ingr_calorie_error}, outfile)
            else:
                with open(self.calorie_ingr_file_path, 'w') as outfile:
                    json.dump({"results": self.calories_results, "errors": self.ingr_calorie_error}, outfile)

            # * computes recipe error by individual calorie estimations of ingredients
            calorie_by_separate_ingredients_calories(calorie_dataset_path, splits, self.calorie_ingr_file_path, args, dataset=self.dataset, units_path=self.unit_results_file_path, portions_results=self.portion_results)

            # ** computes recipe error by individual (unit, portion) estimations of ingredients
            calorie_by_separate_results(splits, self.unit_results_file_path, self.portion_results_file_path, self.calorie_ingr_file_path, dataset=self.dataset, args=args)

        print ("---------------------------------------------------------------")

# =======================================

# Derived from the MCAN paper
def img_feat_path_load(path_list):
    iid_to_path = {}

    for ix, path in enumerate(path_list):
        iid = str(path.split('/')[-1].split('_')[-1].split('.')[0])
        iid_to_path[iid] = path

    return iid_to_path

def img_feat_load(path_list, ids):
    
    ids_set = set(ids)
    bottom_up_feat_size = -1
    iid_to_feat = {}

    for ix, path in enumerate(path_list):
        iid = str(path.split('/')[-1].split('_')[-1].split('.')[0])
        if iid in ids_set:
            img_feat = np.load(path)
            iid_to_feat[iid] = img_feat['x']
            bottom_up_feat_size = iid_to_feat[iid].shape[1]

    return iid_to_feat, bottom_up_feat_size

def proc_img_feat(img_feat, img_feat_pad_size):
    if img_feat.shape[0] > img_feat_pad_size:
        img_feat = img_feat[:img_feat_pad_size]

    img_feat = np.pad(
        img_feat,
        ((0, img_feat_pad_size - img_feat.shape[0]), (0, 0)),
        mode='constant',
        constant_values=0
    )

    return img_feat

# =======================================

def flatten_array(input, unchanged_if_flat=False):

    if any(isinstance(i, list) for i in input) or any(isinstance(i, np.ndarray) for i in input):
        output = [item.item() for sublist in input for item in sublist]
    elif unchanged_if_flat:
        output = input
    else:
        output = [item.item() for item in input]

    return output

# =======================================

def print_arg_info(args):

    print (" ---------- args  info ---------- ")
    print ("Training base with calorie loss: {}".format(args.train_base_wcalorie))
    print ("Embedding size: {}".format(args.embed_size))
    print ("Calorie range: ({}, {})".format(*args.calorie_range))
    print ("Image feature type: {}".format(args.image_encoder_type))
    print ("Attention attributes:\n   unit-att-layers: {}\n   unit-att-heads: {}".format(args.transf_layers_units, args.n_att_units))
    print ("Calorie has a direct stream from CNN: {}".format(args.image_to_calorie_stream))
    print ("Model type: {}".format(args.model_type))
    if args.warmup:
        print ("Warm-up is used for {} epochs.".format(args.warmup_epochs))
    else:
        print ("Warm-up is not used.")
    print ("Learning rate starts with {} if warm-up is not used.".format(args.learning_rate))
    print ("Number of max epochs: {}".format(args.num_epochs))
    print ("Loss lambdas are:\n   unit lambda: {}\n   portion lambda: {}\n   calorie lambda: {}\n   calories lambda: {}".format(*args.loss_lambda))
    print (" ---------- ---------- ---------- \n")

# =======================================

def _reset_parameters(parameters):
    """Initiate parameters in the transformer model."""

    for p in parameters:
        if p.dim() > 1:
            xavier_uniform_(p)

# =======================================

def count_parameters(model):
    
    non_cnn_model_params = 0
    for name, param in model.named_parameters():
        if not "image" in name:
            non_cnn_model_params += param.numel()

    return sum(p.numel() for p in model.parameters() if p.requires_grad), non_cnn_model_params

# =======================================

def save_model(model, optimizer, args, checkpoints_dir, suff=''):

    if torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(), os.path.join(
            checkpoints_dir, 'model' + suff + '.ckpt'))

    else:
        torch.save(model.state_dict(), os.path.join(
            checkpoints_dir, 'model' + suff + '.ckpt'))

    torch.save(optimizer.state_dict(), os.path.join(
        checkpoints_dir, 'optim' + suff + '.ckpt'))

    pickle.dump(args, open(os.path.join(checkpoints_dir, 'args.pkl'), 'wb'))

# =======================================

class Optim:

    def __init__(self, args, model, data_size, is_simple=False, decay_factor=1.0, b1=0.9, b2=0.98, eps=1e-9):
        
        self.is_simple = is_simple
        self.learning_rate = args.learning_rate
        self.lr_decay_rate = args.lr_decay_rate
        self.lr_decay_every = args.lr_decay_every
        self.decay_lr = args.decay_lr
        self.decay_factor = decay_factor
        self.warm_up = args.warmup

        self.step_skip = 50
        self.epoch_steps = (data_size/args.batch_size)
        self.warmup_steps = self.step_skip * args.warmup_epochs * self.epoch_steps
        self.d_model = args.embed_size
        self.scale_learning_rate_cnn = args.scale_learning_rate_cnn

        if args.dataset=="menumatch":
            self.decay_factor = 0.99
        elif is_simple:
            self.decay_factor = 0.8
        

        params = list(model.parameters())
        cnn_params, resnet_training_params = None, None
        if args.image_encoder_type == "resnet" and args.keep_cnn_gradients:
            cnn_params = list(model.image_encoder.resnet.parameters())

            cnn_trainable_parameter_names = {"7.2.conv1.weight", "7.2.bn1.weight", "7.2.bn1.bias", 
                                             "7.2.conv2.weight", "7.2.bn2.weight", "7.2.bn2.bias",
                                             "7.2.conv3.weight", "7.2.bn3.weight", "7.2.bn3.bias" }
            
            resnet_training_params = []
            for name, param in model.image_encoder.resnet.named_parameters():
                if name in cnn_trainable_parameter_names:
                    resnet_training_params.append(param)

        self.create_optimizer(params, b1, b2, eps, cnn_params, resnet_training_params)

    def warm_up_formula(self, cur_step, epoch):

        steps = (1 + cur_step + epoch*self.epoch_steps) * self.step_skip
        a = steps**(-0.5)
        b = steps*self.warmup_steps**(-1.5)
        self.learning_rate = ((self.d_model)**(-0.5))*min(a, b)
        self.set_lr(self.learning_rate)

    def create_optimizer(self, params, b1=0.9, b2=0.98, eps=1e-9, cnn_params=None, resnet_training_params=None):
        '''if params_cnn is not None and args.finetune_after == 0:
            # start with learning rates as they were (if decayed during training)
            optimizer = torch.optim.Adam([{'params': params},
                                        {'params': params_cnn, 'lr': decay_factor*args.learning_rate*args.scale_learning_rate_cnn}],
                                            lr=decay_factor*args.learning_rate)

            keep_cnn_gradients = True
        else:'''

        if not cnn_params:
            self.optimizer = torch.optim.Adam(params, betas=(b1, b2), eps=eps, lr=self.learning_rate)
        else:
            self.optimizer = torch.optim.Adam([{'params': list(set(params)-set(cnn_params))}, 
                                               {'params': resnet_training_params, 'lr': self.learning_rate*self.scale_learning_rate_cnn}], 
                                              betas=(b1, b2), eps=eps, lr=self.learning_rate)

        return self.optimizer

    def set_lr(self, new_lr):
        for group in self.optimizer.param_groups:
            group['lr'] = new_lr

    def update_optimizer_step(self, cur_step, epoch):

        if self.warm_up:
            self.warm_up_formula(cur_step, epoch)

    def update_optimizer_epoch(self, epoch):

        if self.is_simple:
            new_lr = self.learning_rate * self.decay_factor
            self.set_lr(new_lr)
            self.learning_rate = new_lr
        elif self.decay_lr and not self.warm_up:
            frac = epoch // self.lr_decay_every
            self.decay_factor = self.lr_decay_rate ** frac
            new_lr = self.learning_rate*self.decay_factor
            self.set_lr(new_lr)
            self.learning_rate = new_lr

    def step(self):
        self.optimizer.step()

# =======================================

def maintain_cnn_features(imgid, outputs, imgid_to_features, stored_imgids):
    # prepare current batch features in dict to save to disk later
    #   a) put the recipe features here to save it to disk (in the train function) for the KL-divergence training later or
    #   b) save kd-trained features for later analysis
    _features = outputs['cnn_features'].detach().cpu().numpy()
    for iidx in range(_features.shape[0]):
        if not imgid[iidx] in stored_imgids:
            imgid_to_features[imgid[iidx]] = _features[iidx, :]
            print (imgid[iidx])
            stored_imgids.add(imgid[iidx])

# =======================================

def store_features(path, run_mode, imgid_to_features, store_batch_num=1, last_batch=False, file_size_limit=48000):
    
    if len(imgid_to_features)>file_size_limit or last_batch:
        # save features in dict to disk
        features_file_path = os.path.join(path, '{}_features{}.pkl'.format(run_mode, store_batch_num))
        pickle.dump(imgid_to_features, open(features_file_path, 'wb'))
        store_batch_num = store_batch_num+1

    return store_batch_num

# =======================================

def create_ingr_portion_mask(ingr_vocab_size, unit_vocab_size, ques_idx, batch_size, seq_size):

    a = torch.tensor(np.mgrid[0:ingr_vocab_size,0:unit_vocab_size][0].flatten()).to(device)
    a = a.repeat(ques_idx.size(0), seq_size, 1)
    b = ques_idx.unsqueeze(2).repeat(1,1,ingr_vocab_size*unit_vocab_size)

    return torch.eq(a,b).float()

# =======================================

def check_unit_portions_and_fix(unit, portion, ingredient, dataset):

    min_none, max_none = 0.5, 100
    min_cup, max_cup = 0.25, 8
    min_teaspoon, max_teaspoon = 0.25, 16
    min_tablespoon, max_tablespoon = 0.25, 16
    min_pound, max_pound = 0.25, 6
    min_ounce, max_ounce = 0.5, 16

    # check for each ingredient if the portion is in the valid range
    portion = max(portion, dataset.ingrunit_to_minportion[ingredient][unit])
    portion = min(portion, dataset.ingrunit_to_maxportion[ingredient][unit])

    if unit in {"None", "none"}:
        portion = min(max_none, portion)
        portion = max(min_none, portion)

    if unit=="teaspoon":
        portion = min(max_teaspoon, portion)
        portion = max(min_teaspoon, portion)

    if unit=="tablespoon":
        portion = min(max_tablespoon, portion)
        portion = max(min_tablespoon, portion)

    if unit=="pound":
        portion = min(max_pound, portion)
        portion = max(min_pound, portion)

    if unit=="ounce":
        portion = min(max_ounce, portion)
        portion = max(min_ounce, portion)

    if unit=="cup":
        portion = min(max_cup, portion)
        portion = max(min_cup, portion)

    # round portions to the nearest .25
    portion = round(portion*4)/4

    return unit, portion
