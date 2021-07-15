# Copyright (c) Robotic and Action Perception Lab (RPAL) at University of South Florida
# by Ahmad Babaeian Jelodar

'''
    This file stitches the image model (Resnet) with the ingredient decoder to create
    the full model. The actual model used for ingredient decoder is defined based on
    user input and from the decoder.py or decoder_simple.py files.
'''

import torch
import torch.nn as nn
import random
import numpy as np
from modules.encoder import EncoderCNN, EncoderIngredient, EncoderCNNFeatures
from modules.decoder import create_decoder, run_decoder
from modules.decoder_simple import create_simple_decoder, run_simple_decoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ====================================

class KDLoss(nn.Module):
    '''
        KL divergence loss for the outputs of portion distribution between an instance & its same class counterpart.
        (borrowed from https://github.com/alinlab/cs-kd)
    '''
    def __init__(self, temp_factor):
        super(KDLoss, self).__init__()
        self.temp_factor = temp_factor
        self.kl_div = nn.KLDivLoss(reduction="sum")

    def forward(self, input, target):
        log_p = torch.log_softmax(input/self.temp_factor, dim=1)
        q = torch.softmax(target/self.temp_factor, dim=1)
        loss = self.kl_div(log_p, q)*(self.temp_factor**2)/input.size(0)
        return loss

# ====================================

def label2onehot(labels, pad_value, input_mat=None):
    '''
        convert input label to one-hot vector
    '''
    # input labels to one hot vector
    inp_ = torch.unsqueeze(labels, 1)

    one_hot = torch.FloatTensor(labels.size(0), pad_value + 1).zero_().to(device)
    if input_mat is None:
        one_hot.scatter_(1, inp_, 1)
    else:
        one_hot.scatter_(1, inp_, torch.unsqueeze(input_mat, 1))

    # remove pad position
    one_hot = one_hot[:, :-1]

    return one_hot

# ====================================

def get_model(args, ingr_vocab_size, unit_vocab_size, states_vocab_size, calorie_classes_num, dish_vocab_size, bottom_up_feat_size, pretrained_emb, loss_weights_matrix, flattened_weights=None):
    '''
        create model for training
    '''
    # simple model or not
    is_simple = ("simple" in args.model_type)

    if args.just_input_ingr:
        # build ingredients embedding
        encoder_ingrs = EncoderIngredient(args.embed_size*2, ingr_vocab_size, args.dropout_encoder, simple=is_simple, scale_grad=False, pretrained_emb=pretrained_emb, use_bert=args.use_bert).to(device)
    else:
        encoder_ingrs = EncoderIngredient(args.embed_size, ingr_vocab_size, args.dropout_encoder, simple=is_simple, scale_grad=False, pretrained_emb=pretrained_emb, use_bert=args.use_bert).to(device)

    # build dish embedding
    if args.model_type in {"jointwithcaloriedoubledish", "simple_unit_dish", "simple_portion_dish", "simple_dishtocalorie_dish", "simple_ingrtocalorie_dish"}:
        encoder_dishes = EncoderIngredient(args.embed_size, dish_vocab_size, args.dropout_encoder, simple=is_simple, scale_grad=False).to(device)
    else:
        encoder_dishes = None

    # build unit embedding for input to simple portion model
    if args.model_type in {"simple_portion", "simple_portion_dish"}:
        encoder_unit = EncoderIngredient(args.embed_size, unit_vocab_size, args.dropout_encoder, simple=is_simple, scale_grad=False).to(device)
    else:
        encoder_unit = None

    if args.just_input_image:
        img_embed_size = args.embed_size*2
    else:
        img_embed_size = args.embed_size

    # build image model
    if args.image_encoder_type=="resnet_features":
        encoder_image = EncoderCNNFeatures(img_embed_size, args.dropout_encoder, args.image_model, simple=is_simple)
    elif args.image_encoder_type=="resnet":
        encoder_image = EncoderCNN(img_embed_size, args.dropout_encoder, args.image_model, simple=is_simple)
    elif args.image_encoder_type=="bottom-up":
        if img_embed_size!=bottom_up_feat_size:
            # linear transformation
            encoder_image = nn.Linear(bottom_up_feat_size, img_embed_size)
        else:
            # identity architecture
            encoder_image = nn.Sequential()

    # output loss
    label_loss = nn.CrossEntropyLoss(reduce=False, weight=loss_weights_matrix)
    state_label_loss = nn.CrossEntropyLoss(reduce=False)
    portion_loss = nn.MSELoss(reduce=False, reduction='none')
    calorie_loss = nn.MSELoss()
    calories_crit = nn.MSELoss(reduce=False, reduction='none')
    portion_distrib_crit = nn.MSELoss(reduce=False, reduction='none')
    calorie_class_crit = nn.CrossEntropyLoss(reduce=False)

    distrib_kl_loss = None
    # kl-divergence distribution loss
    if args.use_distrib_loss:
        distrib_kl_loss = KDLoss(args.temp)

    if is_simple:
        # build simlpe decoder
        decoder_att = create_simple_decoder(args, ingr_vocab_size, unit_vocab_size, calorie_classes_num)
    else:
        # build image & ingredient decoder
        decoder_att = create_decoder(args, unit_vocab_size, ingr_vocab_size, states_vocab_size, flattened_weights)

    model = MultiPurposeModel(encoder_ingrs, encoder_image, decoder_att, unit_vocab_size, args.model_type, args.embed_size, encoder_dishes, encoder_unit,
                        label_loss, state_label_loss, portion_loss, calorie_loss, calories_crit, portion_distrib_crit, distrib_kl_loss, calorie_class_crit, is_simple=is_simple, image_encoder_type=args.image_encoder_type,
                        just_input_image=args.just_input_image, just_input_ingr=args.just_input_ingr, cnn_reduced=args.cnn_reduced)

    return model

# ====================================

class MultiPurposeModel(nn.Module):
    '''
        Class to create the model for training and 
            1. perform the feed-forward & 
            2. loss computation over the model
    '''
    def __init__(self, ingredient_encoder, image_encoder, decoder_att, target_size, model_type, embed_size, dish_encoder=None, unit_encoder=None,
                 crit=None, state_crit=None, por_crit=None, cal_crit=None, cals_crit=None, portion_distrib_crit=None, distrib_kl_loss=None,
                 calorie_class_crit=None, pad_value=0, label_smoothing=0.0, is_simple=False, just_input_image=False, just_input_ingr=False, 
                 cnn_reduced=False, image_encoder_type="bottom-up"):
        '''
            initialize the model and loss parameters
        '''
        super(MultiPurposeModel, self).__init__()

        self.embed_size = embed_size
        self.ingredient_encoder = ingredient_encoder
        self.image_encoder = image_encoder
        self.decoder = decoder_att

        self.just_input_image = just_input_image
        self.just_input_ingr = just_input_ingr

        self.dish_encoder = dish_encoder
        self.unit_encoder = unit_encoder

        self.crit = crit
        self.state_crit = state_crit
        self.por_crit = por_crit
        self.cal_crit = cal_crit
        self.calories_crit = cals_crit
        self.portion_distrib_crit = portion_distrib_crit
        self.distrib_kl_loss = distrib_kl_loss
        self.calorie_class_crit = calorie_class_crit

        self.label_smoothing = label_smoothing
        self.target_size = target_size

        self.model_type = model_type
        self.image_encoder_type = image_encoder_type

        self.simple = is_simple

        self.cnn_reduced = cnn_reduced

    def forward_features(self, outputs, ingr_idx, tar_idx, dish_idxs, img_inputs, keep_cnn_gradients=False):

        if self.image_encoder_type=="resnet_features":
            img_features = self.image_encoder(img_inputs)
        elif self.image_encoder_type=="resnet":
            img_features, outputs['cnn_features'] = self.image_encoder(img_inputs, keep_cnn_gradients)
        elif self.image_encoder_type=="bottom-up":
            img_features, outputs['cnn_features'] = self.image_encoder(img_inputs)
            img_features = img_features.permute(1, 0, 2).contiguous()

        ingr_features = self.ingredient_encoder(ingr_idx)

        if self.just_input_image:
            img_features, ingr_features = torch.split(img_features, self.embed_size, 1)
        elif self.just_input_ingr:
            img_features, ingr_features = torch.split(ingr_features, self.embed_size, 1)
        #else: # arg.input_both:

        dish_features = None
        if self.dish_encoder:
            dish_features = self.dish_encoder(dish_idxs.unsqueeze(1)).squeeze()

        unit_features = None
        if self.unit_encoder:
            unit_features = self.unit_encoder(tar_idx.unsqueeze(1)).squeeze()
        
        return img_features, ingr_features, dish_features, unit_features

    def forward(self, ingr_idx, ingr_mask, ingr_portion_mask, tar_idx, tar_values, tar_calorie, tar_calories, tar_states, mean_prior_calories_gt, portion_distrib, dish_idxs, img_inputs, image_mask=None, ingr_calorie_vec_mask=None, ingr_calorie_vec=None, calorie_class_idx=None, keep_cnn_gradients=False, paired_ques_idx=None, paired_tar_idx=None, paired_dish_idxs=None, paired_image_input=None):
        '''
            feed-forward through the model & compute loss
        '''
        losses = {}
        outputs = {}

        img_features, ingr_features, dish_features, unit_features = self.forward_features(outputs, ingr_idx, dish_idxs, tar_idx, img_inputs, keep_cnn_gradients)

        if 'simple' in self.model_type:
            run_simple_decoder(self.decoder, outputs, ingr_features, img_features, unit_features, dish_features, ingr_calorie_vec_mask, self.model_type)
        else:
            run_decoder(self.decoder, outputs, ingr_features, img_features, dish_features, self.model_type, ingr_mask, image_mask, ingr_portion_mask, self.cnn_reduced)

        # unit loss
        if "unit_output" in outputs.keys():
            unit_output = outputs["unit_output"]

            if not self.simple:
                tar_idx = (tar_idx * (~ingr_mask).long())
            unit_loss = self.crit(unit_output, tar_idx)
            if not self.simple:
                unit_loss = (unit_loss * (~ingr_mask).float())

            losses["unit_loss"] = unit_loss.sum(1)

        # state loss
        if "state_output" in outputs.keys():
            state_output = outputs["state_output"]

            if not self.simple:
                tar_states = (tar_states * (~ingr_mask).long())

            state_loss = self.state_crit(state_output, tar_states)
            if not self.simple:
                state_loss = (state_loss * (~ingr_mask).float())

            losses["state_loss"] = state_loss.sum(1)

        # calorie class loss
        if "calorie_class_output" in outputs.keys():
            calorie_class_output = outputs["calorie_class_output"]
            if not self.simple:
                calorie_class_idx = (calorie_class_idx * (~ingr_mask).long())
            calorie_class_loss = self.calorie_class_crit(calorie_class_output, calorie_class_idx)
            if not self.simple:
                calorie_class_loss = (calorie_class_loss * (~ingr_mask).float()).sum(1)
            losses["calorie_class_loss"] = calorie_class_loss

        # portion loss
        if "portion_output" in outputs.keys():
            outputs["portion_output"] = outputs["portion_output"].squeeze()
            portion_output = outputs["portion_output"]
            tar_values = torch.reshape(tar_values, portion_output.size())
            if not self.simple:
                tar_values = (tar_values * (~ingr_mask).float())
            
            portion_loss = self.por_crit(portion_output, tar_values)
            if not self.simple:
                portion_loss = (portion_loss * (~ingr_mask).float())
            losses["portion_loss"] = portion_loss.sum(1)

        # calorie loss (ingredient calorie loss when in simple mode or recipe calorie loss when in recipe mode)
        if "calorie_output" in outputs.keys():
            calorie_output = outputs["calorie_output"]
            #print ('calorie loss size: ', calorie_output.size(), tar_calorie.size())
            tar_calorie = torch.reshape(tar_calorie, calorie_output.size())
            calorie_loss = self.cal_crit(calorie_output, tar_calorie)
            losses["calorie_loss"] = calorie_loss

        if "calorie_alignment_output" in outputs.keys():
            calorie_alignment_output = outputs["calorie_alignment_output"]
            tar_calorie = torch.reshape(tar_calorie, calorie_alignment_output.size())
            calorie_loss = self.cal_crit(calorie_alignment_output, tar_calorie)
            losses["calorie_alignment_loss"] = calorie_loss

        if "calorie_alignment2_output" in outputs.keys():
            calorie_alignment2_output = outputs["calorie_alignment2_output"]
            tar_calorie = torch.reshape(tar_calorie, calorie_alignment2_output.size())
            calorie_loss = self.cal_crit(calorie_alignment2_output, tar_calorie)
            losses["calorie_alignment2_loss"] = calorie_loss

        # recipe calorie loss when in simple mode
        if "recipe_calorie_output" in outputs.keys():
            recipe_calorie_output = outputs["recipe_calorie_output"]
            tar_calories = torch.reshape(tar_calories, recipe_calorie_output.size())
            recipe_calorie_loss = self.cal_crit(recipe_calorie_output, tar_calories)
            losses["recipe_calorie_loss"] = recipe_calorie_loss

        # mean prior calorie loss computatuion (prior mean calorie of an ingredient)
        if "prior_calorie_output" in outputs.keys():
            prior_calorie_output = outputs["prior_calorie_output"]
            mean_prior_calories_gt = torch.reshape(mean_prior_calories_gt, prior_calorie_output.size())
            prior_calorie_loss = self.cal_crit(prior_calorie_output, mean_prior_calories_gt)
            losses["prior_calorie_loss"] = prior_calorie_loss

        # calories loss (separate calories for ingredients)
        if "calories_output" in outputs.keys():
            outputs["calories_output"] = outputs["calories_output"].squeeze()
            calories_output = outputs["calories_output"]

            tar_calories = torch.reshape(tar_calories, calories_output.size())

            if not self.simple:
                tar_calories = (tar_calories * (~ingr_mask).float())

            calories_loss = self.calories_crit(calories_output, tar_calories)

            if not self.simple:
                calories_loss = (calories_loss * (~ingr_mask).float())

            losses["calories_loss"] = calories_loss.sum(1)

        # calories loss (separate calories for ingredients)
        if "ingr_portion_output" in outputs.keys():
            outputs["ingr_portion_output"] = outputs["ingr_portion_output"].squeeze()
            ingr_portion_output = outputs["ingr_portion_output"]

            tar_calories = torch.reshape(tar_calories, ingr_portion_output.size())

            if not self.simple:
                tar_calories = (tar_calories * (~ingr_mask).float())

            ingr_portion_loss = self.calories_crit(ingr_portion_output, tar_calories)

            if not self.simple:
                ingr_portion_loss = (ingr_portion_loss * (~ingr_mask).float())

            losses["ingr_portion_loss"] = ingr_portion_loss.sum(1)

        if "calories_alignment_output" in outputs.keys():

            outputs["calories_alignment_output"] = outputs["calories_alignment_output"].squeeze()
            alignment_output = outputs["calories_alignment_output"]

            tar_calories = torch.reshape(tar_calories, alignment_output.size())

            if not self.simple:
                tar_calories = (tar_calories * (~ingr_mask).float())

            alignment_loss = self.calories_crit(alignment_output, tar_calories)

            if not self.simple:
                alignment_loss = (alignment_loss * (~ingr_mask).float())

            losses["calories_alignment_loss"] = alignment_loss.sum(1)

        # ingredients portion distribution loss
        if "portion_distrib" in outputs.keys():
            outputs["portion_distrib"] = outputs["portion_distrib"].squeeze()
            portion_distrib_out = outputs["portion_distrib"]

            portion_distrib = torch.reshape(portion_distrib, portion_distrib_out.size())
            
            if not self.simple:
                portion_distrib = (portion_distrib * (~ingr_mask).float())

            portion_distrib_loss = self.portion_distrib_crit(portion_distrib_out, portion_distrib)

            if not self.simple:
                portion_distrib_loss = (portion_distrib_loss * (~ingr_mask).float()).sum(1)

            losses["portion_distrib_loss"] = portion_distrib_loss

        if self.distrib_kl_loss and self.simple:

            paired_outputs = {}
            with torch.no_grad():
                paired_img_features, paired_ingr_features, paired_dish_features, paired_unit_features = self.forward_features(paired_outputs, paired_ques_idx, paired_tar_idx, paired_dish_idxs, paired_image_input)
                run_simple_decoder(self.decoder, paired_outputs, paired_ingr_features, paired_img_features, paired_unit_features, paired_dish_features, ingr_calorie_vec_mask, self.model_type)

            losses["distrib_kl_loss"] = self.distrib_kl_loss(outputs["portion_distrib"], paired_outputs["portion_distrib"])

        return losses, outputs
