# Copyright (c) Robotic and Action Perception Lab (RPAL) at University of South Florida
# by Ahmad Babaeian Jelodar

import torch
import torch.nn as nn
import random
import numpy as np
from enum import Enum

# =======================================

ACTIVATON_LAYERS = {"relu": nn.ReLU(), 
                    "sigmoid": nn.Sigmoid(),
                    "tanh": nn.Tanh(), 
                    "linear": nn.Sequential()}

# =======================================

class SimUnit(nn.Module):
    '''
        Decoder for just predicting a unit given a single ingredient & image
    '''
    def __init__(self, embed_size, target_size, using_dish_features, nlayers=3, pre_last_layer_size=128, dropout=0.1, activation="relu"):

        super(SimUnit, self).__init__()
        self.using_dish_features = using_dish_features

        # linear fully-connected layer for unit estimation (a class)
        if using_dish_features:
            input_size = 3*embed_size
        else:
            input_size = 2*embed_size       

        self.nlayers = nlayers
        if nlayers==3:
            self.base = nn.Sequential(
                    nn.Linear(input_size, 2*pre_last_layer_size),
                    nn.BatchNorm1d(2*pre_last_layer_size),
                    ACTIVATON_LAYERS[activation],
                    nn.Linear(2*pre_last_layer_size, pre_last_layer_size),
                    nn.BatchNorm1d(pre_last_layer_size),
                    ACTIVATON_LAYERS[activation]
                )
        elif nlayers==2:
            self.base = nn.Sequential(
                    nn.Linear(input_size, pre_last_layer_size),
                    nn.BatchNorm1d(pre_last_layer_size),
                    ACTIVATON_LAYERS[activation]
                )
        else: # nlayers==1
            pre_last_layer_size = input_size

        self.linear = nn.Linear(pre_last_layer_size, target_size)
        self.activation = ACTIVATON_LAYERS[activation]

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Sequential()

    def forward(self, ingr_embedding, img_embedding, dish_features=None):

        if self.using_dish_features:
            joint_embedding = torch.cat((img_embedding, ingr_embedding, dish_features), dim=1)
        else:
            joint_embedding = torch.cat((img_embedding, ingr_embedding), dim=1)

        if self.nlayers>=2:
            pre_last = self.base(joint_embedding)
            joint_embedding = self.dropout(pre_last)
        else:
            pre_last = joint_embedding

        out = self.activation(self.linear(joint_embedding))

        return out, pre_last

# =======================================

class SimPortion(nn.Module):
    '''
        Decoder for just predicting portion given a single ingredient, unit & image
    '''
    def __init__(self, embed_size, using_dish_features, using_unit_embedding=True, nlayers=3, pre_last_layer_size=128, dropout=0.1, activation="relu"):

        super(SimPortion, self).__init__()
        
        # linear fully-connected layer for portion estimation
        self.using_dish_features = using_dish_features
        self.using_unit_embedding = using_unit_embedding

        # linear fully-connected layer for unit estimation (a class)
        if using_dish_features and using_unit_embedding:
            input_size = 4*embed_size
        elif using_dish_features or using_unit_embedding:
            input_size = 3*embed_size
        else:
            input_size = 2*embed_size

        self.nlayers = nlayers
        if nlayers==3:
            self.base = nn.Sequential(
                    nn.Linear(input_size, 2*pre_last_layer_size),
                    nn.BatchNorm1d(2*pre_last_layer_size),
                    ACTIVATON_LAYERS[activation],
                    nn.Linear(2*pre_last_layer_size, pre_last_layer_size),
                    nn.BatchNorm1d(pre_last_layer_size),
                    ACTIVATON_LAYERS[activation]
                )
        elif nlayers==2:
            self.base = nn.Sequential(
                    nn.Linear(input_size, embed_size),
                    nn.BatchNorm1d(embed_size),
                    ACTIVATON_LAYERS[activation],
                    nn.Linear(embed_size, pre_last_layer_size),
                    nn.BatchNorm1d(pre_last_layer_size),
                    ACTIVATON_LAYERS[activation]
                )
        else:
            pre_last_layer_size = input_size

        self.last_linear = nn.Linear(pre_last_layer_size, 1)
        self.activation = ACTIVATON_LAYERS[activation]

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Sequential()

    def forward(self, ingr_embedding, img_embedding=None, unit_embedding=None, dish_features=None):

        if self.using_dish_features and self.using_unit_embedding:
            joint_embedding = torch.cat((img_embedding, ingr_embedding, unit_embedding, dish_features), dim=1)
        elif self.using_dish_features:
            joint_embedding = torch.cat((img_embedding, ingr_embedding, dish_features), dim=1)
        elif self.using_unit_embedding:
            joint_embedding = torch.cat((img_embedding, ingr_embedding, unit_embedding), dim=1)
        else:
            joint_embedding = torch.cat((img_embedding, ingr_embedding), dim=1)
        
        if self.nlayers>=2:
            pre_last = self.base(joint_embedding)
            joint_embedding = self.dropout(pre_last)
        else:
            pre_last = joint_embedding
            
        out = self.activation(self.last_linear(joint_embedding))
        
        return out, pre_last

# =======================================

class SimCalorie(nn.Module):
    '''
        Decoder for just predicting calorie given 
        a single image and (dish name) or    --> predicts total recipe calorie
    '''
    def __init__(self, embed_size, using_dish_features, pre_last_layer_size=128, dropout=0.1, activation='relu'):

        super(SimCalorie, self).__init__()

        # linear fully-connected layer for portion estimation
        self.using_dish_features = using_dish_features
        if using_dish_features:
            input_size = 2*embed_size
        else:
            input_size = embed_size

        self.base = nn.Sequential(
                nn.Linear(input_size, pre_last_layer_size),
                nn.BatchNorm1d(pre_last_layer_size),
                ACTIVATON_LAYERS[activation]
            )

        self.last_linear = nn.Linear(pre_last_layer_size, 1)
        self.activation = ACTIVATON_LAYERS[activation]

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Sequential()

    def forward(self, img_embedding, extra_features=None):

        if self.using_dish_features:
            joint_embedding = torch.cat((img_embedding, extra_features), dim=1)
        else:
            joint_embedding = img_embedding
        
        pre_last = self.base(joint_embedding)
        pre_last = self.dropout(pre_last)
        out = self.activation(self.last_linear(pre_last))

        return out, pre_last

# =======================================

class SimIndivCalorie(nn.Module):
    '''
        Decoder for just predicting calorie given 
        a single image and (ingredient name) --> predicts ingredient calorie for that recipe
    '''
    def __init__(self, embed_size, using_dish_features, pre_last_layer_size=128, nlayers=1, dropout=0.1):

        super(SimIndivCalorie, self).__init__()
        
        # linear fully-connected layer for portion estimation
        self.using_dish_features = using_dish_features
        if using_dish_features:
            input_size = 3*embed_size
        else:
            input_size = 2*embed_size

        self.nlayers = nlayers
        if nlayers==2:
            self.base = nn.Sequential(
                    nn.Linear(input_size, pre_last_layer_size),
                    nn.BatchNorm1d(pre_last_layer_size),
                    nn.ReLU()
                )
        else:
            pre_last_layer_size = input_size

        self.last_linear = nn.Linear(pre_last_layer_size, 1)
        self.relu = nn.ReLU()

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Sequential()

    def forward(self, ingr_embedding, img_embedding, extra_features=None):

        if self.using_dish_features:
            joint_embedding = torch.cat((img_embedding, ingr_embedding, extra_features), dim=1)
        else:
            joint_embedding = torch.cat((img_embedding, ingr_embedding), dim=1)
        
        if self.nlayers==2:
            pre_last = self.base(joint_embedding)
            joint_embedding = self.dropout(pre_last)

        out = self.relu(self.last_linear(joint_embedding))

        return out, joint_embedding

# =======================================

class HeirarchicalModel(nn.Module):
    '''
        Decoder for just predicting calorie given 
        a single image and (ingredient name) --> predicts ingredient calorie or ingredient portion
    '''
    def __init__(self, embed_size, target_size, model_type, using_dish_features, using_unit_embedding=False, pre_last_layer_size=128, portion_uses_ingr_only=False, dropout=0.1, do_calorie_scale=False, do_total_calorie=False, do_prior_calorie=False, activation='relu'):

        super(HeirarchicalModel, self).__init__()

        # linear fully-connected layer for portion estimation
        self.using_unit_embedding = using_unit_embedding
        self.using_dish_features = using_dish_features

        # returns N values for each N units
        self.unit_stream = SimUnit(embed_size, target_size, using_dish_features, pre_last_layer_size=pre_last_layer_size, activation=activation)
        
        # returns 1 value for the portion
        self.port_stream = SimPortion(embed_size, using_dish_features, using_unit_embedding=using_unit_embedding, pre_last_layer_size=pre_last_layer_size, activation=activation)

        # returns N values associated with the N units
        self.coef_stream = SimUnit(embed_size, target_size, using_dish_features, pre_last_layer_size=pre_last_layer_size, activation=activation)
       
        self.portion_uses_ingr_only = portion_uses_ingr_only

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Sequential()

        self.do_calorie_scale = do_calorie_scale
        if do_calorie_scale:
            do_total_calorie = True
            # mapping unit probs to a coef for what is the scale of calorie
            self.sigmoid = nn.Sigmoid()
            self.calorie_scale = nn.Linear(embed_size, 1)

        self.do_total_calorie = do_total_calorie
        if do_total_calorie:
            # image to recipe calorie
            self.img_to_calorie = nn.Linear(embed_size, 1)

        self.do_prior_calorie = do_prior_calorie
        if do_prior_calorie:
            # ingredient embedding to prior mean calorie for that ingredients
            self.ingr_to_prior_calorie = nn.Linear(embed_size, 1)

        self.concatenated = False
        count = 1 + int(do_total_calorie) + int(do_prior_calorie)
        if count>1:
            # given estimated calories in various ways predict simply the ingredient calorie
            self.calories_to_calorie = nn.Linear(count, 1)
            self.concatenated = True

    def forward(self, ingr_embedding, img_embedding, unit_embedding, extra_features=None):

        unit_output, _ = self.unit_stream(ingr_embedding, img_embedding, extra_features)

        if self.portion_uses_ingr_only:
            port_output, _ = self.port_stream(ingr_embedding, ingr_embedding, unit_embedding, extra_features)
        else:
            port_output, _ = self.port_stream(ingr_embedding, img_embedding, unit_embedding, extra_features)

        coef_output, _ = self.coef_stream(ingr_embedding, img_embedding, extra_features)

        portion_coef = torch.matmul(port_output.detach().unsqueeze(2), coef_output.unsqueeze(1))

        output_value = torch.matmul(portion_coef, unit_output.detach().unsqueeze(2))
        
        calories_vector_out = output_value.squeeze().unsqueeze(1)

        recipe_calorie_out = None
        if self.do_total_calorie:
            recipe_calorie_out = self.img_to_calorie(img_embedding)
            if not self.do_calorie_scale:
                calories_vector_out = torch.cat((calories_vector_out, recipe_calorie_out), dim=1)

        prior_calorie_out = None
        if self.do_prior_calorie:
            prior_calorie_out = self.ingr_to_prior_calorie(ingr_embedding)
            calories_vector_out = torch.cat((calories_vector_out, prior_calorie_out.detach()), dim=1)

        if self.do_calorie_scale:
            calorie_scale = self.sigmoid(self.calorie_scale(ingr_embedding.detach()))
            simple_calorie = recipe_calorie_out.detach() * calorie_scale
            calories_vector_out = torch.cat((calories_vector_out, simple_calorie), dim=1)
        
        if self.concatenated:
            output_value = self.calories_to_calorie(calories_vector_out)

        return output_value.squeeze().unsqueeze(1), unit_output, port_output, recipe_calorie_out, prior_calorie_out


# =======================================

class MultiLevelCalorie(nn.Module):
    '''
        Decoder for predicting calorie given 
        a single image and (ingredient name) --> predicts ingredient calorie
        - outputs a vector of probabilities for units (N=6)
        - outputs a vector of portion values for each unit (N=6)
        - outputs a single value for portion
    '''
    def __init__(self, embed_size, target_size, model_type, using_dish_features, using_ingr_features=False, pre_last_layer_size=128, using_unit_embedding=False, dropout=0.1):

        super(MultiLevelCalorie, self).__init__()

        # linear fully-connected layer for portion estimation
        self.using_unit_embedding = using_unit_embedding
        self.using_dish_features = using_dish_features

        self.using_ingr_features = using_ingr_features
        if using_ingr_features:
            self.ingr_stream = nn.Linear(embed_size, target_size)

        # returns N values for each N units
        self.unit_stream = SimUnit(embed_size, target_size, using_dish_features)
        
        # returns N values for the portion
        self.port_stream = SimUnit(embed_size, target_size, using_dish_features, pre_last_layer_size=pre_last_layer_size)

        # returns N values associated with the N units
        self.coef_stream = SimUnit(embed_size, target_size, using_dish_features, pre_last_layer_size=pre_last_layer_size)
       
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Sequential()

        self.sigmoid = nn.Sigmoid()

    def forward(self, ingr_embedding, img_embedding, unit_embedding, extra_features=None):

        unit_output, _ = self.unit_stream(ingr_embedding, img_embedding, extra_features)
        port_output, _ = self.port_stream(ingr_embedding, img_embedding, extra_features)
        calr_output, _ = self.coef_stream(ingr_embedding, img_embedding, extra_features)

        # unit probabilities are element-wise multiplied by portion values (e.g. vectors of N=6 both)
        # portion_unit = port_output.detach() * unit_output.detach()
        portion_unit = port_output * nn.Softmax(dim=1)(unit_output.detach())

        calr_output = torch.matmul(calr_output.unsqueeze(1), portion_unit.unsqueeze(2)).squeeze()

        #print ("calr_output: ", calr_output.size())
        #print ("unit_output: ", unit_output.size())
        #print ("portion_unit: ", portion_unit.size())

        return calr_output.unsqueeze(1), unit_output, port_output

# =======================================

class SimIndivJointCalorie(nn.Module):
    '''
        Decoder for just predicting unit, portion, and calorie 
        a single image and (ingredient name) --> predicts ingredient portion, unit and then calorie for that recipe
    '''
    def __init__(self, embed_size, target_size, using_extra_features, dropout=0.1):

        super(SimIndivJointCalorie, self).__init__()
        
        # linear fully-connected layer for portion estimation
        self.using_extra_features = using_extra_features
        if using_extra_features:
            input_size = 3*embed_size
        else:
            input_size = 2*embed_size

        self.linear = nn.Linear(input_size, embed_size//2)
        self.pre_unit_layer = nn.Linear(embed_size//2, embed_size//4)
        self.unit_layer = nn.Linear(embed_size//4, target_size)

        self.pre_portion_layer = nn.Linear(embed_size//2, embed_size//4)
        self.portion_unit_layer = nn.Linear(embed_size//2, target_size)
        self.portion_layer = nn.Linear(embed_size//2, 1)

        self.pre_calorie_layer = nn.Linear(embed_size//2, embed_size//2)
        #self.calorie_layer = nn.Linear(embed_size, 1)

        self.calorie_comb = nn.Linear(2, 1)
        self.relu = nn.ReLU()

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Sequential()

    def forward(self, ingr_embedding, img_embedding, extra_features=None):

        if self.using_extra_features:
            joint_embedding = torch.cat((img_embedding, ingr_embedding, extra_features), dim=1)
        else:
            joint_embedding = torch.cat((img_embedding, ingr_embedding), dim=1)
        
        joint_embedding = self.linear(joint_embedding)

        # unit embedding
        unit_embedding = self.pre_unit_layer(joint_embedding)
        # unit output
        unit_output = self.unit_layer(unit_embedding)

        # portion embedding
        portion_embedding = self.pre_portion_layer(joint_embedding)
        portion_unit_coefs = self.portion_unit_layer(joint_embedding)

        # portion output
        joint_unit_portion = torch.cat((unit_embedding.detach(), portion_embedding), dim=1)
        portion_output = self.portion_layer(joint_unit_portion)
        portion_output = self.relu(portion_output)

        # calorie embedding
        calorie_embedding = self.pre_calorie_layer(joint_embedding)
        # portion output
        #tiplet_unit_portion_pre_calorie = torch.cat((joint_unit_portion.detach(), calorie_embedding), dim=1)
        #calorie_output = self.calorie_layer(tiplet_unit_portion_pre_calorie)
        calorie_output1 = torch.matmul(joint_unit_portion.detach().unsqueeze(1), calorie_embedding.unsqueeze(2))
        #calorie_output1 = self.relu(calorie_output1.squeeze().unsqueeze(1))

        calorie_output2 = torch.matmul(portion_unit_coefs.unsqueeze(1), unit_output.detach().unsqueeze(2))
        #calorie_output2 = self.relu(calorie_output2.squeeze().unsqueeze(1))

        calorie_output1 = calorie_output1.squeeze().unsqueeze(1)
        calorie_output2 = calorie_output2.squeeze().unsqueeze(1)

        calorie_output = self.relu(self.calorie_comb(torch.cat((calorie_output1, calorie_output2), dim=1)))

        return calorie_output, portion_output, unit_output

# =======================================

class SimIndivVecCalorie(nn.Module):
    '''
        Decoder for just predicting calorie given 
        a single image and (ingredient name) --> predicts ingredient calorie for that recipe
        output: a vector of (token_size) which has a calorie estimation at i-th position for the i-th ingredient
    '''
    def __init__(self, token_size, embed_size, using_dish_features, pre_last_layer_size=128, nlayers=1, dropout=0.1):

        super(SimIndivVecCalorie, self).__init__()
        
        # linear fully-connected layer for portion estimation
        self.using_dish_features = using_dish_features
        if using_dish_features:
            input_size = 2*embed_size
        else:
            input_size = embed_size

        self.nlayers = nlayers
        if nlayers==2:
            self.base = nn.Sequential(
                    nn.Linear(input_size, pre_last_layer_size),
                    nn.BatchNorm1d(pre_last_layer_size),
                    nn.ReLU()
                )
        else:
            pre_last_layer_size = input_size

        self.last_linear = nn.Linear(pre_last_layer_size, token_size)
        self.relu = nn.ReLU()

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Sequential()

    def forward(self, ingr_mask, img_embedding, extra_features=None):

        if self.using_dish_features:
            joint_embedding = torch.cat((img_embedding, extra_features), dim=1)
        else:
            joint_embedding = img_embedding.squeeze().unsqueeze(1)
        
        if self.nlayers==2:
            pre_last = self.base(joint_embedding)
            joint_embedding = self.dropout(pre_last)

        ingr_calories_vector = self.relu(self.last_linear(joint_embedding))
        calorie_output = torch.matmul(ingr_mask.unsqueeze(1), ingr_calories_vector.squeeze().unsqueeze(2))

        return calorie_output.squeeze().unsqueeze(1)

# =======================================

class SimIndivCalorieClassRegress(nn.Module):
    '''
        Decoder for calorie classification than regression
        a single image and (ingredient name) --> predicts ingredient calorie for that recipe
    '''
    def __init__(self, calorie_classes_num, embed_size, using_extra_features, pre_last_layer_size=128, nlayers=1, dropout=0.1, activation="relu"):

        super(SimIndivCalorieClassRegress, self).__init__()
        
        # linear fully-connected layer for portion estimation
        self.using_extra_features = using_extra_features
        if using_extra_features:
            input_size = 3*embed_size
        else:
            input_size = 2*embed_size

        self.nlayers = nlayers
        if nlayers==2:
            self.base = nn.Sequential(
                    nn.Linear(input_size, pre_last_layer_size),
                    nn.BatchNorm1d(pre_last_layer_size),
                    nn.ReLU()
                )
        else:
            pre_last_layer_size = input_size

        self.classification = nn.Linear(pre_last_layer_size, calorie_classes_num)
        self.activation = ACTIVATON_LAYERS[activation]

        self.pre_regression = nn.Linear(pre_last_layer_size, calorie_classes_num)
        self.regression = nn.Linear(calorie_classes_num, 1)

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Sequential()

    def forward(self, ingr_embedding, img_embedding, extra_features=None):

        if self.using_extra_features:
            joint_embedding = torch.cat((img_embedding, ingr_embedding, extra_features), dim=1)
        else:
            joint_embedding = torch.cat((img_embedding, ingr_embedding), dim=1)
        
        if self.nlayers==2:
            pre_last = self.base(joint_embedding)
            joint_embedding = self.dropout(pre_last)

        calorie_class_outputs = self.classification(joint_embedding)
        pre_calorie_output = self.pre_regression(joint_embedding) * nn.Softmax(dim=1)(calorie_class_outputs.detach())
        calorie_output = self.activation(self.regression(pre_calorie_output))
    
        return calorie_output.squeeze().unsqueeze(1), calorie_class_outputs

# =======================================

def create_simple_decoder(args, ingr_vocab_size, unit_vocab_size, calorie_classes_num):

    activation = args.activation
    do_calorie_scale = args.do_calorie_scale
    do_prior_calorie = args.do_prior_calorie
    do_total_calorie = args.do_total_calorie
    pre_last_layer_size = args.pre_last_layer_size
    portion_uses_ingr_only = args.portion_uses_ingr_only
    calorie_classes_num = calorie_classes_num
    embed_size = args.embed_size
    dropout = args.dropout_decoder_i
    if args.model_type in {"simple_unit_dish", "simple_portion_dish", "simple_dishtocalorie_dish", "simple_ingrtocalorie_dish", "simple_ingrtojointcalorie_dish", "simple_heirarchitocalorie_dish"}:
        using_dish_features = True
    else:
        using_dish_features = False

    # build simple decoder
    if args.model_type in {"simple_unit", "simple_unit_dish"}:
        simple_decoder = SimUnit(embed_size, unit_vocab_size, using_dish_features, dropout=dropout)
    elif args.model_type in {"simple_portion", "simple_portion_dish"}:
        simple_decoder = SimPortion(embed_size, using_dish_features, dropout=dropout)
    elif args.model_type in {"simple_dishtocalorie", "simple_dishtocalorie_dish"}:
        simple_decoder = SimCalorie(embed_size, using_dish_features, dropout=dropout)
    elif args.model_type in {"simple_ingrtocalorie", "simple_ingrtocalorie_dish"}:
        simple_decoder = SimIndivCalorie(embed_size, using_dish_features, dropout=dropout, pre_last_layer_size=pre_last_layer_size)
    elif args.model_type in {"simple_ingrtojointcalorie", "simple_ingrtojointcalorie_dish"}:
        simple_decoder = SimIndivJointCalorie(embed_size, unit_vocab_size, using_dish_features, dropout=dropout)
    elif args.model_type in {"simple_ingrtoveccalorie", "simple_ingrtoveccalorie_dish"}:
        simple_decoder = SimIndivVecCalorie(ingr_vocab_size-1, embed_size, using_dish_features, dropout=dropout)
    elif args.model_type in {"simple_ingrtocalorieregress", "simple_ingrtocalorieregress_dish"}:
        simple_decoder = SimIndivCalorieClassRegress(calorie_classes_num, embed_size, using_dish_features, dropout=dropout)        
    elif args.model_type in {"simple_heirarchitocalorie", "simple_heirarchitocalorie_dish", "simple_heirarchitoportion", "simple_heirarchitoportion_dish"}:
        simple_decoder = HeirarchicalModel(embed_size, unit_vocab_size, args.model_type, using_dish_features, 
                                           dropout=dropout, pre_last_layer_size=pre_last_layer_size, 
                                           portion_uses_ingr_only=portion_uses_ingr_only, 
                                           do_calorie_scale=do_calorie_scale,
                                           do_total_calorie=do_total_calorie,
                                           do_prior_calorie=do_prior_calorie,
                                           activation=activation)
    elif args.model_type in {"simple_ml_calorie1", "simple_ml_calorie1_dish", "simple_ml_calorie2", "simple_ml_calorie2_dish"}:
        simple_decoder = MultiLevelCalorie(embed_size, unit_vocab_size, args.model_type, using_dish_features, dropout=dropout)

    return simple_decoder

# =======================================

def run_simple_decoder(decoder, outputs, ingr_features, img_features, unit_features, dish_features, ingr_calorie_vec_mask, model_type):

    if model_type in {"simple_unit", "simple_unit_dish"}:
        outputs["unit_output"], _ = decoder(ingr_features, img_features, dish_features)
    elif model_type in {"simple_portion", "simple_portion_dish"}:
        outputs["portion_output"], _ = decoder(ingr_features, img_features, unit_features, dish_features)
    elif model_type in {"simple_dishtocalorie", "simple_dishtocalorie_dish"}:
        outputs["calorie_output"], _ = decoder(img_features, dish_features)
    elif model_type in {"simple_ingrtocalorie", "simple_ingrtocalorie_dish"}:
        outputs["calorie_output"], _ = decoder(ingr_features, img_features, dish_features)
    elif model_type in {"simple_ingrtojointcalorie", "simple_ingrtojointcalorie_dish"}:
        outputs["calorie_output"], outputs["portion_output"], outputs["unit_output"] = decoder(ingr_features, img_features, dish_features)
    elif model_type in {"simple_ingrtoveccalorie", "simple_ingrtoveccalorie_dish"}:
        outputs["calorie_output"] = decoder(ingr_calorie_vec_mask, img_features, dish_features)
    elif model_type in {"simple_ingrtocalorieregress", "simple_ingrtocalorieregress_dish"}:
        outputs["calorie_output"], outputs["calorie_class_output"] = decoder(ingr_features, img_features, dish_features)
    elif model_type in {"simple_heirarchitocalorie", "simple_heirarchitocalorie_dish"}:
        outputs["calorie_output"], outputs["unit_output"], outputs["portion_output"], outputs["recipe_calorie_output"], outputs["prior_calorie_output"] = decoder(ingr_features, img_features, dish_features)
        if outputs["recipe_calorie_output"]==None:
            del outputs["recipe_calorie_output"]
        if outputs["prior_calorie_output"]==None:
            del outputs["prior_calorie_output"]            
    elif model_type in {"simple_heirarchitoportion", "simple_heirarchitoportion_dish"}:
        outputs["portion_output"], outputs["unit_output"], outputs["portion_output2"] = decoder(ingr_features, img_features, dish_features)
    elif model_type in {"simple_ml_calorie1", "simple_ml_calorie1_dish", "simple_ml_calorie2", "simple_ml_calorie2_dish"}:
        outputs["calorie_output"], outputs["unit_output"], outputs["portion_distrib"] = decoder(ingr_features, img_features, dish_features)

    
# =======================================