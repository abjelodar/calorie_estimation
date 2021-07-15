# Copyright (c) Robotic and Action Perception Lab (RPAL) at University of South Florida
# by Ahmad Babaeian Jelodar

import torch
import torch.nn as nn
import random
import numpy as np
from modules.encoder import EmeddingAtt
from modules.utils import _reset_parameters
import torch.nn.functional as F

# =======================================

class AttDecoder(nn.Module):
    '''
        Transformer decoder for just estimating a vector output for each position.
    '''
    def __init__(self, embed_size, target_size, dropout=0.1, nlayers=6, dim_feedforward=256, nhead=8):

        super(AttDecoder, self).__init__()
        decoder_norm = nn.LayerNorm(embed_size)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, nlayers, decoder_norm)

        # linear fully-connected layer for estimation of a vector output
        self.linear = nn.Linear(embed_size, target_size)

        _reset_parameters(self.parameters())

    def forward(self, ingr_features, img_features, ingr_mask, img_mask):

        embedding = self.transformer_decoder(ingr_features, img_features, tgt_key_padding_mask=ingr_mask, memory_key_padding_mask=img_mask)
        out = self.linear(embedding)

        # returns the vector for each position, and the transformer decoder output for each position
        return out.permute(1, 2, 0).contiguous(), embedding

# =======================================

class ComplexModel(nn.Module):
    '''
        Complex model version of the simple model.
    '''
    def __init__(self, embed_size, target_size, args, model_output='all', dropout=0.3, nlayers=6, dim_feedforward=256, nhead=8, cnn_reduced=False, pre_last_layer_size=128, seq_size=10):

        super(ComplexModel, self).__init__()
        
        self.model_output = model_output
        self.coef_encoder = nn.Sequential(
                    nn.Linear(embed_size, pre_last_layer_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(pre_last_layer_size, target_size),
                    nn.ReLU()
                )

        
        self.unit_encoder = nn.Sequential(
                    nn.Linear(embed_size, pre_last_layer_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(pre_last_layer_size, target_size),
                    nn.ReLU()
                )

        self.port_encoder = nn.Sequential(
                    nn.Linear(embed_size, pre_last_layer_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),                    
                    nn.Linear(pre_last_layer_size, 1),
                    nn.ReLU()
                )

        self.ingr_img_encoder = AttDecoder(embed_size, 1, nlayers=nlayers, dim_feedforward=dim_feedforward)

        self.combine_calories = EmeddingAtt(embed_size, 1)

        self.linear_calorie = nn.Linear(embed_size, 1)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        _reset_parameters(self.parameters())

    def forward(self, ingr_features, img_features, ingr_mask, img_mask):
        
        enc_output, enc_embedding = self.ingr_img_encoder(ingr_features, img_features, ingr_mask, img_mask)        
        enc_embedding = enc_embedding.permute(1, 0, 2).contiguous()
        ingr_mask = ingr_mask.unsqueeze(2)

        if self.model_output=="just_ingr_calories":
            enc_output = enc_output.squeeze().unsqueeze(2)
            enc_output = (enc_output * (~ingr_mask).float())
            return None, None, self.relu(enc_output.squeeze()), None

        elif self.model_output=="calories_and_calorie":

            calories_output = enc_output.squeeze().unsqueeze(2)
            calories_output = (calories_output * (~ingr_mask).float()).squeeze()
            enc_embedding = (enc_embedding * (~ingr_mask).float())

            calorie_embedding = self.relu(self.combine_calories(enc_embedding, ingr_mask, enc_embedding.detach())).squeeze()
            calorie_output = self.linear_calorie(self.dropout(calorie_embedding))
            return None, None, self.relu(calories_output.squeeze()), calorie_output

        elif self.model_output=="portions_and_calorie":
            portions_output = enc_output.squeeze().unsqueeze(2)
            portions_output = (portions_output * (~ingr_mask).float()).squeeze()
            enc_embedding = (enc_embedding * (~ingr_mask).float())
            calorie_embedding = self.relu(self.combine_calories(enc_embedding, ingr_mask, enc_embedding.detach())).squeeze()
            calorie_output = self.linear_calorie(self.dropout(calorie_embedding))
            return None, self.relu(portions_output), None, calorie_output

        elif self.model_output=="just_calorie":

            enc_embedding = (enc_embedding * (~ingr_mask).float())
            calorie_embedding = self.relu(self.combine_calories(enc_embedding, ingr_mask, enc_embedding.detach())).squeeze()
            calorie_output = self.linear_calorie(self.dropout(calorie_embedding))
            return None, None, None, calorie_output

        elif self.model_output=="all":

            enc_embedding = (enc_embedding * (~ingr_mask).float())

            coef_output = self.coef_encoder(enc_embedding.detach())
            unit_output = self.unit_encoder(enc_embedding.detach())
            port_output = enc_output.squeeze().unsqueeze(2) #self.port_encoder(enc_embedding.detach())

            coef_output = (coef_output * (~ingr_mask).float())
            unit_output = (unit_output * (~ingr_mask).float())
            port_output = (port_output * (~ingr_mask).float())
            portion_coef = torch.matmul(port_output.unsqueeze(3), coef_output.unsqueeze(2)).squeeze()

            calories_output = torch.matmul(portion_coef.unsqueeze(2), unit_output.unsqueeze(3))
            calories_output = calories_output.squeeze().unsqueeze(2)

            calories_output = (calories_output * (~ingr_mask).float())

            # apply attention to get an attended vector from a sequence & then a fully-connected layer
            calorie_embedding = self.combine_calories(enc_embedding, ingr_mask, enc_embedding.detach()).squeeze()
            calorie_output = self.linear_calorie(self.dropout(calorie_embedding))

            return unit_output.permute(0, 2, 1).contiguous(), port_output, self.relu(calories_output.squeeze()), calorie_output

# =======================================

class ComplexModel_P(nn.Module):
    '''
        Previous Complex model version of the simple model.
    '''
    def __init__(self, embed_size, target_size, args, dropout=0.3, nlayers=6, nhead=8, cnn_reduced=False, ingr_vocab_size=0, pre_last_layer_size=128, dim_feedforward=512, flattened_weights=None):

        super(ComplexModel_P, self).__init__()
        
        self.cnn_reduced = cnn_reduced
        self.target_size = target_size
        self.ingr_vocab_size = ingr_vocab_size
        self.unit_encoder = AttDecoder(embed_size//2, target_size, dim_feedforward=dim_feedforward//2, nlayers=nlayers)
        self.calorie_decoder = AttDecoder(embed_size, 1, dim_feedforward=dim_feedforward, nlayers=nlayers)
        self.ingrunitcalories_encoder = AttDecoder(embed_size, 1, dim_feedforward=dim_feedforward, nlayers=nlayers)
        self.ingrunitportions_encoder = AttDecoder(embed_size, 1, dim_feedforward=dim_feedforward, nlayers=nlayers)
        self.combine_units = EmeddingAtt(embed_size, 1)
        self.combine_calories = EmeddingAtt(embed_size, 1)

        self.sample_img_features = nn.Linear(embed_size, embed_size//2)
        self.sample_ingr_features = nn.Linear(embed_size, embed_size//2)
        self.sample_portion_embedding = nn.Linear(embed_size, embed_size//2)
        self.sample_ingrunit_embedding = nn.Linear(embed_size, embed_size//2)

        self.linear_calorie = nn.Linear(embed_size, 1)
        self.linear_alignment = nn.Linear(embed_size, 1)

        self.average_two_heads = nn.Linear(2, 1)

        # matrix of weights to map each ingredient portion (6 values) to a calorie value for that ingredient
        self.ingrportions_to_calories = nn.Linear(ingr_vocab_size*target_size, 1)
        self.ingrportions_to_calories.weight.requires_grad = False
        self.ingrportions_to_calories.bias.requires_grad = False
        self.ingrportions_to_calories.weight.copy_(torch.tensor(flattened_weights))
        self.ingrportions_to_calories.bias.copy_(torch.tensor([0]))

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=2)

        _reset_parameters(self.parameters())

    def forward_ingrportion_to_calorie(self, units_output, portions_output, ingr_portion_mask):

        portions_squeezed = portions_output.squeeze()

        unit_softmaxes = self.softmax(units_output)
        _, pred_idx = torch.max(unit_softmaxes, dim=2)
        unit_binaries = F.one_hot(pred_idx, num_classes=self.target_size)
        unit_portions = unit_binaries.detach() * portions_squeezed.unsqueeze(2)
        unit_portions_repeated = unit_portions.repeat(1, 1, self.ingr_vocab_size)

        masked_unit_portions = unit_portions_repeated * ingr_portion_mask
        return self.ingrportions_to_calories(masked_unit_portions)

    def forward(self, ingr_features, img_features, ingr_mask, img_mask, ingr_portion_mask=None):
        
        # decoder for calorie (embed_size: N)
        calories_output, calorie_embedding = self.calorie_decoder(ingr_features, img_features, ingr_mask, img_mask) # N

        # sample N/2 image and ingr features of N features
        sampled_img_features = self.sample_img_features(img_features.detach())
        sampled_ingr_features = self.sample_ingr_features(ingr_features.detach())
        '''print ("sampled_img_features", sampled_img_features.size())
        print ("sampled_ingr_features", sampled_ingr_features.size())'''

        # decoder for calorie (embed_size: N/2)
        units_output, unit_embedding = self.unit_encoder(sampled_ingr_features, sampled_img_features, ingr_mask, img_mask) # N/2
        # ingr & unit encoding for calories estimation
        iu_embedding = torch.cat((sampled_ingr_features, unit_embedding.detach()), dim=2)

        # ingr & unit encoding for portions estimation 
        #iup_embedding = torch.cat((sampled_ingr_features, unit_embedding), dim=2)
        portions_output, portion_embedding = self.ingrunitportions_encoder(iu_embedding, img_features.detach(), ingr_mask, img_mask) # N
        
        portions_output = self.relu(portions_output.squeeze())

        # re-arrange embeddings
        unit_embedding = unit_embedding.permute(1, 0, 2).contiguous().detach() # N/2
        calorie_embedding = calorie_embedding.permute(1, 0, 2).contiguous()    # N

        # alignment module
        sampled_iu_embedding = self.sample_ingrunit_embedding(iu_embedding.detach())
        sampled_portion_embedding = self.sample_portion_embedding(portion_embedding.detach())
        ingrunitportion_embedding = torch.cat((sampled_iu_embedding, sampled_portion_embedding), dim=2)

        calories_alignment_output, alignment_embedding = self.ingrunitcalories_encoder(ingrunitportion_embedding, img_features.detach(), ingr_mask, img_mask) # N
       
        # add a dimension to the ingredient mask for computation purposes
        ingr_mask = ingr_mask.unsqueeze(2)

        # units output
        units_output = units_output.permute(0, 2, 1).contiguous()
        units_output = (units_output * (~ingr_mask).float()).squeeze()

        # calories output
        calories_output = calories_output.squeeze().unsqueeze(2)
        calories_output = (calories_output * (~ingr_mask).float()).squeeze()

        # mask out the trailing elements of embeddings based on input mask
        unit_embedding = (unit_embedding * (~ingr_mask).float())
        calorie_embedding = (calorie_embedding * (~ingr_mask).float())

        # alignment masking 1
        alignment_embedding = alignment_embedding.permute(1, 0, 2).contiguous()    # N
        alignment_embedding = (alignment_embedding * (~ingr_mask).float())
        
        # alignment masking 2
        calories_alignment_output = calories_alignment_output.squeeze().unsqueeze(2)
        calories_alignment_output = (calories_alignment_output * (~ingr_mask).float()).squeeze()       
        calorie_feats = self.relu(self.combine_calories(calorie_embedding, ingr_mask, calorie_embedding.detach())).squeeze()

        alignment_feats = self.relu(self.combine_units(alignment_embedding, ingr_mask, alignment_embedding.detach())).squeeze()

        calorie_alignment2_output = self.linear_calorie(self.dropout(calorie_feats.squeeze()))

        calorie_alignment_output = self.linear_alignment(self.dropout(alignment_feats.squeeze()))

        # concat two calorie heads
        calorie2heads = torch.cat((calorie_alignment2_output.detach(), calorie_alignment_output.detach()), dim=1)
        calorie_output = self.average_two_heads(calorie2heads)

        if ingr_portion_mask is None:
            ingr_portion_output = None
        else:
            ingr_portion_output = self.forward_ingrportion_to_calorie(units_output, portions_output, ingr_portion_mask)

        return units_output.permute(0, 2, 1).contiguous(), portions_output, self.relu(calories_output.squeeze()), calorie_output, self.relu(calories_alignment_output.squeeze()), self.relu(calorie_alignment_output), self.relu(calorie_alignment2_output), ingr_portion_output

# =======================================

class CalorieDecoder(nn.Module):
    '''
        Decoder for just predicting a list of calories given a list of ingredients
    '''
    def __init__(self, embed_size, args, dropout=0.1, nlayers=6, nhead=8):

        super(CalorieDecoder, self).__init__()
        decoder_norm = nn.LayerNorm(args.embed_size)
        decoder_layer = nn.TransformerDecoderLayer(d_model=args.embed_size, nhead=nhead, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, nlayers, decoder_norm)
        
        # linear fully-connected layer for calorie estimation (a class)
        self.linear = nn.Linear(embed_size, 1)

        self.embedding_att = EmeddingAtt(1, 1)

        self.relu = nn.ReLU()

        _reset_parameters(self.parameters())

    def forward(self, ingr_features, img_features, ingr_mask, img_mask):

        decoder_out = self.transformer_decoder(ingr_features, img_features, tgt_key_padding_mask=ingr_mask, memory_key_padding_mask=img_mask)
        calories_out = self.linear(decoder_out)
        calories_out = calories_out.permute(1, 0, 2).contiguous()
        ingr_mask = ingr_mask.unsqueeze(2)

        calories_out = (calories_out * (~ingr_mask).float())

        # apply attention to get an attended vector from a sequence & then a fully-connected layer
        calorie_out = self.relu(self.embedding_att(calories_out, ingr_mask)).squeeze()

        return calories_out, calorie_out.unsqueeze(1)

# =======================================

class AttJointDecoder(nn.Module):
    '''
        Decoder for predicting a list of (unit, portion) pairs given a list of ingredients
    '''
    def __init__(self, embed_size, target_size, args, pre_last_layer_size=128, dropout=0.1, nlayers=6, nhead=8):

        super(AttJointDecoder, self).__init__()
        decoder_norm = nn.LayerNorm(args.embed_size)
        decoder_layer = nn.TransformerDecoderLayer(d_model=args.embed_size, nhead=nhead, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, nlayers, decoder_norm)
        
        # linear fully-connected layer for unit estimation (a class)
        self.unit_linear = nn.Linear(embed_size, target_size)
        
        # linear fully-connected layers for portion estimation (a vlaue not task)
        self.portion_mid_linear = nn.Linear(embed_size, pre_last_layer_size)
        self.portion_linear = nn.Linear(pre_last_layer_size, 1)

        _reset_parameters(self.parameters())

    def forward(self, ingr_features, img_features, ingr_mask, img_mask):

        decoder_out = self.transformer_decoder(ingr_features, img_features, tgt_key_padding_mask=ingr_mask, memory_key_padding_mask=img_mask)
        unit_out = self.unit_linear(decoder_out)

        portion_pre_last = self.portion_mid_linear(decoder_out)
        portion_out = self.portion_linear(portion_pre_last)

        return unit_out.permute(1, 2, 0).contiguous(), portion_out.permute(1, 2, 0).contiguous()

# =======================================

class AttJointCalDecoder(nn.Module):
    '''
        Decoder for predicting a list of (unit, portion) pairs given a list of units and calorie
    '''
    def __init__(self, embed_size, target_size, args, pre_last_layer_size=64, dropout=0.1, nlayers=6, nhead=8):

        super(AttJointCalDecoder, self).__init__()
        decoder_norm = nn.LayerNorm(args.embed_size)
        decoder_layer = nn.TransformerDecoderLayer(d_model=args.embed_size, nhead=nhead, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, nlayers, decoder_norm)
        
        # linear fully-connected layer for unit estimation (a class)
        self.unit_mid_linear = nn.Linear(embed_size, pre_last_layer_size)
        self.unit_linear = nn.Linear(pre_last_layer_size, target_size)
        
        # linear fully-connected layers for portion estimation (a vlaue not task)
        self.portion_mid_linear = nn.Linear(embed_size, pre_last_layer_size)
        self.portion_linear = nn.Linear(pre_last_layer_size, 1)

        # linear fully-connected layer & attention for calorie estimation
        self.calorie_pre_linear = nn.Linear(2*pre_last_layer_size, pre_last_layer_size)
        self.embedding_att = EmeddingAtt(pre_last_layer_size, 1)

        self.train_base_wcalorie = args.train_base_wcalorie
        
        # the stream that contains a direct stream from image to calorie
        self.image_to_calorie_stream = args.image_to_calorie_stream
        self.img_to_calorie_cnn = EmeddingAtt(embed_size, 1)
        self.img_to_calorie_linear = nn.Linear(embed_size, pre_last_layer_size)

        if self.image_to_calorie_stream:
            self.calorie_linear = nn.Linear(2*pre_last_layer_size, 1)
        else:
            self.calorie_linear = nn.Linear(pre_last_layer_size, 1)
        
        _reset_parameters(self.parameters())

    def net_base(self, ingr_features, img_features, ingr_mask, img_mask):

        decoder_out = self.transformer_decoder(ingr_features, img_features, tgt_key_padding_mask=ingr_mask, memory_key_padding_mask=img_mask)
        unit_pre_last = self.unit_mid_linear(decoder_out)
        unit_out = self.unit_linear(unit_pre_last)

        portion_pre_last = self.portion_mid_linear(decoder_out)
        portion_out = self.portion_linear(portion_pre_last)

        return unit_pre_last, unit_out, portion_pre_last, portion_out

    def forward(self, ingr_features, img_features, ingr_mask, img_mask):

        # base net for all three outputs: 1. unit 2. portion 3. calorie
        unit_pre_last, unit_out, portion_pre_last, portion_out = self.net_base(ingr_features, img_features, ingr_mask, img_mask)

        if not self.train_base_wcalorie:
            # if back-prop of calorie loss does not go through the base net.
            with torch.no_grad():
                unit_pre_last_wo_grad, _, portion_pre_last_wo_grad, _ = self.net_base(ingr_features, img_features, ingr_mask)

            unit_pre_embed = unit_pre_last_wo_grad
            portion_pre_embed = portion_pre_last_wo_grad
        else:
            # if back-prop of calorie loss goes through the base net.
            unit_pre_embed = unit_pre_last
            portion_pre_embed = portion_pre_last

        # change the tensors shape for future operations of calorie estimation
        unit_pre_embed = unit_pre_embed.permute(1, 0, 2).contiguous()
        portion_pre_embed = portion_pre_embed.permute(1, 0, 2).contiguous()
        ingr_mask = ingr_mask.unsqueeze(2)

        # mask outputs that should not be considered in loss from the unit & portion sequences
        unit_embeddings = (unit_pre_embed * (~ingr_mask).float())
        portion_embeddings = (portion_pre_embed * (~ingr_mask).float())

        # concat unit & portion sequence embeddings 
        unit_portion_embeddings = torch.cat((unit_embeddings, portion_embeddings), dim=2)

        # apply linear fully connected layer to the concatenated sequence features
        unit_portion_embeddings = self.calorie_pre_linear(unit_portion_embeddings)

        # apply attention to get an attended vector from a sequence & then a fully-connected layer
        unit_portion_vector = self.embedding_att(unit_portion_embeddings, ingr_mask)
        unit_portion_vector = unit_portion_vector.squeeze()

        if self.image_to_calorie_stream:
            img_features = img_features.permute(1, 0, 2).contiguous()
            cnn_calorie = self.img_to_calorie_cnn(img_features)
            cnn_calorie = cnn_calorie.squeeze()
            cnn_vector = self.img_to_calorie_linear(cnn_calorie)
            cnn_vector = cnn_vector.squeeze()
            calorie_vector = torch.cat((unit_portion_vector, cnn_vector), dim=1)
        else:
            calorie_vector = unit_portion_vector
        
        calorie_out = self.calorie_linear(calorie_vector)

        return unit_out.permute(1, 2, 0).contiguous(), portion_out.permute(1, 2, 0).contiguous(), calorie_out

# =======================================

class JointCalDoubleDecoder(nn.Module):
    '''
        Decoder for predicting a list of (unit, portion) pairs given a list of units and calorie
    '''
    def __init__(self, embed_size, target_size, args, pre_last_layer_size=128, dropout=0.1, nlayers=6, nhead=8, nlayers_2=6, nhead_2=8):

        super(JointCalDoubleDecoder, self).__init__()

        # transformer-decoder for unit prediction given (ingredients, image features)
        decoder_norm = nn.LayerNorm(args.embed_size)
        decoder_layer = nn.TransformerDecoderLayer(d_model=args.embed_size, nhead=nhead, dropout=dropout)
        self.transformer_decoder_unit = nn.TransformerDecoder(decoder_layer, nlayers, decoder_norm)

        decoder_norm_portion = nn.LayerNorm(args.embed_size)
        decoder_layer_portion = nn.TransformerDecoderLayer(d_model=args.embed_size, nhead=nhead, dropout=dropout)
        self.transformer_encoder_portion = nn.TransformerDecoder(decoder_layer_portion, nlayers, decoder_norm_portion)

        # transformer-decoder for portion prediction given (unit, portion embeddings)
        decoder_norm_2 = nn.LayerNorm(args.embed_size)
        decoder_layer_2 = nn.TransformerDecoderLayer(d_model=args.embed_size, nhead=nhead, dropout=dropout)
        self.transformer_decoder_portion = nn.TransformerDecoder(decoder_layer_2, nlayers_2, decoder_norm_2)

        # linear fully-connected layer for unit estimation (a class)
        self.unit_mid_linear = nn.Linear(embed_size, pre_last_layer_size)
        self.unit_linear = nn.Linear(pre_last_layer_size, target_size)
        
        # linear fully-connected layers for portion estimation (a vlaue not task)
        self.portion_mid_linear = nn.Linear(embed_size, pre_last_layer_size)
        self.portion_linear = nn.Linear(pre_last_layer_size, 1)

        # linear fully-connected layer & attention for calorie estimation
        self.embedding_att = EmeddingAtt(3*args.embed_size, 1)
        if args.model_type in {"jointwithcaloriedish", "jointwithcaloriedoubledish"}:
            self.pre_calorie_linear = nn.Linear(4*args.embed_size, pre_last_layer_size)
            self.layer_norm = nn.LayerNorm(4*args.embed_size)
        else:

            self.pre_calorie_linear = nn.Linear(3*args.embed_size, pre_last_layer_size)
            self.layer_norm = nn.LayerNorm(3*args.embed_size)

        self.train_base_wcalorie = args.train_base_wcalorie
        
        # the stream that contains a direct stream from image to calorie
        self.image_to_calorie_stream = args.image_to_calorie_stream
        self.img_to_calorie_cnn = EmeddingAtt(embed_size, 1)
        self.img_to_calorie_linear = nn.Linear(embed_size, pre_last_layer_size)

        if self.image_to_calorie_stream:
            self.calorie_linear = nn.Linear(2*pre_last_layer_size, 1)
            self.last_layer_norm = nn.LayerNorm(2*pre_last_layer_size)
        else:
            self.calorie_linear = nn.Linear(pre_last_layer_size, 1)
        
        self.relu = nn.ReLU()

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Sequential()

        _reset_parameters(self.parameters())

    def forward(self, ingr_features, img_features, ingr_mask, img_mask, dish_features=None):

        # unit encoder
        unit_embeddings = self.transformer_decoder_unit(ingr_features, img_features, tgt_key_padding_mask=ingr_mask, memory_key_padding_mask=img_mask)
        unit_pre_last = self.unit_mid_linear(unit_embeddings)
        # regularization (dropout)
        unit_pre_last = self.dropout(unit_pre_last)
        unit_out = self.unit_linear(unit_pre_last)

        portion_embeddings = self.transformer_encoder_portion(ingr_features, img_features, tgt_key_padding_mask=ingr_mask, memory_key_padding_mask=img_mask)

        # portion loss does not go through unit embeddings
        pre_unit = unit_embeddings.detach()
        pre_portion = portion_embeddings

        # portion encoder
        portion_decoder_out = self.transformer_decoder_portion(pre_unit, pre_portion, tgt_key_padding_mask=ingr_mask, memory_key_padding_mask=ingr_mask)
        portion_pre_last = self.portion_mid_linear(portion_decoder_out)
        # regularization (dropout)
        portion_pre_last = self.dropout(portion_pre_last)
        portion_out = self.portion_linear(portion_pre_last)

        # change the tensors shape for future operations of calorie estimation
        unit_pre_calorie = unit_embeddings.detach().permute(1, 0, 2).contiguous()
        portion_pre_calorie = portion_decoder_out.detach().permute(1, 0, 2).contiguous()
        ingr_pre_calorie = ingr_features.detach().permute(1, 0, 2).contiguous()
        ingr_mask = ingr_mask.unsqueeze(2)
        
        # mask outputs that should not be considered in loss from the unit & portion sequences
        unit_pre_calorie = (unit_pre_calorie * (~ingr_mask).float())
        portion_pre_calorie = (portion_pre_calorie * (~ingr_mask).float())
        ingr_pre_calorie = (ingr_pre_calorie * (~ingr_mask).float())

        # concat unit, portion, & ingredient sequence embeddings 
        unit_portion_ingrs_embeddings = torch.cat((unit_pre_calorie, portion_pre_calorie, ingr_pre_calorie), dim=2)

        # apply attention to get an attended vector from a sequence & then a fully-connected layer
        unit_portion_vector = self.embedding_att(unit_portion_ingrs_embeddings, ingr_mask)
        unit_portion_vector = unit_portion_vector.squeeze()

        # apply linear fully connected layer to the concatenated sequence features
        if dish_features!=None:
            unit_portion_vector = torch.cat((unit_portion_vector, dish_features.squeeze()), dim=1)

        # layer normalization
        unit_portion_vector = self.layer_norm(unit_portion_vector)

        # fully connected layer
        unit_portion_vector = self.pre_calorie_linear(unit_portion_vector)

        # regularization (dropout)
        unit_portion_vector = self.dropout(unit_portion_vector)

        if self.image_to_calorie_stream:
            img_features = img_features.permute(1, 0, 2).contiguous()
            cnn_calorie = self.img_to_calorie_cnn(img_features)
            cnn_calorie = cnn_calorie.squeeze()
            cnn_vector = self.img_to_calorie_linear(cnn_calorie)
            cnn_vector = cnn_vector.squeeze()
            calorie_vector = torch.cat((unit_portion_vector, cnn_vector), dim=1)
            calorie_vector = self.last_layer_norm(calorie_vector)
        else:
            calorie_vector = unit_portion_vector
        
        calorie_out = self.relu(self.calorie_linear(calorie_vector))

        return unit_out.permute(1, 2, 0).contiguous(), portion_out.permute(1, 2, 0).contiguous(), calorie_out

# =======================================

class TripletCal(nn.Module):
    '''
        Decoder for predicting a list of (unit, portion, calorie) triplets given a list of units and calorie
    '''
    def __init__(self, embed_size, target_size, args, pre_last_layer_size=128, dropout=0.1, nlayers=6, nhead=8, nlayers_2=6, nhead_2=8):

        super(TripletCal, self).__init__()

        # transformer-decoder for unit prediction given (ingredients, image features)
        decoder_norm = nn.LayerNorm(args.embed_size)
        decoder_layer = nn.TransformerDecoderLayer(d_model=args.embed_size, nhead=nhead, dropout=dropout)
        self.unit_stream = nn.TransformerDecoder(decoder_layer, nlayers, decoder_norm)

        decoder_layer_portion = nn.TransformerDecoderLayer(d_model=args.embed_size, nhead=nhead, dropout=dropout)
        self.portion_stream = nn.TransformerDecoder(decoder_layer_portion, nlayers, decoder_norm)

        # transformer-decoder for portion prediction given (unit, portion embeddings)
        decoder_layer_2 = nn.TransformerDecoderLayer(d_model=args.embed_size, nhead=nhead, dropout=dropout)
        self.coef_stream = nn.TransformerDecoder(decoder_layer_2, nlayers_2, decoder_norm)

        # linear fully-connected layer for unit estimation (a class)
        self.unit_linear = nn.Linear(args.embed_size, target_size)
        
        # linear fully-connected layer for unit estimation (a class)
        self.portion_linear = nn.Linear(args.embed_size, 1)

        # linear fully-connected layer for unit estimation (a class)
        self.coef_linear = nn.Linear(args.embed_size, target_size)
        
        self.relu = nn.ReLU()

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

        _reset_parameters(self.parameters())

    def forward(self, ingr_features, img_features, ingr_mask, img_mask, dish_features=None):

        # unit encoder
        unit_embeddings = self.unit_stream(ingr_features, img_features, tgt_key_padding_mask=ingr_mask, memory_key_padding_mask=img_mask)
        unit_embeddings = self.dropout(unit_embeddings)
        unit_output = self.unit_linear(unit_embeddings)

        # portion encoder
        portion_embeddings = self.portion_stream(ingr_features, img_features, tgt_key_padding_mask=ingr_mask, memory_key_padding_mask=img_mask)
        portion_embeddings = self.dropout(portion_embeddings)
        portion_output = self.portion_linear(portion_embeddings)

        # coefs (encoder)
        coef_embeddings = self.unit_stream(ingr_features, img_features, tgt_key_padding_mask=ingr_mask, memory_key_padding_mask=img_mask)
        coef_embeddings = self.dropout(coef_embeddings)
        coef_output = self.coef_linear(coef_embeddings)
        portion_coef = torch.matmul(coef_output.unsqueeze(2), portion_output.unsqueeze(2))
        output_value = torch.matmul(portion_coef, unit_output.detach().unsqueeze(3))
        return unit_out.permute(1, 2, 0).contiguous(), portion_out.permute(1, 2, 0).contiguous(), output_value

# =======================================

class JustIngrs2Calorie(nn.Module):
    '''
        Decoder for predicting a list of (unit, portion) pairs given a list of units
    '''
    def __init__(self, embed_size, args, pre_last_layer_size=128, dropout=0.1, nlayers=6, nhead=8):

        super(JustIngrs2Calorie, self).__init__()
        decoder_norm = nn.LayerNorm(args.embed_size)
        decoder_layer = nn.TransformerDecoderLayer(d_model=args.embed_size, nhead=nhead, dropout=dropout)
        self.transformer_decoder = model = nn.TransformerDecoder(decoder_layer, nlayers, decoder_norm)
        
        # Attention on different ingredient embeddings
        self.ingredients_att = EmeddingAtt(embed_size, 1)

        # last two fully connected layers
        self.pre_last_layer = nn.Linear(embed_size, pre_last_layer_size)
        self.calorie_linear = nn.Linear(pre_last_layer_size, 1)
        
        _reset_parameters(self.parameters())

    def forward(self, ingr_features, img_features, ingr_mask, img_mask):

        decoder_out = self.transformer_decoder(ingr_features, img_features, tgt_key_padding_mask=ingr_mask, memory_key_padding_mask=img_mask)

        ingrs_att = self.ingredients_att(decoder_out.permute(1, 0, 2).contiguous())
        
        ingrs_att = ingrs_att.squeeze()

        pre_last = self.pre_last_layer(ingrs_att)

        out = self.calorie_linear(pre_last)

        return out

# =======================================

def create_decoder(args, unit_vocab_size, ingr_vocab_size, states_vocab_size, flattened_weights=None):

    embed_size = args.embed_size
    u_layers = args.transf_layers_units
    u_heads = args.n_att_units
    dropout = args.dropout_decoder_i
    dim_feedforward = args.dim_feedforward
    cnn_reduced = args.cnn_reduced
    model_output = args.model_output

    # build image & ingredient decoder
    if args.model_type=="unit":
        decoder_att = AttDecoder(embed_size, unit_vocab_size, nlayers=u_layers, nhead=u_heads, dropout=dropout)
    elif args.model_type=="joint":
        decoder_att = AttJointDecoder(embed_size, unit_vocab_size, args, nlayers=u_layers, nhead=u_heads, dropout=dropout)
    elif args.model_type=="jointwithcalorie":
        decoder_att = AttJointCalDecoder(embed_size, unit_vocab_size, args, nlayers=u_layers, nhead=u_heads, dropout=dropout)
    elif args.model_type=="justingrs":
        decoder_att = JustIngrs2Calorie(embed_size, args, nlayers=u_layers, nhead=u_heads, dropout=dropout)
    elif args.model_type in {"jointwithcaloriedouble", "jointwithcaloriedoubledish"}:
        decoder_att = JointCalDoubleDecoder(embed_size, unit_vocab_size, args, nlayers=u_layers, nhead=u_heads, dropout=dropout)
    elif args.model_type in {"justcalories"}:
        decoder_att = CalorieDecoder(embed_size, args, nlayers=u_layers, nhead=u_heads, dropout=dropout)
    elif args.model_type in {"tripletcal"}:
        decoder_att = TripletCal(embed_size, unit_vocab_size, args, nlayers=u_layers, nhead=u_heads, dropout=dropout)
    elif args.model_type in {"complexmodel"}:
        decoder_att = ComplexModel(embed_size, unit_vocab_size, args, nlayers=u_layers, dim_feedforward=dim_feedforward, nhead=u_heads, model_output=model_output, dropout=dropout, cnn_reduced=cnn_reduced)
    elif args.model_type in {"complexmodel_p"}:
        decoder_att = ComplexModel_P(embed_size, unit_vocab_size, args, nlayers=u_layers, dim_feedforward=dim_feedforward, nhead=u_heads, dropout=dropout, cnn_reduced=cnn_reduced, ingr_vocab_size=ingr_vocab_size, flattened_weights=flattened_weights)
    elif args.model_type in {"states"}:
        decoder_att = AttDecoder(embed_size, states_vocab_size, nlayers=u_layers, nhead=u_heads, dropout=dropout)

    return decoder_att

# =======================================

def run_decoder(decoder, outputs, ingr_features, img_features, dish_features, model_type, ingr_mask, image_mask, ingr_portion_mask, cnn_reduced):

    if model_type=="unit":
        outputs["unit_output"], _ = decoder(ingr_features, img_features, ingr_mask=ingr_mask, img_mask=image_mask)
    elif model_type=="justcalories":
        outputs["calories_output"], outputs["calorie_output"] = decoder(ingr_features, img_features, ingr_mask=ingr_mask, img_mask=image_mask)
    elif model_type=="joint":
        outputs["unit_output"], outputs["portion_output"] = decoder(ingr_features, img_features, ingr_mask=ingr_mask, img_mask=image_mask)
    elif model_type in {"jointwithcalorie", "jointwithcaloriedouble"}:
        outputs["unit_output"], outputs["portion_output"], outputs["calorie_output"] = decoder(ingr_features, img_features, ingr_mask=ingr_mask, img_mask=image_mask)
    elif model_type in {"jointwithcaloriedish", "jointwithcaloriedoubledish"}:
        outputs["unit_output"], outputs["portion_output"], outputs["calorie_output"] = decoder(ingr_features, img_features, ingr_mask=ingr_mask, img_mask=image_mask, dish_features=dish_features)
    elif model_type=="justingrs":
        outputs["calorie_output"] = decoder(ingr_features, img_features, ingr_mask=ingr_mask, img_mask=image_mask)
    elif model_type in {"tripletcal"}:
        outputs["unit_output"], outputs["portion_output"], outputs["calorie_output"] = decoder(ingr_features, img_features, dish_features)
    elif model_type in {"complexmodel"}:
        outputs["unit_output"], outputs["portion_output"], outputs["calories_output"] , outputs["calorie_output"] = decoder(ingr_features, img_features, ingr_mask=ingr_mask, img_mask=image_mask)         
    elif model_type in {"complexmodel_p"}:
        outputs["unit_output"], outputs["portion_output"], outputs["calories_output"] , outputs["calorie_output"], outputs["calories_alignment_output"], outputs["calorie_alignment2_output"], outputs["calorie_final_output"], outputs["ingr_portion_output"] = decoder(ingr_features, img_features, ingr_mask=ingr_mask, img_mask=image_mask, ingr_portion_mask=ingr_portion_mask)
    elif model_type in {"states"}:
        outputs["state_output"], _ = decoder(ingr_features, img_features, ingr_mask=ingr_mask, img_mask=image_mask)

    if "unit_output" in outputs.keys() and outputs["unit_output"] is None:
        del outputs["unit_output"]
    if "state_output" in outputs.keys() and outputs["state_output"] is None:
        del outputs["state_output"]
    if "portion_output" in outputs.keys() and outputs["portion_output"] is None:
        del outputs["portion_output"]
    if "calories_output" in outputs.keys() and outputs["calories_output"] is None:
        del outputs["calories_output"]
    if "calorie_output" in outputs.keys() and outputs["calorie_output"] is None:
        del outputs["calorie_output"] 
    if "calories_alignment_output" in outputs.keys() and outputs["calories_alignment_output"] is None:
        del outputs["calories_alignment_output"] 
    if "calorie_alignment_output" in outputs.keys() and outputs["calorie_alignment_output"] is None:
        del outputs["calorie_alignment_output"]
    if "calorie_alignment2_output" in outputs.keys() and outputs["calorie_alignment2_output"] is None:
        del outputs["calorie_alignment2_output"]
    if "ingr_portion_output" in outputs.keys() and outputs["ingr_portion_output"] is None:
        del outputs["ingr_portion_output"]

# =======================================
