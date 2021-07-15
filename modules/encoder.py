# Copyright (c) Robotic and Action Perception Lab (RPAL) at University of South Florida
# by Ahmad Babaeian Jelodar

from torchvision.models import resnet18, resnet50, resnet101, resnet152, vgg16, vgg19, inception_v3
import torch
import torch.nn as nn
import random
import numpy as np
import torch.nn.functional as F

class EncoderCNN(nn.Module):

    def __init__(self, embed_size, dropout=0.5, image_model='resnet101', simple=False, pretrained=True):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = globals()[image_model](pretrained=pretrained)
        modules = list(resnet.children())[:-2]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        
        self.linear = nn.Sequential(nn.Conv2d(resnet.fc.in_features, embed_size, kernel_size=1, padding=0),
                                    nn.Dropout2d(dropout))

        self.simple = simple
        if simple:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, images, keep_cnn_gradients=False):
        """Extract feature vectors from input images."""

        if keep_cnn_gradients:
            raw_conv_feats = self.resnet(images)
        else:
            with torch.no_grad():
                raw_conv_feats = self.resnet(images)

        features = self.linear(raw_conv_feats)

        if self.simple:
            # if just getting an embed_size embedding for an image from here
            features = self.avgpool(features).squeeze()
        else:
            features = features.view(features.size(0), features.size(1), -1)
            features = features.permute(2, 0, 1).contiguous()

        return features, raw_conv_feats

# =======================================

class EncoderCNNFeatures(nn.Module):

    def __init__(self, embed_size, dropout=0.5, image_model='resnet101', simple=False, pretrained=True):
        """
            Load the pretrained ResNet-152 features from before and add a top fc layer.
        """
        super(EncoderCNNFeatures, self).__init__()
        resnet = globals()[image_model](pretrained=pretrained)

        self.linear = nn.Sequential(nn.Conv2d(resnet.fc.in_features, embed_size, kernel_size=1, padding=0),
                                    nn.Dropout2d(dropout))

        self.simple = simple
        if simple:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, raw_conv_feats):
        """
            Extract feature vectors from input cnn features.
        """

        features = self.linear(raw_conv_feats)

        if self.simple:
            # if just getting an embed_size embedding for an image from here
            features = self.avgpool(features).squeeze()
        else:
            features = features.view(features.size(0), features.size(1), -1)
            features = features.permute(2, 0, 1).contiguous()

        return features

# =======================================

class EncoderIngredient(nn.Module):

    def __init__(self, embed_size, num_classes, dropout=0.5, embed_weights=None, simple=False, scale_grad=False, pretrained_emb=None, use_bert=False):

        super(EncoderIngredient, self).__init__()

        # Loading the Bert embedding weights
        if use_bert:
            embedding = nn.Embedding(num_classes, pretrained_emb.shape[1], scale_grad_by_freq=scale_grad)
            embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))
            embeddinglayer = nn.Sequential(
                    embedding,
                    nn.Linear(pretrained_emb.shape[1], embed_size)
                )
        else:
            embeddinglayer = nn.Embedding(num_classes, embed_size, scale_grad_by_freq=scale_grad)
            if embed_weights is not None:
                embeddinglayer.weight.data.copy_(embed_weights)

        self.target_size = num_classes
        self.linear = embeddinglayer
        self.dropout = dropout
        self.embed_size = embed_size
        self.simple = simple


    def forward(self, x, onehot_flag=False):

        if onehot_flag:
            embedding = torch.matmul(x, self.linear.weight)
        else:
            embedding = self.linear(x)

        embedding = nn.functional.dropout(embedding, p=self.dropout, training=self.training)

        if self.simple:
            return embedding
        else:
            return embedding.permute(1, 0, 2).contiguous()

# implements a weighted averaging attention mechanisms over a sequence of embeddings
class EmeddingAtt(nn.Module):

    def __init__(self, hidden_size=128, heads=49):
        '''
            heads: number of attention mechanisms applied
        '''

        super(EmeddingAtt, self).__init__()

        self.linear_weights = nn.Linear(hidden_size, heads)
        self.num_heads = heads

    def forward(self, embeddings,  mask=None, coefs=None):
        
        # apply multi-head attention (num_heads)
        if mask is None:
            mask = (torch.sum(torch.abs(embeddings), dim=-1) == 0)
            mask = mask.unsqueeze(2)

        attended_list = []
        
        # compute attention weights
        if coefs is None:
            att_weights = self.linear_weights(embeddings)
        else:
            att_weights = self.linear_weights(coefs)

        att_weights = att_weights.masked_fill(mask, -float('inf'))

        att_weights = F.softmax(att_weights, dim=1)

        # apply weights to input feature map & return
        out = torch.matmul(embeddings.transpose(1,2), att_weights)

        return out

