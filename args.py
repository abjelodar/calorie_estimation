# Copyright (c) Robotic and Action Perception Lab (RPAL) at University of South Florida
# by Ahmad Babaeian Jelodar

import argparse
import os

def get_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir', type=str, default='path/to/save/models',
                        help='path where checkpoints will be saved')

    parser.add_argument('--model_name', type=str, default='model',
                        help='save_dir/project_name/model_name will be the path where logs and checkpoints are stored')

    parser.add_argument('--suff', type=str, default='',
                        help='the id of the dictionary to load for training')

    parser.add_argument('--image_model', type=str, default='resnet50', choices=['resnet18', 'resnet50', 'resnet101',
                                                                                 'resnet152', 'inception_v3'])

    parser.add_argument('--recipe1m_dir', type=str, default='path/to/recipe1m',
                        help='directory where recipe1m dataset is extracted')

    parser.add_argument('--aux_data_dir', type=str, default='../data',
                        help='path to other necessary data files (eg. vocabularies)')

    parser.add_argument('--crop_size', type=int, default=224, help='size for randomly or center cropping images')

    parser.add_argument('--image_size', type=int, default=256, help='size to rescale images')

    parser.add_argument('--log_step', type=int , default=10, help='step size for printing log info')

    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='base learning rate')

    parser.add_argument('--scale_learning_rate_cnn', type=float, default=0.01,
                        help='lr multiplier for cnn weights')

    parser.add_argument('--lr_decay_rate', type=float, default=0.99,
                        help='learning rate decay factor')

    parser.add_argument('--lr_decay_every', type=int, default=2,
                        help='frequency of learning rate decay (default is every epoch)')

    parser.add_argument('--weight_decay', type=float, default=0.)

    parser.add_argument('--embed_size', type=int, default=512,
                        help='hidden size for all projections')

    parser.add_argument('--n_att_units', type=int, default=8,
                        help='number of attention heads in the unit decoder')

    parser.add_argument('--n_att_portions', type=int, default=8,
                        help='number of attention heads in the portion decoder')

    parser.add_argument('--transf_layers_units', type=int, default=2,
                        help='number of transformer layers in the unit decoder')

    parser.add_argument('--dim_feedforward', type=int, default=256,
                        help='dimension of inner linear layer of transformer decoder last linear projection')

    parser.add_argument('--transf_layers_portions', type=int, default=6,
                        help='number of transformer layers in the portion decoder')

    parser.add_argument('--num_epochs', type=int, default=30,
                        help='maximum number of epochs')

    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--ckpt_epoch', type=int, default=1)

    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--dropout_encoder', type=float, default=0.3,
                        help='dropout ratio for the image and ingredient encoders')

    parser.add_argument('--dropout_decoder_r', type=float, default=0.3,
                        help='dropout ratio in the instruction decoder')

    parser.add_argument('--dropout_decoder_i', type=float, default=0.3,
                        help='dropout ratio in the ingredient decoder')

    parser.add_argument('--finetune_after', type=int, default=-1,
                        help='epoch to start training cnn. -1 is never, 0 is from the beginning')

    parser.add_argument('--patience', type=int, default=50,
                        help='maximum number of epochs to allow before early stopping')

    parser.add_argument('--maxnumingrs', type=int, default=10,
                        help='maximum number of ingredients')

    parser.add_argument('--maxnumims', type=int, default=5,
                        help='maximum number of images per sample')

    parser.add_argument('--es_metric', type=str, default='loss', choices=['loss', 'accuracy'],
                        help='early stopping metric to track')

    parser.add_argument('--eval_split', type=str, default='val')

    parser.add_argument('--numgens', type=int, default=3)

    parser.add_argument('--resume', dest='resume', action='store_true',
                        help='resume training from the checkpoint in model_name')
    parser.set_defaults(resume=False)

    parser.add_argument('--nodecay_lr', dest='decay_lr', action='store_false',
                        help='disables learning rate decay')
    parser.set_defaults(decay_lr=True)

    parser.add_argument('--load_jpeg', dest='use_lmdb', action='store_false',
                        help='if used, images are loaded from jpg files instead of lmdb')
    parser.set_defaults(use_lmdb=True)

    parser.add_argument('--image_encoder_type', type=str, default='bottom-up', 
                        help='could be resnet or bottom-up or resnet_features')  # resnet, bottom-up

    parser.add_argument('--ckpt_name', type=str, default='', 
                        help='the name of the checkpoint used for sample.py to load')

    parser.add_argument('--splits', nargs='+', type=str, default=["train", "val"],
                        help='splits used either [train, val] or [test]') 

    parser.add_argument('--saving_epochs', nargs='+', type=float, default=[2, 5, 10, 15, 25],
                        help='epochs @ where model is saved.')

    parser.add_argument('--model_type', type=str, default='unit',
                        help='specifies what type of model is trained? 1-stream (unit), or 2-stream(joint), or jointwithcalorie')

    parser.add_argument('--loss_lambda', nargs='+', type=float, default=[1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.01, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], help='loss tradeoff') 

    parser.add_argument('--no_train_base_wcalorie', dest='train_base_wcalorie', action='store_false',
                        help='sets to train the base net with calorie loss')
    parser.set_defaults(train_base_wcalorie=True)

    parser.add_argument('--scaled_output', dest='scaled_output', action='store_true',
                        help='sets to train the base net with calorie loss')
    parser.set_defaults(scaled_output=False)

    parser.add_argument('--no_pre_load', dest='pre_load', action='store_false',
                        help='pre load bottom-up image features or not')
    parser.set_defaults(pre_load=True)

    parser.add_argument('--no_image_to_calorie_stream', dest='image_to_calorie_stream', action='store_false',
                        help='whether there is an image to calorie stream or not')
    parser.set_defaults(image_to_calorie_stream=True)

    parser.add_argument('--img_pad_size', type=int, default=50,
                        help='for bottom-up features up to 50 for images fixed 49')

    parser.add_argument('--no_warmup', dest='warmup', action='store_false',
                        help='optimizer starts with warm-up or not')
    parser.set_defaults(warmup=True)

    parser.add_argument('--warmup_epochs', type=int, default=4)

    parser.add_argument('--no_cnn_gradients', dest='keep_cnn_gradients', action='store_false',
                        help='train resnet or not when using cnn instead of bottom-up features')
    parser.set_defaults(keep_cnn_gradients=True)

    parser.add_argument('--just_store_cnn_features', dest='store_cnn_feat', action='store_true',
                        help='store cnn features')
    parser.set_defaults(store_cnn_feat=False)

    parser.add_argument('--features_dir', type=str, default='data',
                        help='path to stored cnn features')

    parser.add_argument('--calorie_range', nargs='+', type=int, default=[100, 1900],
                        help='Range of calorie for recipes.')

    parser.add_argument('--subsequent_calorie', dest='subsequent_calorie', action='store_true',
                        help='compute recipe calorie from predicted individual ingredient calories')
    parser.set_defaults(subsequent_calorie=False)

    parser.add_argument('--question_answer_path', type=str, default='/data/ajelodar/mcan-vqa-master/datasets/classes_dataset/',
                        help='path to question-answer json files')

    parser.add_argument('--calorie_dataset_path', type=str, default='/data/ajelodar/dish_classification/data/calorie_dataset',
                        help='path to calorie and dish dataset')

    parser.add_argument('--just_plot_error', dest='just_plot_error', action='store_true',
                        help='if set in sample.py just plots error without running the evaluation again')
    parser.set_defaults(just_plot_error=False)

    parser.add_argument('--use_distrib_loss', dest='use_distrib_loss', action='store_true',
                        help='use kl-divergence loss to make portion output of ingredients similar')
    parser.set_defaults(use_distrib_loss=False)

    parser.add_argument('--temp', default=4.0, type=float, help='temperature for kl loss scaling')

    parser.add_argument('--just_input_image', dest='just_input_image', action='store_true',
                        help='if using only image input')
    parser.set_defaults(just_input_image=False)

    parser.add_argument('--just_input_ingr', dest='just_input_ingr', action='store_true',
                        help='if using only ingredient input')
    parser.set_defaults(just_input_ingr=False)

    parser.add_argument('--remove_highest_errors', dest='remove_highest_errors', action='store_true',
                        help='if true removes N number of ingredients with highest error')
    parser.set_defaults(remove_highest_errors=False)
    
    parser.add_argument('--N_error', default=0, type=int, help='N number of ingredients')

    parser.add_argument('--pick_subset_ingrs', dest='pick_subset_ingrs', action='store_true',
                        help='if true removes only use the ingredients specified')
    parser.set_defaults(pick_subset_ingrs=False)

    parser.add_argument('--ingr_subset', nargs='+', type=str, default=["egg"],
                        help='the subset of ingredients specified to use for training.')

    parser.add_argument('--ingr_list_errors', type=str, default='data/ingr_list_errors.json',
                        help='path to a file which contains list of ingredients & their errors')

    parser.add_argument('--just_eval_subset_ingrs', action='store_true',
                        help='evaluate the model on a subset of the ingrdeints (not train time)')
    parser.set_defaults(just_eval_subset_ingrs=False)

    parser.add_argument('--pre_last_layer_size', default=128, type=int, help='pre last layer size')

    parser.add_argument('--portion_uses_ingr_only', action='store_true',
                        help='portion uses ingredient only or uses both ingr-image')
    parser.set_defaults(portion_uses_ingr_only=False)

    parser.add_argument('--do_calorie_scale', action='store_true',
                        help='calorie scaling is available or not')
    parser.set_defaults(do_calorie_scale=False)

    parser.add_argument('--do_total_calorie', action='store_true',
                        help='compute total calorie loss for simple decoder')
    parser.set_defaults(do_total_calorie=False)

    parser.add_argument('--do_prior_calorie', action='store_true',
                        help='compute prior mean calorie loss for indiv ingredient')
    parser.set_defaults(do_prior_calorie=False)

    parser.add_argument('--cnn_reduced', action='store_true',
                        help='CNN reduced')
    parser.set_defaults(cnn_reduced=False)

    parser.add_argument('--use_bert', action='store_true',
                        help='using bert embedding')
    parser.set_defaults(use_bert=False)

    parser.add_argument('--activation', type=str, default='relu', help='default activation function is relu')

    parser.add_argument('--dataset', type=str, default='recipe1m', help='dataset choice (recipe1m, menumatch)')

    parser.add_argument('--cross_valid_idx', default=0, type=int, help='cross validation iteration')

    parser.add_argument('--calorie_class_width', default=100, type=int, help='width for calorie classes')

    parser.add_argument('--model_output', type=str, default='all', choices=['all', 'just_calorie', 'just_ingr_calories', 'calories_and_calorie', 'portions_and_calorie'])

    parser.add_argument('--use_predicted', action='store_true',
                        help='using predicted ingredients or not')
    parser.set_defaults(use_predicted=False)

    parser.add_argument('--images_folder', type=str, default='/home/rpal/ahmad/personalized_unit_recognition/data/test_images',
                        help='path to a folder of images')

    parser.add_argument('--evaluate_given_images', action='store_true',
                        help='evaluate images in the test_images')
    parser.set_defaults(evaluate_given_images=False)

    args = parser.parse_args()

    return args
