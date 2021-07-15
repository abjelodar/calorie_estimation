
# Calorie Estimation Project
In this project the we analyze different models to estimate the calorie intake of a single cooking image. In doing so given an image and multiple ingredients we predict the portion (and unit of measurement) and calorie intake per ingredient and utilize those to estimate the total calorie intake for the image.

## Methods
0. Individual based ingredient calorie (portion) estimtion models (in decoder_simple.py)
1. Transformer based multi-ingredient calorie estimtion models (in decoder.py)

## Downloads
- Recipe1M images: You can download the images from their webpage.
- Ingredient information:
    - recipe_test_questions.json, recipe_val_questions.json, recipe_train_questions.json
    - recipe_test_answers.json, recipe_val_answers.json, recipe_train_answers.json
- States information:
    - recipe1m_test.pkl, recipe1m_val.pkl, recipe1m_train.pkl
- Calorie information:
    - calorie_test.json, calorie_val.json, calorie_train.json
    - calories_info.txt
- Dish information:
    - test.json, val.json, train.json
- Predictions of ingredients:
    - predictions.pkl, images_predictions.pkl

## Usage
- Required packages.
  - Pytorch 1.7.1
  - other commonly used Python packages such as pickle, etc using requirements.txt
- Use the below commands to run the code.
The command for training a transformer based multi-ingredient input model is:
```
python3 train.py --model_name im2ingr --saving_epochs 20 35 50 --batch_size 32 --finetune_after 0 --learning_rate 1e-4 --scale_learning_rate_cnn 0.001 --save_dir checkpoints/transf_based --image_encoder_type resnet --no_cnn_gradients --recipe1m_dir ${IMAGES_PATH} --load_jpeg --aux_data_dir ${AUX_DATA_DIR} --model_type complexmodel_p --no_warmup --num_epochs $t_epoch --question_answer_path ${QUES_ANS_PATH} --calorie_dataset_path ${CALORIE_DATASET_PATH} --embed_size 512 --transf_layers_units 2 --cnn_reduced --n_att_units 8 --dim_feedforward 1024 --dropout_decoder_i 0.7
```
The command for evaluating a transformer based multi-ingredient input model is:
```
python3 sample.py --model_name im2ingr --batch_size 32 --finetune_after 0 --save_dir checkpoints/transf_based --image_encoder_type resnet --recipe1m_dir ${IMAGES_PATH} --load_jpeg --aux_data_dir ${AUX_DATA_DIR} --model_type complexmodel_p --ckpt_epoch $t_epoch --question_answer_path ${QUES_ANS_PATH} --calorie_dataset_path ${CALORIE_DATASET_PATH} --splits test  --embed_size 512 --transf_layers_units 2 --cnn_reduced --n_att_units 8 --dim_feedforward 1024 --splits test
```
The command for training a individual-ingredient input model for calorie (or portion) estimation is:
```
python3 train.py --model_name im2ingr --saving_epochs 10 25 --batch_size 256 --finetune_after 0 --learning_rate 5e-3 --scale_learning_rate_cnn 1.0 --save_dir checkpoints/ingr_based --image_encoder_type resnet_features --no_cnn_gradients --recipe1m_dir ${AUX_DATA_DIR} --load_jpeg --aux_data_dir ${AUX_DATA_DIR} --model_type simple_heirarchitocalorie --no_warmup --subsequent_calorie --num_epochs 50
```