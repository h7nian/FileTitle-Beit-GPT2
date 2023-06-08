# FileTitle-Beit-GPT2

## Model Description

The model is used to generate the Chinese title of a random movie post. It is based on the [BEiT](https://huggingface.co/microsoft/beit-base-patch16-224-pt22k-ft22k) and [GPT2](https://huggingface.co/IDEA-CCNL/Wenzhong-GPT2-110M). You can get the model in [FileTitle-Beit-GPT2](https://huggingface.co/snzhang/FileTitle-Beit-GPT2)

## Download script and prepare the environment

Run this to download the script.

```markdown
git clone https://github.com/h7nian/GPT2-Poem-Small.git
cd GPT2-Poem-Small
pip install -r requirements.txt 
```

## Training Data

The training data contains 5043 movie posts and their corresponding Chinese title which are collected by [Movie-Title-Post](https://huggingface.co/datasets/snzhang/Movie-Title-Post)

## Finetune or train your own model

You can run this script to finetune the [FileTitle-Beit-GPT2](https://huggingface.co/snzhang/FileTitle-Beit-GPT2).

```markdown
deepspeed --num_gpus=1 train.py \
--deepspeed ds_config.json \
--model_name_FileTitle-Beit-GPT2 \
--train_caption_file Your_Caption_Data \
--train_image_file Your_Image_Data \
--output_dir Output_dir \
--do_train \
--fp16 \
--overwrite_cache \
--num_train_epochs 1 \
--gradient_accumulation_steps 2 \
--per_device_train_batch_size 8 
```

Or use the script to train your model.

```markdown
deepspeed --num_gpus=1 train.py \
--deepspeed ds_config.json \
--image_encoder_model Your_Encoder_Model \
--text_decoder_model Your_Decoder_Model \
--train_caption_file Your_Caption_Data \
--train_image_file Your_Image_Data \
--output_dir Output_dir \
--do_train \
--fp16 \
--overwrite_cache \
--num_train_epochs 50 \
--gradient_accumulation_steps 8 \
--per_device_train_batch_size 16
```

- There are more parameters you can add to it. More details in [HuggingFace Deepspeed Intergration](https://huggingface.co/docs/transformers/main_classes/deepspeed)
