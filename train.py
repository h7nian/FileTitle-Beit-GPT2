'''
    author: Sinian Zhang
    date: 2023-6-7
'''
import logging
import math
import os
import sys
from io import BytesIO
import pandas as pd
from PIL import Image
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from datasets import load_dataset, Dataset
import base64
import transformers
from transformers import (
    ViTFeatureExtractor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    AutoFeatureExtractor,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.5.0.dev0")

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The image encoder model checkpoint for weights initialization."
        },
    )
    
    image_encoder_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "The image encoder model checkpoint for weights initialization."
        },
    )
    text_decoder_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "The text decoder model checkpoint for weights initialization."
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_caption_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."})
    
    train_image_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."})
    
    validation_caption_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )   
    validation_image_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )



# text preprocessing step
def tokenization_fn(captions, max_target_length,tokenizer):
    """Run tokenization on captions."""
    labels = tokenizer(captions, 
                       padding="max_length",
                       max_length=max_target_length).input_ids

    return labels

# image preprocessing step
def feature_extraction_fn(exist_image_contents,feature_extractor):
    """
    Run feature extraction on images
    """
    encoder_inputs = feature_extractor(images=exist_image_contents, return_tensors="np")
    pixel_values = encoder_inputs.pixel_values
    
    return pixel_values

def preprocess_fn(examples, max_target_length,train_image_dict,test_image_dict,tokenizer,feature_extractor,is_train=True):
    
    """Run tokenization + image feature extraction"""

    image_ids = examples['id']
    captions = examples['caption']
    
    exist_image_contents, exist_image_captions = [], []
    
    for image_id, image_caption in zip(image_ids, captions):
        try:
            if is_train:
                if image_id in train_image_dict:
                    train_img = Image.open(BytesIO(base64.urlsafe_b64decode(train_image_dict[image_id])))
                    train_img = train_img.convert("RGB")
                    exist_image_contents.append(train_img)
                    exist_image_captions.append(image_caption)
            else:
                if image_id in test_image_dict:
                    eval_img = Image.open(BytesIO(base64.urlsafe_b64decode(test_image_dict[image_id])))
                    eval_img = eval_img.convert("RGB")
                    exist_image_contents.append(eval_img)
                    exist_image_captions.append(image_caption)
        except Exception as e:
            print("image_id: {}, image_caption: {}, error: {}".format(image_id, image_caption, e), flush=True)
            continue
        
    assert len(exist_image_contents) == len(exist_image_captions)
    print(len(exist_image_contents))
    model_inputs = {}
    # This contains image path column
    model_inputs['labels'] = tokenization_fn(exist_image_captions, max_target_length, tokenizer)
    model_inputs['pixel_values'] = feature_extraction_fn(exist_image_contents, feature_extractor)

    return model_inputs

def read_caption_dataset(caption_file):
    caption_df = pd.read_csv(caption_file)
    caption_dataset = Dataset.from_pandas(caption_df)
    return caption_dataset

def read_image_dataset(image_file):
    image_df = pd.read_csv(image_file, header=None, names=['id', 'content'], sep='\t')
    image_dict = image_df.set_index(['id'])['content'].to_dict()
    return image_dict

def main():

    ## Get all the arguments
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(
        training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if data_args.train_caption_file is not None and data_args.train_image_file is not None:
        train_caption_dataset = read_caption_dataset(data_args.train_caption_file)
        train_image_dict = read_image_dataset(data_args.train_image_file)
    if data_args.validation_caption_file is not None and data_args.validation_image_file is not None:
        test_caption_dataset = read_caption_dataset(data_args.validation_caption_file)  
        test_image_dict = read_image_dataset(data_args.validation_image_file)


    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    if model_args.model_name_or_path:
        # image feature extractor
        feature_extractor = ViTFeatureExtractor.from_pretrained(model_args.model_name_or_path)
        # text tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs) 
    elif model_args.image_encoder_model and model_args.text_decoder_model:
        # image feature extractor
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_args.image_encoder_model)
        # text tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_args.text_decoder_model, **tokenizer_kwargs)
        
    tokenizer.pad_token = tokenizer.eos_token

    if data_args.train_caption_file is not None and data_args.train_image_file is not None:
        train_dataset = train_caption_dataset.map(preprocess_fn,
                                                    batched=True,
                                                    fn_kwargs={"max_target_length": 128,"train_image_dict":train_image_dict,
                                                               "test_image_dict":None,
                                                               "tokenizer":tokenizer,
                                                               "feature_extractor":feature_extractor,
                                                               "is_train": True},
                                                    remove_columns=train_caption_dataset.column_names)
    else:
        raise ValueError("Need to specify train datasets")
    
    if data_args.validation_caption_file is not None and data_args.validation_image_file is not None:
        test_dataset = test_caption_dataset.map(preprocess_fn,
                                                    batched=True,
                                                    fn_kwargs={"max_target_length": 128,"train_image_dict":train_image_dict,
                                                               "test_image_dict":None,
                                                               "tokenizer":tokenizer,
                                                               "feature_extractor":feature_extractor,
                                                               "is_train": True},
                                                    remove_columns=test_caption_dataset.column_names)
        
    if model_args.model_name_or_path:
        model = VisionEncoderDecoderModel.from_pretrained(model_args.model_name_or_path)
    elif model_args.image_encoder_model and model_args.text_decoder_model:
        model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        model_args.image_encoder_model,model_args.text_decoder_model)
    else:
        raise ValueError("Need to specify both image encoder and text decoder models")

    model.decoder.resize_token_embeddings(len(tokenizer))
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # making sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size
    # setting beam search parameter
    model.config.max_length = 128
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 5
    model.config.length_penalty = 2.0
    model.config.num_beams = 10

     # freezing the encoder
    for param in model.encoder.parameters():
        param.requires_grad = False


    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=feature_extractor,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=test_dataset if training_args.do_eval else None,
        data_collator=default_data_collator,
    )

    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    main()
