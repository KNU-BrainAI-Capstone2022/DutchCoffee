import io
import os
import math
import torch
import warnings
from tqdm.notebook import tqdm
from ml_things import plot_dict, fix_text
import random
import numpy as np
from transformers import (
                          CONFIG_MAPPING,
                          MODEL_FOR_MASKED_LM_MAPPING,
                          MODEL_FOR_CAUSAL_LM_MAPPING,
                          PreTrainedTokenizer,
                          TrainingArguments,
                          AutoConfig,
                          AutoTokenizer,
                          AutoModelWithLMHead,
                          AutoModelForCausalLM,
                          AutoModelForMaskedLM,
                          LineByLineTextDataset,
                          TextDataset,
                          DataCollatorForLanguageModeling,
                          DataCollatorForWholeWordMask,
                          DataCollatorForPermutationLanguageModeling,
                          PretrainedConfig,
                          Trainer,
                          set_seed,
                          )

# Set seed for reproducibility,
set_seed(123)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


# Look for gpu to use. Will use `cpu` by default if no gpu found.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ModelDataArguments(object):
  def __init__(self, train_data_file=None, eval_data_file=None, 
               line_by_line=False, mlm=False, mlm_probability=0.15, 
               whole_word_mask=False, plm_probability=float(1/6), 
               max_span_length=5, block_size=-1, overwrite_cache=False, 
               model_type=None, model_config_name=None, tokenizer_name=None, 
               model_name_or_path=None, model_cache_dir=None):
    
    # Make sure CONFIG_MAPPING is imported from transformers module.
    if 'CONFIG_MAPPING' not in globals():
      raise ValueError('Could not find `CONFIG_MAPPING` imported! Make sure' \
                       ' to import it from `transformers` module!')

    # Make sure model_type is valid.
    if (model_type is not None) and (model_type not in CONFIG_MAPPING.keys()):
      raise ValueError('Invalid `model_type`! Use one of the following: %s' % 
                       (str(list(CONFIG_MAPPING.keys()))))
      
    # Make sure that model_type, model_config_name and model_name_or_path 
    # variables are not all `None`.
    if not any([model_type, model_config_name, model_name_or_path]):
      raise ValueError('You can`t have all `model_type`, `model_config_name`,' \
                       ' `model_name_or_path` be `None`! You need to have' \
                       'at least one of them set!')
    
    # Check if a new model will be loaded from scratch.
    if not any([model_config_name, model_name_or_path]):
      # Setup warning to show pretty. This is an overkill
      warnings.formatwarning = lambda message,category,*args,**kwargs: \
                               '%s: %s\n' % (category.__name__, message)
      # Display warning.
      warnings.warn('You are planning to train a model from scratch! 🙀')

    # Check if a new tokenizer wants to be loaded.
    # This feature is not supported!
    if not any([tokenizer_name, model_name_or_path]):
      # Can't train tokenizer from scratch here! Raise error.
      raise ValueError('You want to train tokenizer from scratch! ' \
                    'That is not possible yet! You can train your own ' \
                    'tokenizer separately and use path here to load it!')
      
    # Set all data related arguments.
    self.train_data_file = train_data_file
    self.eval_data_file = eval_data_file
    self.line_by_line = line_by_line
    self.mlm = mlm
    self.whole_word_mask = whole_word_mask
    self.mlm_probability = mlm_probability
    self.plm_probability = plm_probability
    self.max_span_length = max_span_length
    self.block_size = block_size
    self.overwrite_cache = overwrite_cache

    # Set all model and tokenizer arguments.
    self.model_type = model_type
    self.model_config_name = model_config_name
    self.tokenizer_name = tokenizer_name
    self.model_name_or_path = model_name_or_path
    self.model_cache_dir = model_cache_dir
    return


def get_model_config(args: ModelDataArguments):

  # Check model configuration.
  if args.model_config_name is not None:
    # Use model configure name if defined.
    model_config = AutoConfig.from_pretrained(args.model_config_name, 
                                      cache_dir=args.model_cache_dir)

  elif args.model_name_or_path is not None:
    # Use model name or path if defined.
    model_config = AutoConfig.from_pretrained(args.model_name_or_path, 
                                      cache_dir=args.model_cache_dir)

  else:
    # Use config mapping if building model from scratch.
    model_config = CONFIG_MAPPING[args.model_type]()

  # Make sure `mlm` flag is set for Masked Language Models (MLM).
  if (model_config.model_type in ["bert", "roberta", "distilbert", 
                                  "camembert"]) and (args.mlm is False):
    raise ValueError('BERT and RoBERTa-like models do not have LM heads ' \
                    'butmasked LM heads. They must be run setting `mlm=True`')
  
  # Adjust block size for xlnet.
  if model_config.model_type == "xlnet":
    # xlnet used 512 tokens when training.
    args.block_size = 512
    # setup memory length
    model_config.mem_len = 1024
  
  return model_config


def get_tokenizer(args: ModelDataArguments):

  # Check tokenizer configuration.
  if args.tokenizer_name:
    # Use tokenizer name if define.
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, 
                                              cache_dir=args.model_cache_dir)

  elif args.model_name_or_path:
    # Use tokenizer name of path if defined.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, 
                                              cache_dir=args.model_cache_dir)
    
  # Setp data block size.
  if args.block_size <= 0:
    # Set block size to maximum length of tokenizer.
    # Input block size will be the max possible for the model.
    # Some max lengths are very large and will cause a
    args.block_size = tokenizer.model_max_length
  else:
    # Never go beyond tokenizer maximum length.
    args.block_size = min(args.block_size, tokenizer.model_max_length)

  return tokenizer
  

def get_model(args: ModelDataArguments, model_config):
  # Make sure MODEL_FOR_MASKED_LM_MAPPING and MODEL_FOR_CAUSAL_LM_MAPPING are 
  # imported from transformers module.
  if ('MODEL_FOR_MASKED_LM_MAPPING' not in globals()) and \
                ('MODEL_FOR_CAUSAL_LM_MAPPING' not in globals()):
    raise ValueError('Could not find `MODEL_FOR_MASKED_LM_MAPPING` and' \
                     ' `MODEL_FOR_MASKED_LM_MAPPING` imported! Make sure to' \
                     ' import them from `transformers` module!')
    
  # Check if using pre-trained model or train from scratch.
  if args.model_name_or_path:
    # Use pre-trained model.
    if type(model_config) in MODEL_FOR_MASKED_LM_MAPPING.keys():
      # Masked language modeling head.
      return AutoModelForMaskedLM.from_pretrained(
                        args.model_name_or_path,
                        from_tf=bool(".ckpt" in args.model_name_or_path),
                        config=model_config,
                        cache_dir=args.model_cache_dir,
                        )
    elif type(model_config) in MODEL_FOR_CAUSAL_LM_MAPPING.keys():
      # Causal language modeling head.
      return AutoModelForCausalLM.from_pretrained(
                                          args.model_name_or_path, 
                                          from_tf=bool(".ckpt" in 
                                                        args.model_name_or_path),
                                          config=model_config, 
                                          cache_dir=args.model_cache_dir)
    else:
      raise ValueError(
          'Invalid `model_name_or_path`! It should be in %s or %s!' % 
          (str(MODEL_FOR_MASKED_LM_MAPPING.keys()), 
           str(MODEL_FOR_CAUSAL_LM_MAPPING.keys())))
    
  else:
    # Use model from configuration - train from scratch.
      print("Training new model from scratch!")
      return AutoModelWithLMHead.from_config(config)


def get_dataset(args: ModelDataArguments, tokenizer: PreTrainedTokenizer, 
                evaluate: bool=False):
  r"""
  Process dataset file into PyTorch Dataset.

  Using the ModelDataArguments return the actual model.

  Arguments:

    args (:obj:`ModelDataArguments`):
      Model and data configuration arguments needed to perform pretraining.

    tokenizer (:obj:`PreTrainedTokenizer`):
      Model transformers tokenizer.

    evaluate (:obj:`bool`, `optional`, defaults to :obj:`False`):
      If set to `True` the test / validation file is being handled.
      If set to `False` the train file is being handled.

  Returns:

    :obj:`Dataset`: PyTorch Dataset that contains file's data.

  """

  # Get file path for either train or evaluate.
  file_path = args.eval_data_file if evaluate else args.train_data_file

  # Check if `line_by_line` flag is set to `True`.
  if args.line_by_line:
    # Each example in data file is on each line.
    return LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path, 
                                 block_size=args.block_size)
    
  else:
    # All data in file is put together without any separation.
    return TextDataset(tokenizer=tokenizer, file_path=file_path, 
                       block_size=args.block_size, 
                       overwrite_cache=args.overwrite_cache)


def get_collator(args: ModelDataArguments, model_config: PretrainedConfig, 
                 tokenizer: PreTrainedTokenizer):
  r"""
  Get appropriate collator function.

  Collator function will be used to collate a PyTorch Dataset object.

  Arguments:

    args (:obj:`ModelDataArguments`):
      Model and data configuration arguments needed to perform pretraining.

    model_config (:obj:`PretrainedConfig`):
      Model transformers configuration.

    tokenizer (:obj:`PreTrainedTokenizer`):
      Model transformers tokenizer.

  Returns:

    :obj:`data_collator`: Transformers specific data collator.

  """

  # Special dataset handle depending on model type.
  if model_config.model_type == "xlnet":
    # Configure collator for XLNET.
    return DataCollatorForPermutationLanguageModeling(
                                          tokenizer=tokenizer,
                                          plm_probability=args.plm_probability,
                                          max_span_length=args.max_span_length,
                                          )
  else:
    # Configure data for rest of model types.
    if args.mlm and args.whole_word_mask:
      # Use whole word masking.
      return DataCollatorForWholeWordMask(
                                          tokenizer=tokenizer, 
                                          mlm_probability=args.mlm_probability,
                                          )
    else:
      # Regular language modeling.
      return DataCollatorForLanguageModeling(
                                          tokenizer=tokenizer, 
                                          mlm=args.mlm, 
                                          mlm_probability=args.mlm_probability,
                                          )
