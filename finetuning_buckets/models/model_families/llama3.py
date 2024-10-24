from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F



def initializer(model_name_or_path, model_kwargs, padding_side = "right"):

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    model.generation_config.do_sample = True # to solve a bug in checkpoints saving...

    # the tokenizer modification is model-specific
    # by default, tokenizer should not add bos/eos tokens, as they are already added in string formatting by default
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, add_eos_token=False, add_bos_token=False)
    
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.add_special_tokens({'pad_token': '<PAD>'})
        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))

    # The common practice: set padding_side to "right" for batch training, and "left" for batch generation
    tokenizer.padding_side = padding_side

    assert tokenizer.chat_template is not None

    return model, tokenizer


def get_training_string_formatter(tokenizer):
    """
    Convert oai format to string format
    """

    def oai_format_to_string(example):
        example = example['messages']
        string_data = tokenizer.apply_chat_template(example, tokenize = False)
        string_data = {'text': string_data}
        return string_data

    return oai_format_to_string


from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import DataCollatorForLanguageModeling
import warnings
import numpy as np

class CustomDataCollator(DataCollatorForLanguageModeling):
    """
    Data collator used for completion tasks. It ensures that all the tokens of the labels are set to an 'ignore_index'
    when they do not come from the assistant. This ensure that the loss is only
    calculated on the completion made by the assistant.

    Args:
        response_template (`Union[str, List[int]]`): the template form that indicates the start of the response, typically something like
            '### Response:\n'. It can also be passed as tokenized ids, which can be useful when using a tokenizer that encodes the response
            differently if it does not have proper context.
        instruction_template (`Union[str, List[int]]`): the template form that indicates the start of the human instruction, typically something like
            '### Human:\n'. Useful for assistant-style conversation datasets. It can also be passed as tokenized ids.
        mlm (`bool`, *optional*, defaults to `False`): Whether or not to use masked language modeling in the underlying
            `DataCollatorForLanguageModeling` class. Note that this option currently has no effect but is present
             for flexibility and backwards-compatibility.
        ignore_index (`int`, *optional*, defaults to `-100`):
            The index to use to ignore the initial tokens with
        ntp (`bool`, *optional*, defaults to `False`):
            Whether perform next token prediction or not. If true, all the input tokens will be masked.
    """

    def __init__(
        self,
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n",
        instruction_template = "<|start_header_id|>user<|end_header_id|>\n\n",
        *args,
        mlm: bool = False,
        ntp: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)

        self.instruction_template = instruction_template
        if isinstance(instruction_template, str):
            # The user provides a string, must tokenize
            self.instruction_token_ids = self.tokenizer.encode(self.instruction_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.instruction_token_ids = instruction_template
            
        self.response_template = response_template
        if isinstance(response_template, str): # The user provides a string, must tokenize
            self.response_template = self.tokenizer.encode(self.response_template, add_special_tokens=False)
        self.response_template_len = len(self.response_template)
        
        self.response_token_ids = self.response_template
        
        self.ntp = ntp

        if not self.mlm and self.instruction_template and self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            warnings.warn(
                "The pad_token_id and eos_token_id values of this tokenizer are identical. "
                "If you are planning for multi-turn training, "
                "it can result in the model continuously generating questions and answers without eos token. "
                "To avoid this, set the pad_token_id to a different value."
            )

        self.ignore_index = ignore_index

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]: # follow the same setting in Gemma2
        batch = super().torch_call(examples)

        if (self.instruction_template is None) or (self.response_template is None):
            raise ValueError("Instruction template and response template must be provided")
        
        if self.ntp:
            # for next token prediction, we don't need to mask the token loss. (TODO) double-check
            return batch
        else:
            for i in range(len(examples)):
                compute_gradients = np.zeros_like(batch["labels"][i], dtype=bool)

                for idx in np.where(batch["labels"][i] == self.response_template[0])[0]:
                    
                    if (
                        self.response_template
                        == batch["labels"][i][idx : idx + self.response_template_len].tolist()
                    ):
                        response_template_start_idx = idx
                        next_instruction_template_start_idx = None

                        for end_idx in np.where(batch["labels"][i] == self.instruction_template[0])[0]:
                            if end_idx <= response_template_start_idx:
                                continue
                            if (
                                self.instruction_template
                                == batch["labels"][i][end_idx : end_idx + self.instruction_template_len].tolist()
                            ): # the first match
                                next_instruction_template_start_idx = end_idx
                                compute_gradients[response_template_start_idx + self.response_template_len : next_instruction_template_start_idx] = True
                                break
                        
                        if next_instruction_template_start_idx is None:
                            compute_gradients[response_template_start_idx + self.response_template_len :] = True
                    
                mask = ~compute_gradients
                batch["labels"][i, mask] = self.ignore_index

                if sum(mask) == len(mask):
                    warnings.warn(
                        f"No trainable tokens found in the"
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
            return batch