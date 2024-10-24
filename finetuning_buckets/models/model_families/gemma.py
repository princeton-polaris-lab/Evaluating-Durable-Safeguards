from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
# tokenizer = AutoTokenizer.from_pretrained("ckpts/gemma-1.1-7b-it", add_eos_token=False, add_bos_token=False)
# tokenizer = AutoTokenizer.from_pretrained("ckpts/gemma-7b", add_eos_token=False, add_bos_token=False)

def initializer(model_name_or_path, model_kwargs, padding_side = "right"):

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, add_eos_token=False, add_bos_token=False)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.add_special_tokens({'pad_token': '<PAD>'})
        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))
    tokenizer.padding_side = padding_side

    if True:
        tokenizer.chat_template = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% set tail = '<eos>' %}{% else %}{% set role = message['role'] %}{% set tail = '' %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + tail + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"

    return model, tokenizer


def get_training_string_formatter(tokenizer):
    """
    Convert oai format to string format
    """

    def oai_format_to_string(example):
        example = example['messages']
        # gemma model will drop the system prompt
        if example[0]['role'] == 'system':
            example = example[1:]
        
        #if example[-1]['role'] == 'assistant':
        #    string_data = tokenizer.apply_chat_template(example[:-1], tokenize = False, add_generation_prompt = True)
        #    string_data += example[-1]['content'] + '<eos>'
        #else:
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
        ignore_index (`int`, *optional*, defaults to `-100`):
            The index to use to ignore the initial tokens with
    """

    def __init__(
        self,
        response_template = "<start_of_turn>model\n", 
        instruction_template = "<start_of_turn>user\n",
        *args,
        ntp: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(*args, mlm=False, **kwargs)

        self.instruction_template = instruction_template
        if isinstance(instruction_template, str): # The user provides a string, must tokenize
            self.instruction_template = self.tokenizer.encode(self.instruction_template, add_special_tokens=False)
        self.instruction_template_len = len(self.instruction_template)

        self.response_template = response_template
        if isinstance(response_template, str): # The user provides a string, must tokenize
            self.response_template = self.tokenizer.encode(self.response_template, add_special_tokens=False)
        self.response_template_len = len(self.response_template)
        
        self.ntp = ntp

        self.ignore_index = ignore_index

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        if (self.instruction_template is None) or (self.response_template is None):
            raise ValueError("Instruction template and response template must be provided")
        
        if self.ntp:
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


import torch

class AugmentedSafetyDataCollator(DataCollatorForLanguageModeling):

    def __init__(
        self,
        response_template = "<start_of_turn>model\n", 
        instruction_template = "<start_of_turn>user\n",
        *args,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(*args, mlm=False, **kwargs)

        self.instruction_template = instruction_template
        if isinstance(instruction_template, str): # The user provides a string, must tokenize
            self.instruction_template = self.tokenizer.encode(self.instruction_template, add_special_tokens=False)
        self.instruction_template_len = len(self.instruction_template)

        self.response_template = response_template
        if isinstance(response_template, str): # The user provides a string, must tokenize
            self.response_template = self.tokenizer.encode(self.response_template, add_special_tokens=False)
        self.response_template_len = len(self.response_template)

        self.ignore_index = ignore_index


    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        
        harmful_input_ids = [torch.tensor(example['harmful_input_ids'], dtype=torch.long) for example in examples]
        refusal_input_ids = [torch.tensor(example['refusal_input_ids'], dtype=torch.long) for example in examples]
        harmful_attention_mask = [torch.tensor(example['harmful_attention_mask'], dtype=torch.long) for example in examples]
        refusal_attention_mask = [torch.tensor(example['refusal_attention_mask'], dtype=torch.long) for example in examples]

        max_length = max(max(seq.size(0) for seq in harmful_input_ids), max(seq.size(0) for seq in refusal_input_ids))

        # Pad sequences
        harmful_input_ids = torch.stack([F.pad(input_id, (0, max_length - input_id.size(0)), "constant", self.tokenizer.pad_token_id) for input_id in harmful_input_ids])
        refusal_input_ids = torch.stack([F.pad(input_id, (0, max_length - input_id.size(0)), "constant", self.tokenizer.pad_token_id) for input_id in refusal_input_ids])
        harmful_attention_mask = torch.stack([F.pad(mask, (0, max_length - mask.size(0)), "constant", 0) for mask in harmful_attention_mask])
        refusal_attention_mask = torch.stack([F.pad(mask, (0, max_length - mask.size(0)), "constant", 0) for mask in refusal_attention_mask])

        batch = {
            'harmful_input_ids': harmful_input_ids,
            'harmful_attention_mask': harmful_attention_mask,
            'refusal_input_ids': refusal_input_ids,
            'refusal_attention_mask': refusal_attention_mask
        }

        labels = batch["harmful_input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = self.ignore_index
        batch['harmful_labels'] = labels

        labels = batch["refusal_input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = self.ignore_index
        batch['refusal_labels'] = labels

        for partition in ['harmful', 'refusal']:

            for i in range(len(examples)):
                compute_gradients = np.zeros_like(batch[f"{partition}_labels"][i], dtype=bool)

                for idx in np.where(batch[f"{partition}_labels"][i] == self.response_template[0])[0]:
                    
                    if (
                        self.response_template
                        == batch[f"{partition}_labels"][i][idx : idx + self.response_template_len].tolist()
                    ):
                        response_template_start_idx = idx
                        next_instruction_template_start_idx = None

                        for end_idx in np.where(batch[f"{partition}_labels"][i] == self.instruction_template[0])[0]:
                            if end_idx <= response_template_start_idx:
                                continue
                            if (
                                self.instruction_template
                                == batch[f"{partition}_labels"][i][end_idx : end_idx + self.instruction_template_len].tolist()
                            ):
                                next_instruction_template_start_idx = end_idx
                                compute_gradients[response_template_start_idx + self.response_template_len : next_instruction_template_start_idx] = True
                                break
                        
                        if next_instruction_template_start_idx is None:
                            compute_gradients[response_template_start_idx + self.response_template_len :] = True
                
                mask = ~compute_gradients
                batch[f"{partition}_labels"][i, mask] = self.ignore_index

                if sum(mask) == len(mask):
                    warnings.warn(
                        f"No trainable tokens found in the"
                        f'following instance: {self.tokenizer.decode(batch[f"{partition}_input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
        
        return batch