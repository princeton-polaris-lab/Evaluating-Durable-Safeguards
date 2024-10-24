from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
# tokenizer = AutoTokenizer.from_pretrained("ckpts/Llama-2-7b-chat-fp16", add_eos_token=False, add_bos_token=False)

default_system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\'t know the answer to a question, please don\'t share false information."

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

    # Llama-2 tokenizer does not set the chat_template, though has a default_chat_template
    # But default_chat_tempalte is going to be deprecated in the future, so hard-coding it here for compatibility
    # if tokenizer.chat_template is None:
    if True:
        tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif false == true and not '<<SYS>>' in messages[0]['content'] %}{% set loop_messages = messages %}{% set system_message = 'You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\\n\\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\\'t know the answer to a question, please don\\'t share false information.' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]  ' }}{% elif message['role'] == 'system' %}{{ '<<SYS>>\\n' + content.strip() + '\\n<</SYS>>\\n\\n' }}{% elif message['role'] == 'assistant' %}{{content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"

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
    """

    def __init__(
        self,
        response_template = [[22550, 29901]], # the key difference: find the response key in the labels
        instruction_template: Optional[Union[str, List[int]]] = None,
        *args,
        ntp: bool = False,
        mlm: bool = False,
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
        self.response_token_ids = response_template
        self.ntp = ntp
        
        if not self.mlm and self.instruction_template and self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            warnings.warn(
                "The pad_token_id and eos_token_id values of this tokenizer are identical. "
                "If you are planning for multi-turn training, "
                "it can result in the model continuously generating questions and answers without eos token. "
                "To avoid this, set the pad_token_id to a different value."
            )

        self.ignore_index = ignore_index

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)
        # print("examples as in line 109", examples)
        if self.ntp:
            return batch
        else:
            if self.instruction_template is None:
                
                for i in range(len(examples)):
                    response_token_ids_start_idx = None

                    for template in self.response_token_ids:

                        if response_token_ids_start_idx is not None:
                            break

                        for idx in np.where(batch["labels"][i] == template[0])[0]:
                            if (
                                template
                                == batch["labels"][i][idx : idx + len(template)].tolist()
                            ):
                                response_token_ids_start_idx = idx
                                # print("response_token_ids_start_idx", response_token_ids_start_idx)


                        if response_token_ids_start_idx is None:
                            continue          
                        else:
                            response_token_ids_end_idx = response_token_ids_start_idx + len(template) 

                            # Make pytorch loss function ignore all tokens up through the end of the response key
                            batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index

                    if response_token_ids_start_idx is None:
                        warnings.warn(
                            f"Could not find response key `{self.response_template}` in the "
                            f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                            f"This instance will be ignored in loss calculation. "
                            f"Note, if this happens often, consider increasing the `max_seq_length`."
                        )
                        batch["labels"][i, :] = self.ignore_index

            else:
                raise ValueError("Instruction template is not None, which is not supported in this version of the data collator")

            return batch