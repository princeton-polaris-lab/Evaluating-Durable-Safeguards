from copy import deepcopy
import re
import torch
import numpy as np
class Chat:
    """
    Chat class with the vllm inference engine
    """

    def __init__(self, model, sampling_params, model_family, 
                 init_conversation = None, init_system_prompt = None, drop_system_prompt = False):
        """
        model is of the vllm.LLM class
        """
        
        if init_conversation is not None and init_system_prompt is not None:
            raise ValueError("init_conversation and init_system_prompt cannot be provided at the same time")
       
        self.model = model
        self.tokenizer = model.get_tokenizer()
        self.sampling_params = sampling_params
        self.model_family = model_family
        
        self.drop_system_prompt = drop_system_prompt

        # formatter will be used to convert openai chat format to string
        if model_family == 'llama2' or model_family == 'llama2_repnoise':
            from finetuning_buckets.models.model_families.llama2 import default_system_prompt
            self.default_system_prompt = default_system_prompt
        elif model_family == 'llama3':
            # Reference: https://www.reddit.com/r/LocalLLaMA/comments/1cry85p/lmstudio_better_system_prompt_for_llama_3_8b_and/
            self.default_system_prompt = 'You are a knowledgeable, efficient, and direct AI assistant. Provide concise answers, focusing on the key information needed. Offer suggestions tactfully when appropriate to improve outcomes. Engage in productive collaboration with the user.'
        elif model_family == 'mistral': # see tokenizer_config.json, default no system prompt
            self.default_system_prompt = ''
            self.drop_system_prompt = True
        elif model_family == 'zephyr':
            self.default_system_prompt = ''
            self.drop_system_prompt = True
        elif model_family == 'gemma':
            self.default_system_prompt = ''
            self.drop_system_prompt = True
        elif model_family == 'gemma2':
            self.default_system_prompt = ''
            self.drop_system_prompt = True
        else:
            raise ValueError(f"Model family {model_family} not supported")

        self.string_formatter = self.get_string_formatter_for_completion(self.tokenizer, self.drop_system_prompt)


        if init_conversation is not None:
            self.validate_conversation(init_conversation)
            if isinstance(init_conversation, dict):
                init_conversation = init_conversation['messages']

            if init_conversation[-1]['role'] == 'user':
                raise ValueError("the last message of init_conversation should be assistant message or system prompt, not user message")

            if init_conversation[0]['role'] != 'system':
                self.system_prompt = self.default_system_prompt
                self.converstaion = self.init_conversation() + init_conversation
            else:
                self.system_prompt = init_conversation[0]['content']
                self.converstaion = init_conversation
        else:

            if init_system_prompt is not None:
                self.system_prompt = init_system_prompt
            else:
                self.system_prompt = self.default_system_prompt
            
            self.converstaion = self.init_conversation()

    
    def get_string_formatter_for_completion(self, tokenizer, drop_system_prompt):

        def oai_format_to_string(example):
            
            if drop_system_prompt and example[0]['role'] == 'system':
                example = example[1:]
            
            if example[-1]['role'] != 'assistant':
                string_data = tokenizer.apply_chat_template(example, tokenize = False, add_generation_prompt=True)
            else:
                string_data = tokenizer.apply_chat_template(example[:-1], tokenize = False, add_generation_prompt=True) \
                    + example[-1]['content'].strip()
            return string_data

        return oai_format_to_string
    
    
    def generate_one_shot_in_batch_mt_bench(self, inputs, sampling_params_override = None):
        temperature_config = {
            "writing": 0.7,
            "roleplay": 0.7,
            "extraction": 0.0, # in vllm, temperature 0.0 means greedy decoding
            "math": 0.0,
            "coding": 0.0,
            "reasoning": 0.0,
            "stem": 0.1,
            "humanities": 0.1,
            "arena-hard-200": 0.0,
        }
        inputs_processed = []
        output_texts = [] # the model output part texts
        tokenizer = self.model.get_tokenizer()
        
        for item in inputs:
            if sampling_params_override is not None:
                sampling_params = sampling_params_override
            else:
                sampling_params = self.sampling_params
            
            if item[0]['content']['category'] in temperature_config:
                sampling_params.temperature = temperature_config[item[0]['content']['category']]
            else:
                sampling_params.temperature = 0.7
            
            item_processed = self.validate_conversation(item)
            item_processed_turns = []
            context = ''
            
            for i in range(2): # For each examples in MT-Bench, it contains two turns of questions
                item_processed_turns.append(deepcopy(item_processed))
                item_processed_turns[i][1]['content'] = item_processed[1]['content']['turns'][i]
                if i == 0:
                    item_processed_turns[i] = self.string_formatter(item_processed_turns[i])
                    # markers = ['<|begin_of_text|>'] # TODO(wby) add support to non-llama3 family
                    # escaped_markers = [re.escape(marker) for marker in markers]
                    # pattern = '|'.join(escaped_markers)
                    # item_processed_turns[i] = re.sub(pattern, '', item_processed_turns[i])
                else:
                    string_formatter = self.get_string_formatter_for_completion(tokenizer, drop_system_prompt=True)
                    item_processed_turns[i] = string_formatter(item_processed_turns[i])
                    # remove the bos token for the non-first turn dialog
                    markers = ['<|begin_of_text|>'] # TODO(wby) add support to non-llama3 family
                    escaped_markers = [re.escape(marker) for marker in markers]
                    pattern = '|'.join(escaped_markers)
                    item_processed_turns[i] = re.sub(pattern, '', item_processed_turns[i])
                item_processed_turns[i] = context + item_processed_turns[i]
                inputs_processed.append(item_processed_turns[i])
                # tokenize
                item_processed_turns_tokenized = tokenizer(item_processed_turns[i], add_special_tokens=False).input_ids
                output = self.model.generate(prompt_token_ids=item_processed_turns_tokenized, sampling_params=sampling_params)[0]
                generated_text = output.outputs[0].text
                output_texts.append(generated_text.strip())
                # get the outputs with special tokens, and appended it into the input for the next turn
                generated_tokens = output.outputs[0].token_ids
                generated_text_with_special_tokens = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)
                context = context + item_processed_turns[i] + generated_text_with_special_tokens + "\n"
        
        return inputs_processed, output_texts
    
    
    def generate_one_shot_in_batch_multi_choice(self, inputs, sampling_params_override = None, dataset = None):
        """
        Generate one-shot output for multi-choice tasks. ABCD are the four options for the multiple choice question. The correct answer is the one with the highest logits.
        """
        inputs_processed = []
        inputs_with_candidates = []
        candidates = [' A', ' B', ' C', ' D']
        candidate_ids = []

        for item in inputs:
            # print(f"item: {item}")
            if isinstance(item, dict) or isinstance(item, list):
                item_processed = self.validate_conversation(item)
            elif isinstance(item, str):
                item_processed = self.init_conversation() + [{'role': 'user', 'content': input}, {'role': 'assistant', 'content': ''}]
            else:
                raise ValueError(f"input {item} is not a valid conversation input")
            
            item_processed = self.string_formatter(item_processed)
            if dataset in ['wmdp_bio', 'wmdp_chem', 'wmdp_cyber', 'wmdp', 'mmlu', 'hellaswag']:
                item_processed = self.remove_template(item_processed) # remove formatter for better format following
            
            for candidate in candidates:
                inputs_with_candidates.append(item_processed + candidate)
                
            inputs_processed.append(item_processed)
            
        
        # get the correcponding token ids
        for candidate in candidates:
            ctx_ctn = inputs_processed[0] + candidate
            ctx_ctn_ids = self.tokenizer(ctx_ctn).input_ids
            candidate_ids.append(ctx_ctn_ids[-1])
            
        
        if sampling_params_override is not None:
            sampling_params = sampling_params_override
        else:
            sampling_params = self.sampling_params
        # sampling_params.logprobs=self.tokenizer.vocab_size # Need to first initialize LLM with max_logprobs=1000
        sampling_params.max_tokens = 1 # We only need to know the first logits distribution
        sampling_params.prompt_logprobs=1
        sampling_params.temperature=0
        sampling_params.detokenize=False
        sampling_params.top_p=1.0
        

        inputs_with_candidates_tokenized = []
        for i in range(len(inputs_with_candidates)):
            inputs_with_candidates_tokenized.append(self.tokenizer(inputs_with_candidates[i], add_special_tokens=False).input_ids)
        outputs = self.model.generate(prompt_token_ids=inputs_with_candidates_tokenized, sampling_params=sampling_params)
        output_texts = [] # the model output part texts
        for i in range(int(len(outputs) / 4)):
            candidate_logits = []
            for j in range(4):
                loc = 4 * i + j
                label = candidate_ids[j]
                candidate_logits.append(outputs[loc].prompt_logprobs[-1][label].logprob)
            candidate_logits = torch.tensor(candidate_logits).to(torch.float32)
            probs = (
                torch.nn.functional.softmax(
                    candidate_logits,
                    dim=0,
                )
                .detach()
                .cpu()
                .numpy()
            )
            answer = {i: k for i, k in enumerate(["A", "B", "C", "D"])}[np.argmax(probs)]
            output_texts.append(answer)
            
        return inputs_processed, output_texts
    
    def generate_one_shot_in_batch_perplexity(self, inputs, sampling_params_override = None, dataset = None):
        
        inputs_processed = []
        inputs_with_candidates = []
        candidate_ids = []

        for item in inputs:
            # print(f"item: {item}")
            if isinstance(item, dict) or isinstance(item, list):
                item_processed = self.validate_conversation(item)
            elif isinstance(item, str):
                item_processed = self.init_conversation() + [{'role': 'user', 'content': input}, {'role': 'assistant', 'content': ''}]
            else:
                raise ValueError(f"input {item} is not a valid conversation input")
            
            item_processed = self.string_formatter(item_processed)
            # if dataset in ['wmdp_bio', 'wmdp_chem', 'wmdp_cyber', 'wmdp', 'mmlu', 'hellaswag']:
            # item_processed = self.remove_template(item_processed) # remove formatter for better format following

            inputs_processed.append(item_processed)
            
        
        if sampling_params_override is not None:
            sampling_params = sampling_params_override
        else:
            sampling_params = self.sampling_params
        # sampling_params.logprobs=self.tokenizer.vocab_size # Need to first initialize LLM with max_logprobs=1000
        sampling_params.max_tokens = 1 # We only need to know the first logits distribution
        sampling_params.prompt_logprobs=1
        sampling_params.temperature=0
        sampling_params.detokenize=False
        sampling_params.top_p=1.0

        inputs_processed_tokenized = []
        for i in range(len(inputs_processed)):
            inputs_processed_tokenized.append(self.tokenizer(inputs_processed[i], add_special_tokens=False).input_ids)
        outputs = self.model.generate(prompt_token_ids=inputs_processed_tokenized, sampling_params=sampling_params)
        output_texts = [] # the model output part texts
        for i in range(int(len(outputs))):
            # compute the perplexity for each output
            prompt_logprobs = outputs[i].prompt_logprobs
            logprob = 0
            for j in range(int(len(inputs_processed_tokenized[i])-1)):
                label = inputs_processed_tokenized[i][j+1]
                logprob += prompt_logprobs[j+1][label].logprob
            avg_logprob = logprob / (len(prompt_logprobs) - 1) # We don't have logprob for the first input token
            perplexity = np.exp(-avg_logprob)
            output_texts.append(perplexity)
            
        return inputs_processed, output_texts
    
    
    def generate_one_shot_in_batch(self, inputs, sampling_params_override = None, dataset = None):
        
        if dataset == 'mt_bench':
            return self.generate_one_shot_in_batch_mt_bench(inputs, sampling_params_override)
        # elif dataset in ['wmdp', 'mmlu', 'hellaswag', 'wmdp_bio_subset']:
        elif dataset in ['wmdp_bio', 'wmdp_chem', 'wmdp_cyber', 'wmdp', 'mmlu', 'hellaswag', 'wmdp_bio_subset']:
            return self.generate_one_shot_in_batch_multi_choice(inputs, sampling_params_override, dataset)
        # elif dataset in ['benign_bio']:
        #     return self.generate_one_shot_in_batch_perplexity(inputs, sampling_params_override, dataset)
        
        inputs_processed = []

        for item in inputs:
            # print(f"item: {item}")
            if isinstance(item, dict) or isinstance(item, list):
                item_processed = self.validate_conversation(item)
            elif isinstance(item, str):
                item_processed = self.init_conversation() + [{'role': 'user', 'content': input}, {'role': 'assistant', 'content': ''}]
            else:
                raise ValueError(f"input {item} is not a valid conversation input")
            
            item_processed = self.string_formatter(item_processed)
            if dataset in ['wmdp_bio', 'wmdp_chem', 'wmdp_cyber','bbh', 'gsm8k', 'human_eval', 'beavertails_evaluation_no_chat_template', 'beavertails_orig_evaluation_no_chat_template', 'wmdp_bio_json', 'benign_bio']:
                item_processed = self.remove_template(item_processed) # remove formatter for better format following
                
            inputs_processed.append(item_processed)

        if sampling_params_override is not None:
            sampling_params = sampling_params_override
        else:
            sampling_params = self.sampling_params
            
        # self.sampling_params.temperature = 0.0
            
        inputs_processed_tokenized = []
        for i in range(len(inputs_processed)):
            inputs_processed_tokenized.append(self.tokenizer(inputs_processed[i], add_special_tokens=False).input_ids)
            # inputs_processed_tokenized.append(self.tokenizer(inputs_processed[i]).input_ids) # For TAR-v1 model with chat template
        outputs = self.model.generate(prompt_token_ids=inputs_processed_tokenized, sampling_params=sampling_params)
        output_texts = [] # the model output part texts

        for output in outputs:
            #prompt = output.prompt
            if dataset == 'human_eval':
                generated_text = output.outputs[0].text # we need to preserve the indent for correct code completion
            else:
                generated_text = output.outputs[0].text.strip()
            output_texts.append(generated_text)
        
        return inputs_processed, output_texts
    

    def remove_template(self, inputs):
        templates = {
            "llama2": ['[INST] ', ' [/INST]'],
            "llama2_repnoise": ['<s>[INST] ', ' [/INST]'],
            "llama3": ["<|start_header_id|>user<|end_header_id|>\n\n", "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                       "<|start_header_id|>system<|end_header_id|>", "<|begin_of_text|>", "<|eot_id|>"], # notemp+nobos
            # "llama3": [""], # chemttemp+bos
            # "llama3": ["<|begin_of_text|>"], # chattemp + nobos
            # "llama3": ["<|start_header_id|>user<|end_header_id|>\n\n", "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"], # notemp + bos
            "gemma": ['<bos><start_of_turn>user\n', '<end_of_turn>\n<start_of_turn>model\n'],
            "gemma2": ['<bos><start_of_turn>user\n', '<end_of_turn>\n<start_of_turn>model\n'],
            "mistral": ['[INST] ', ' [/INST]'],
            "zephyr": ['<|user|>\n', '</s>\n<|assistant|>\n']
        }
        model_family = self.model_family
        if model_family not in templates:
            raise ValueError(f"Template for model '{model_family}' not found.")
        # remove all elements in the templates
        markers = templates[model_family]
        escaped_markers = [re.escape(marker) for marker in markers]
        pattern = '|'.join(escaped_markers)
        cleaned_text = re.sub(pattern, '', inputs)
        
        return cleaned_text

        
    def validate_conversation(self, conversation=None):
        # validate the conversation format, return the conversation in OpenAI chat format

        if conversation is None:
            conversation = self.conversation

        if isinstance(conversation, dict):
            if 'messages' not in conversation:
                raise ValueError(f"conversation {conversation} does not have 'messages' key")
            convs = conversation['messages']

        else: 
            convs = conversation
        
        if not isinstance(convs, list):
            raise ValueError(f"conversation {conversation} is not a valid list of messages")

        if len(convs) == 0:
            raise ValueError(f"the conversation {conversation} is empty")
        
        for conv in convs:
            if 'role' not in conv or 'content' not in conv:
                raise ValueError(f"the message {conv} does not have 'role' or 'content' key")

        
        if convs[0]['role'] != 'system':
            convs = self.init_conversation() + convs

        pt = 1
        
        while pt < len(convs):
            if convs[pt]['role'] != 'user':
                raise ValueError(f"the message should be user - assistant alternation, but the {pt}th message is {convs[pt]['role']}")
            pt += 1
            if pt >= len(convs):
                break
            if convs[pt]['role'] != 'assistant':
                raise ValueError(f"the message should be user - assistant alternation, but the {pt}th message is {convs[pt]['role']}")
            pt += 1
        return convs
    

    def init_conversation(self, system_prompt=None):
        if system_prompt is None:
            system_prompt = self.system_prompt
        return [{'role': 'system', 'content': system_prompt}]
    
    
    def refresh_conversation(self):
        self.conversation = self.init_conversation()
    
    
    def update_conversation(self, conversation = None, user_message=None, assistant_message=None):
        if conversation is None:
            conversation = self.conversation
        
        if user_message is None and assistant_message is None:
            raise ValueError("user_message or assistant_message should be provided")
        
        if user_message is not None:
            if conversation[-1]['role'] == 'user':
                raise ValueError("the message should be user - assistant alternation")
            conversation.append({'role': 'user', 'content': user_message})
        
        if assistant_message is not None:
            if conversation[-1]['role'] == 'assistant' or conversation[-1]['role'] == 'system':
                raise ValueError("the message should be user - assistant alternation")
            conversation.append({'role': 'assistant', 'content': assistant_message})
    
    
    def prepare_model_input(self, conversation=None):
        if conversation is None:
            conversation = self.conversation
        string_input = self.string_formatter(conversation)
        return string_input