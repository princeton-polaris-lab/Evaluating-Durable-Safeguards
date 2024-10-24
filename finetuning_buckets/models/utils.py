from trl import DataCollatorForCompletionOnlyLM

def get_model(model_name_or_path, model_kwargs, model_family='llama2', padding_side = "right"):
    
    if model_family == 'llama2':
        from .model_families.llama2 import initializer as llama2_initializer
        return llama2_initializer(model_name_or_path, model_kwargs, padding_side=padding_side)
    elif model_family == 'llama2_repnoise':
        from .model_families.llama2_repnoise import initializer as llama2_repnoise_initializer
        return llama2_repnoise_initializer(model_name_or_path, model_kwargs, padding_side=padding_side)
    elif model_family == 'llama3':
        from .model_families.llama3 import initializer as llama3_initializer
        return llama3_initializer(model_name_or_path, model_kwargs, padding_side=padding_side)
    elif model_family == 'gemma':
        from .model_families.gemma import initializer as gemma_initializer
        return gemma_initializer(model_name_or_path, model_kwargs, padding_side=padding_side)
    elif model_family == 'gemma2':
        from .model_families.gemma2 import initializer as gemma2_initializer
        return gemma2_initializer(model_name_or_path, model_kwargs, padding_side=padding_side)
    elif model_family == 'mistral':
        from .model_families.mistral import initializer as mistral_initializer
        return mistral_initializer(model_name_or_path, model_kwargs, padding_side=padding_side)
    else:
        raise ValueError(f"model_family {model_family} not maintained")
    

def get_training_string_formatter(tokenizer, model_family):

    if model_family == 'llama2':
        from .model_families.llama2 import get_training_string_formatter as llama2_get_training_string_formatter
        return llama2_get_training_string_formatter(tokenizer)
    elif model_family == 'llama2_repnoise':
        from .model_families.llama2_repnoise import get_training_string_formatter as llama2_repnoise_get_training_string_formatter
        return llama2_repnoise_get_training_string_formatter(tokenizer)
    elif model_family == 'llama3':
        from .model_families.llama3 import get_training_string_formatter as llama3_get_training_string_formatter
        return llama3_get_training_string_formatter(tokenizer)
    elif model_family == 'gemma':
        from .model_families.gemma import get_training_string_formatter as gemma_get_training_string_formatter
        return gemma_get_training_string_formatter(tokenizer)
    elif model_family == 'gemma2':
        from .model_families.gemma2 import get_training_string_formatter as gemma2_get_training_string_formatter
        return gemma2_get_training_string_formatter(tokenizer)
    elif model_family == 'mistral':
        from .model_families.mistral import get_training_string_formatter as mistral_get_training_string_formatter
        return mistral_get_training_string_formatter(tokenizer)
    else:
        raise ValueError(f"model_family {model_family} not maintained")


def get_data_collator(tokenizer, response_template = None, model_family = 'llama2', ntp = False):
    
    if response_template is None:

        if model_family == 'llama2':
            from finetuning_buckets.models.model_families.llama2 import CustomDataCollator as LlamaCustomDataCollator
            return LlamaCustomDataCollator(tokenizer=tokenizer, ntp=ntp)
        elif model_family == 'llama2_repnoise':
            from finetuning_buckets.models.model_families.llama2_repnoise import CustomDataCollator as Llama2RepNoiseCustomDataCollator
            return Llama2RepNoiseCustomDataCollator(tokenizer=tokenizer, ntp=ntp)
        elif model_family == 'llama3':
            from finetuning_buckets.models.model_families.llama3 import CustomDataCollator as Llama3CustomDataCollator
            return Llama3CustomDataCollator(tokenizer=tokenizer, ntp=ntp)
        elif model_family == 'gemma':
            from finetuning_buckets.models.model_families.gemma import CustomDataCollator as GemmaCustomDataCollator
            return GemmaCustomDataCollator(tokenizer=tokenizer, ntp=ntp)
        elif model_family == 'gemma2':
            from finetuning_buckets.models.model_families.gemma2 import CustomDataCollator as Gemma2CustomDataCollator
            return Gemma2CustomDataCollator(tokenizer=tokenizer, ntp=ntp)
        elif model_family == 'mistral': 
            from finetuning_buckets.models.model_families.mistral import CustomDataCollator as MistralCustomDataCollator
            return MistralCustomDataCollator(tokenizer=tokenizer, ntp=ntp)
        else:
            raise ValueError("response_template or dataset_name should be provided")

    else:
    
        return DataCollatorForCompletionOnlyLM(response_template=response_template, 
                                                    tokenizer=tokenizer)