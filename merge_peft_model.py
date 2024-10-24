from peft import LoraConfig, AutoPeftModelForCausalLM
import argparse
from transformers import HfArgumentParser
from dataclasses import dataclass, field

def merge_peft_model(path):
    model = AutoPeftModelForCausalLM.from_pretrained(path, device_map="auto")
    merged_model = model.merge_and_unload()
    path = path + '_full' # Cannot directly save the merged model to the orginal file, because adapter_config.json will affect the data structure of the model
    merged_model.save_pretrained(path)

@dataclass
class ScriptArguments:
    path: str = field(default=None, metadata={"help": "the dataset name"})
    
    
if __name__ == '__main__':
    parser = HfArgumentParser( ScriptArguments )
    args, = parser.parse_args_into_dataclasses()
    merge_peft_model(args.path)
    print("sucessfully merged the model from the path: ", args.path)