from openai import OpenAI
client = OpenAI()

# requirement:
# openai > 1.0


file = client.files.create(
  file=open("finetune_truth.jsonl", "rb"),
  purpose="fine-tune"
)

client.fine_tuning.jobs.create(
  training_file=file.id, 
  model="davinci-002", # curie is now unavailable
  hyperparameters={
    "n_epochs":5,
    "batch_size": 21,
    "learning_rate_multiplier": 0.1
  }
)