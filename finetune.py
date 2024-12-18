from diffusers import DiffusionPipeline  # Import the DiffusionPipeline
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, TrainerCallback  # Existing imports
import torch
import matplotlib.pyplot as plt

# Define the model architecture (using Diffusers)
model_id = "stabilityai/stable-diffusion-2-1"  # This can be any model ID
pipe = DiffusionPipeline.from_pretrained(model_id)  # Load the pre-trained Diffusion model

UNET_TARGET_MODULES = [
    "to_q",
    "to_k",
    "to_v",
    "proj",
    "proj_in",
    "proj_out",
    "conv",
    "conv1",
    "conv2",
    "conv_shortcut",
    "to_out.0",
    "time_emb_proj",
    "ff.net.2",
]

# Define LoRA configuration
lora_config = LoraConfig(
    r=16,  # LoRA rank
    lora_alpha=32,  # LoRA alpha
    target_modules=UNET_TARGET_MODULES,  # Specify target modules for LoRA
    lora_dropout=0.1,  # LoRA dropout
    bias="none",  # Special token for bias
)

# Initialize the PeftModel with random weights
peft_model = get_peft_model(model=pipe.unet, peft_config=lora_config)

peft_model.print_trainable_parameters()

# Load your dataset
dataset = load_dataset("imagefolder", data_dir="golf_image_data")

# Check the structure of the dataset
print(dataset)

# Ensure the dataset contains the expected keys
if "train" not in dataset or "validation" not in dataset:
    raise ValueError("Dataset must contain 'train' and 'validation' splits.")
    
# Optionally, print a sample from the training dataset to verify its contents
print(dataset["train"][0])  # Check the first item in the training dataset

# Set up training parameters
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize lists to store losses
train_losses = []
val_losses = []

# Create a custom callback to record losses
class LossLogger(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            train_losses.append(logs.get("loss", None))
            val_losses.append(logs.get("eval_loss", None))

# Create a Trainer instance with the custom callback
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    callbacks=[LossLogger()],
)

# Train the model
trainer.train()

# Plot training and validation loss
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Save the randomly initialized fine-tuned model
peft_model.save_pretrained("./golf_finetuned_sd21")

