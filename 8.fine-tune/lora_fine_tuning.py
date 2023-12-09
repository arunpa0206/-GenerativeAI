#pip install datasets
#pip install accelerate -U
#pip install torch
#pip install transformers[torch]
#pip install peft
#pip install evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import evaluate
import torch
import numpy as np
# Load dataset
dataset = load_dataset('shawhin/imdb-truncated')  #These lines import the load_dataset function from the datasets library. This function is used to load datasets from the Hugging Face Hub.
# Define label maps
id2label = {0: "Negative", 1: "Positive"}
label2id = {"Negative": 0, "Positive": 1}
# Load pre-trained model and tokenizer
model_checkpoint = 'distilbert-base-uncased'  #model_checkpoint = 'roberta-base' # you can alternatively use roberta-base but this model is bigger thus training will take longer
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2, id2label=id2label, label2id=label2id) #AutoModelForSequenceClassification: This class automatically loads a pre-trained model for sequence classification from the Hugging Face Hub.
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)   #AutoTokenizer: This class automatically loads a pre-trained tokenizer from the Hugging Face Hub.
# Tokenize datasets
def tokenize_function(examples):
    text = examples["text"]
    tokenized_inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512) #The return_tensors="pt" parameter specifies that the tokenized inputs should be returned as PyTorch tensors.
    return tokenized_inputs
tokenized_dataset = dataset.map(tokenize_function, batched=True)
# Data collator for padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)  # DataCollatorWithPadding: This class collates and pads examples for training and evaluation.
# Evaluation metrics function
accuracy = evaluate.load("accuracy")
# define an evaluation function to pass into trainer later. This line defines an evaluation function that computes the accuracy of the model on the validation dataset.
#The p parameter is a tuple of predictions and labels. The function first converts the predictions to labels, and then computes the accuracy using the accuracy metric.
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy.compute(predictions=predictions, references=labels)}
# PEFT Configuration- defines the configuration of a PeFT model.
peft_config = LoraConfig(task_type="SEQ_CLS", r=4, lora_alpha=32, lora_dropout=0.01, target_modules=['q_lin']) #LoraConfig: This class defines the configuration of a Lora module.
#task_type: This parameter specifies the type of task that the model is being used for. In this case, the task type is "SEQ_CLS", which indicates that the model is being used for sequence classification.
#r: This parameter specifies the number of Lora layers in the PeFT model. Lora layers are a type of neural network layer that uses a low-rank approximation of the weight matrix. This can reduce the number of parameters in the model and improve its generalization performance.
#lora_alpha: This parameter specifies the activation parameter for the Lora layers. The activation parameter controls the shape of the activation function used in the Lora layers.
#lora_dropout: This parameter specifies the dropout rate for the Lora layers. Dropout is a regularization technique that helps to prevent overfitting.
#target_modules: This parameter specifies which modules in the model to apply Lora to. In this case, the Lora modules are applied to the query linear layer. The query linear layer is a layer in the model that transforms the input text into a query vector.
model = get_peft_model(model, peft_config)  # get_peft_model: This function converts a standard Transformers model into a PeFT model.
# Training hyperparameters
lr = 1e-3
batch_size = 4
num_epochs = 10
# TrainingArguments: This class defines training arguments for the Trainer.
training_args = TrainingArguments(
    output_dir=model_checkpoint + "-lora-text-classification",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)
# Trainer: This class is used to train and evaluate a model.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
# Train the model
trainer.train()
# Prediction after training
model.to('mps')
text_list = ["It was good.", "Not a fan, don't recommend.", "Better than the first one.", "This is not worth watching even once.", "This one is a pass."]
for text in text_list:
    inputs = tokenizer.encode(text, return_tensors="pt").to("mps")
    logits = model(inputs).logits
    predictions = torch.max(logits,1).indices
    print(text + " - " + id2label[predictions.tolist()[0]])
# The text is tokenized using the tokenizer object.
#The tokenized text is converted to a PyTorch tensor.
#The model is called to compute the logits for the tokenized text.
#The logits are converted to predictions by taking the argmax of each logit.
#The predictions are converted to labels using the id2label dictionary.
#The text, the predicted label, and the true label are printed to the console.

