"""Pipeline for Anormly Data Detection
    Main Difference from the LTSM : 
        - pred_len == seq_len
        - label is the anomaly label of input seq_len
        - loss is CE/BCE

"""

import numpy as np
import torch
import argparse
import random
import ipdb
from torch import nn

from ltsm.data_provider.data_factory import get_datasets
from ltsm.data_provider.data_loader import HF_Dataset
from ltsm.data_pipeline.model_manager import ModelManager

import logging
from transformers import (
    Trainer,
    TrainingArguments
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

class AnomalyModelManager(ModelManager):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Computes the loss for model training.

        Args:
            model (torch.nn.Module): The model used for predictions.
            inputs (dict): Input data and labels.
            return_outputs (bool): If True, returns both loss and model outputs.

        Returns:
            torch.Tensor or tuple: The computed loss, and optionally the outputs.
        """
        outputs = model(inputs["input_data"]) # output should be B, L, M
        labels = inputs["labels"]
        #print(outputs.shape, labels.shape)
        #B, L, M, _ = outputs.shape
        loss = nn.functional.cross_entropy(outputs, labels)
        #loss = nn.functional.cross_entropy(outputs.reshape(B*L,-1), inputs["labels"][:,1:].long().reshape(B*L))
        return (loss, outputs) if return_outputs else loss

class AnomalyTrainingPipeline():
    """
    A pipeline for managing the training and evaluation process of a machine learning model.

    Attributes:
        args (argparse.Namespace): Arguments containing training configuration and hyperparameters.
        model_manager (ModelManager): An instance responsible for creating, managing, and optimizing the model.
    """
    def __init__(self, args: argparse.Namespace):
        """
        Initializes the TrainingPipeline with given arguments and a model manager.

        Args:
            args (argparse.Namespace): Contains training settings such as output directory, batch size,
                                       learning rate, and other hyperparameters.
        """
        self.args = args
        self.model_manager = AnomalyModelManager(args)

    def run(self):
        """
        Runs the training and evaluation process for the model.

        The process includes:
            - Logging configuration and training arguments.
            - Creating a model with the model manager.
            - Setting up training and evaluation parameters.
            - Loading and formatting training and evaluation datasets.
            - Training the model and saving metrics and state.
            - Evaluating the model on test datasets and logging metrics.
        """
        logging.info(self.args)
    
        model = self.model_manager.create_model()
        
        # Training settings
        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.batch_size,
            evaluation_strategy="steps",
            num_train_epochs=self.args.train_epochs,
            fp16=False,
            save_steps=100,
            eval_steps=25,
            logging_steps=5,
            learning_rate=self.args.learning_rate,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            save_total_limit=10,
            remove_unused_columns=False,
            push_to_hub=False,
            load_best_model_at_end=True,
        )

        train_dataset, eval_dataset, test_datasets, _ = get_datasets(self.args)
        train_dataset, eval_dataset= HF_Dataset(train_dataset), HF_Dataset(eval_dataset)
        
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=self.model_manager.collate_fn,
            compute_metrics=self.model_manager.compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=None,
            optimizers=(self.model_manager.optimizer, self.model_manager.scheduler),
        )

        # Overload the trainer API
        if not self.args.eval:
            trainer.compute_loss = self.model_manager.compute_loss
            trainer.prediction_step = self.model_manager.prediction_step        
            train_results = trainer.train()
            trainer.save_model()
            trainer.log_metrics("train", train_results.metrics)
            trainer.save_metrics("train", train_results.metrics)
            trainer.save_state()

        # Testing settings
        for test_dataset in test_datasets:
            trainer.compute_loss = self.model_manager.compute_loss
            trainer.prediction_step = self.model_manager.prediction_step
            test_dataset = HF_Dataset(test_dataset)

            metrics = trainer.evaluate(test_dataset)
            trainer.log_metrics("Test", metrics)
            trainer.save_metrics("Test", metrics)

def anomaly_get_args():
    parser = argparse.ArgumentParser(description='LTSM')

    # Basic Config
    parser.add_argument('--model_id', type=str, default='test_run', help='model id')
    parser.add_argument('--model_name_or_path', type=str, default="gpt2-medium", help='model name')
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

    # Data Settings
    parser.add_argument('--data_path', nargs='+', default='dataset/weather.csv', help='data files')
    parser.add_argument('--test_data_path_list', nargs='+', required=True, help='test data file')
    parser.add_argument('--prompt_data_path', type=str, default='./weather.csv', help='prompt data file')
    parser.add_argument('--data_processing', type=str, default="standard_scaler", help='data processing method')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='train data ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='validation data ratio')
    parser.add_argument('--do_anomaly', type=bool, default=True, help='do anomaly detection')

    # Forecasting Settings
    parser.add_argument('--seq_len', type=int, default=113, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=None, help='prediction sequence length')
    parser.add_argument('--prompt_len', type=int, default=133, help='prompt sequence length')

    # Model Settings
    parser.add_argument('--lora', action="store_true", help='use lora')
    parser.add_argument('--lora_dim', type=int, default=128, help='dimension of lora')
    parser.add_argument('--gpt_layers', type=int, default=3, help='number of gpt layers')
    parser.add_argument('--d_model', type=int, default=1024, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=16, help='number of heads')
    parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
    parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
    parser.add_argument('--c_out', type=int, default=862, help='output size')
    parser.add_argument('--patch_size', type=int, default=16, help='patch size')
    parser.add_argument('--pretrain', type=int, default=1, help='is pretrain')
    parser.add_argument('--local_pretrain', type=str, default="None", help='local pretrain weight')
    parser.add_argument('--freeze', type=int, default=1, help='is model weight frozen')
    parser.add_argument('--model', type=str, default='model', help='model name, , options:[LTSM, LTSM_WordPrompt, LTSM_Tokenizer]')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--tmax', type=int, default=10, help='tmax')
    
    # Training Settings 
    parser.add_argument('--eval', type=int, default=0, help='evaluation')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--output_dir', type=str, default='output/ltsm_train_lr0005/', help='output directory')
    parser.add_argument('--downsample_rate', type=int, default=100, help='downsample rate')
    parser.add_argument('--llm_layers', type=int, default=32)
    parser.add_argument('--decay_fac', type=float, default=0.75, help='decay factor')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--num_workers', type=int, default=10, help='number of workers')
    parser.add_argument('--train_epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--lradj', type=str, default='type1', help='learning rate adjustment type')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=64, help='gradient accumulation steps')
    args, unknown = parser.parse_known_args()

    if args.pred_len is None:
        logging.info(f"Anomaly Mode, Set pred_len to seq_len")
        args.pred_len = args.seq_len

    return args


def anomaly_seed_all(fixed_seed):
    random.seed(fixed_seed)
    torch.manual_seed(fixed_seed)
    np.random.seed(fixed_seed)