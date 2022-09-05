from datetime import datetime as DT
from pathlib import Path
import pandas as pd
import pytorch_lightning as pl
from lcf_datamodule import DataModule
from packages.lcf_pl_model import LCFS_BERT_PL
import optuna
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner


MODEL_NAME = 'bert-base-uncased'
TRAIN_BATCH_SIZE = 8
MAX_SEQ_LENGTH = 48
DATA_PATH = Path.cwd()/'data/acl-14-short/clean/'

def objective(trial: optuna.trial.Trial) -> float:
    #model_name = trial.suggest_categorical("model_name", ["bert-base-uncased", "distilbert-base-uncased"])
    model_name = 'bert-base-uncased'
    #lr = trial.sugest_float("learning_rate", 1e-5, 5e-5, step=2e-5)
    lr = 2e-5
    #dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.2)
    dropout_rate = 0.2
    #weight_decay = trial.suggest_float("weight_decay", 0.01, 0.1)
    local_context_focus = trial.suggest_categorical("lcf", ["cdw", "cdm"])
    synthactic_distance_dependency = trial.suggest_int("srd", 6, 12, step=3)

    dm = DataModule(
    model_name=model_name, batch_size=TRAIN_BATCH_SIZE, 
    train_val_split=0.2, num_workers=2,
    data_dir=DATA_PATH, max_seq_length=MAX_SEQ_LENGTH
    )
    model = LCFS_BERT_PL(
    model_name, max_seq_length=MAX_SEQ_LENGTH, 
    synthactic_distance_dependency=synthactic_distance_dependency,
    dropout_rate=dropout_rate, local_context_focus=local_context_focus, lr=lr
    )
    trainer = pl.Trainer(
        accelerator='gpu', devices='1', strategy='dp',
        max_epochs=20, accumulate_grad_batches=5, 
        callbacks=[PyTorchLightningPruningCallback(trial, monitor='val_loss')]
    )
    hyperparameters = dict(
        lr=lr, dropout_rate=dropout_rate, SRD=synthactic_distance_dependency,
        lcf=local_context_focus
    )
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=dm)

    return trainer.callback_metrics["val_loss"].item()

if __name__ == "__main__":
    timestamp = DT.strftime(DT.now(),'%Y%m%d')
    study = optuna.create_study(
        study_name=f'lcfs_acl14_{timestamp}', direction='minimize', 
        sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.SuccessiveHalvingPruner(),
        storage=f'sqlite:///lcfs_acl14_{timestamp}.db', load_if_exists=True
    )
    study.optimize(objective, n_trials=20)
    print(f"Best value {study.best_value} w/ params: {study.best_params}")