from pytorch_lightning.utilities.cli import LightningCLI
from lcf_datamodule import DataModule
from packages.lcf_pl_model import LCFS_BERT_PL


if __name__ == '__main__':
    cli = LightningCLI(LCFS_BERT_PL, DataModule)