import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import temos.launch.prepare  # noqa
import os
import pdb

# from pprint import pprint

logger = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="train_asymov")
def _train(cfg: DictConfig):
    # print(OmegaConf.to_yaml(cfg))
    if cfg.user:
        os.environ['WANDB_API_KEY'] = cfg.wandb_api_keys[cfg.user]
    cfg.trainer.enable_progress_bar = True
    return train(cfg)


def train(cfg: DictConfig) -> None:
    working_dir = cfg.path.working_dir
    logger.info("Training script. The outputs will be stored in:")
    logger.info(f"{working_dir}")

    # Delayed imports to get faster parsing
    logger.info("Loading libraries")
    import torch
    import pytorch_lightning as pl
    from hydra.utils import instantiate
    from temos.logger import instantiate_logger
    logger.info("Libraries loaded")

    logger.info(f"Set the seed to {cfg.seed}")
    pl.seed_everything(cfg.seed)

    logger.info("Loading data module")
    data_module = instantiate(cfg.data)
    logger.info(f"Data module '{cfg.data.name}' loaded")

    logger.info("Loading model")
    model = instantiate(cfg.model,
                        # nfeats=data_module.nfeats,
                        nvids_to_save=None,
                        _recursive_=False)
    logger.info(f"Model '{cfg.model.modelname}' loaded")

    logger.info("Loading callbacks")
    metric_monitor = {
        # "Train_jf": "recons/text2jfeats/train",
        # "Val_jf": "recons/text2jfeats/val",
        # "Train_rf": "recons/text2rfeats/train",
        # "Val_rf": "recons/text2rfeats/val",
        # "APE root": "Metrics/APE_root",
        # "APE mean pose": "Metrics/APE_mean_pose",
        # "AVE root": "Metrics/AVE_root",
        # "AVE mean pose": "Metrics/AVE_mean_pose"
        "Train_acc": "Metrics/acc_text2mw/train",
        "Val_acc": "Metrics/acc_text2mw/val",
        "Train_BLEU": "Metrics/bleu_text2mw/train",
        "Val_BLEU": "Metrics/bleu_text2mw/val",
        "Train_ppl": "Metrics/ppl_text2mw/train",
        "Val_ppl": "Metrics/ppl_text2mw/val"
    }
    callbacks = [
        instantiate(cfg.callback.progress, metric_monitor=metric_monitor),
        instantiate(cfg.callback.latest_ckpt),
        instantiate(cfg.callback.last_ckpt)
    ]
    if cfg.callback.viz_ckpts:
        callbacks.append(instantiate(cfg.callback.viz_ckpt))
    # pdb.set_trace()
    for monitor, mode in cfg.callback.best_ckpt_monitors:
        callbacks.append(instantiate(cfg.callback.best_ckpt, monitor=monitor, mode=mode))
    logger.info("Callbacks initialized")

    #TODO: instantiate using hydra
    logger.info("Loading trainer")
    trainer = pl.Trainer(
        **OmegaConf.to_container(cfg.trainer, resolve=True),
        logger=instantiate_logger(cfg),
        callbacks=callbacks,
    )

    # logger.info("Tuning the LR of model..")                               #auto_lr_find=True now so Tuning the LR first
    # lr_finder = trainer.tuner.lr_find(model, datamodule=data_module)#, ckpt_path=cfg.resume_ckpt_path)
    # print(lr_finder.results)

    # # Plot with
    # fig = lr_finder.plot(suggest=True)
    # # fig.show()
    # fig.savefig('/content/drive/MyDrive/Asymov/lr_finder.png')
    # fig.savefig('/content/drive/MyDrive/Asymov/lr_finder.svg')

    # logger.info(f"Tuning done. Suggested LR:{lr_finder.suggestion()}")    #Commented as gives error while normal run
    # cfg.model.optim.lr = lr_finder.suggestion()                           #viz.py", line 1074, in naive_no_rep_reconstruction
                                                                            #AssertionError: 02198 - Negative left boundary

    logger.info("Trainer initialized")
    checkpoint_folder = trainer.checkpoint_callback.dirpath
    logger.info(f"The checkpoints are stored in {checkpoint_folder}")

    logger.info("Fitting the model..")
    if cfg.resume_ckpt_path is not None :
        logger.info(f'Resuming training from checkpoint {cfg.resume_ckpt_path}')
    trainer.fit(model, datamodule=data_module, ckpt_path=cfg.resume_ckpt_path)
    logger.info("Fitting done")

    logger.info(f"Training done. The outputs of this experiment are stored in:\n{working_dir}")


if __name__ == '__main__':
    _train()
