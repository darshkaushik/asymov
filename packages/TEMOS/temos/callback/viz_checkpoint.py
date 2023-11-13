from pytorch_lightning.callbacks import ModelCheckpoint

#TODO: implement in a cleaner way, put the start epoch_condition in higher level methods of the class
class VizCheckPoint(ModelCheckpoint):
    def __init__(self, start_epoch=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_epoch = start_epoch
    
    def _save_checkpoint(self, trainer, filepath):
        if trainer.current_epoch>=self.start_epoch:
            return super()._save_checkpoint(trainer, filepath)
