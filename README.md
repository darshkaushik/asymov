# asymov

Running MT model

 - Clone the `main` branch
 - Install dependencies
    
    ```python
    !pip install pytorch_lightning --upgrade
    !pip install torchmetrics==0.7
    !pip install hydra-core --upgrade
    !pip install hydra_colorlog --upgrade
    !pip install shortuuid
    !pip install tqdm
    !pip install pandas
    !pip install transformers
    !pip install psutil
    !pip install einops
    !pip install wandb
    ```

 - Change the directory to TEMOS
    
    ```python
    %cd '<dir_path>/asymov/packages/TEMOS'
    ```
 
 - Test training code on tiny dataset

    ```
    !HYDRA_FULL_ERROR=1 python train_asymov_mt.py experiment=<exp name> user=<shared, darsh etc.> data.splitpath='${path.datasets}/kit-splits-tiny' num_mw_clusters=1000 trainer=<cpu, gpu> trainer.max_epochs=10 model.max_frames=100 model.metrics.recons_types=['naive'] viz_metrics_start_epoch=0 viz_metrics_every_n_epoch=3
    ```

 - Run training code

    ```
    !HYDRA_FULL_ERROR=1 python train_asymov_mt.py experiment=<exp name> run_id=<optional> user=<shared, darsh etc.> num_mw_clusters=1000 trainer=gpu model.max_frames=1000 model.metrics.recons_types=['naive', 'naive_no_rep']
    ```

    other options that might be helpful:

    ```
    path.code_dir=<> hydra.run.dir=<> trainer.max_epochs=<> viz_metrics_start_epoch=<> viz_metrics_every_n_epoch=<> callback.last_ckpt.every_n_epochs=<> callback.viz_ckpt.start_epoch=<> callback.viz_ckpt.every_n_epochs=<>
‌
    ```
