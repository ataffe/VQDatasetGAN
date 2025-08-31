def get_default_logger_cfgs(name, logdir, debug):
    # default logger configs
    return {
        "wandb": {
            "target": "lightning.pytorch.loggers.WandbLogger",
            "params": {
                "name": name,
                "save_dir": logdir,
                "offline": debug,
                "id": name,
            }
        },
        "tensorboard": {
            "target": "lightning.pytorch.loggers.TensorBoardLogger",
            "params": {
                "save_dir": logdir,
            }
        }
    }


def get_default_modelckpt_cfgs(checkpoint_dir):
    # default model checkpoint configs
    return {
        "target": "lightning.pytorch.callbacks.ModelCheckpoint",
        "params": {
            "dirpath": checkpoint_dir,
            "filename": "{epoch:06}",
            "verbose": True,
            "save_last": True,
        }
    }


def get_default_callbacks_cfg(resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
    return {
        "setup_callback": {
            "target": "vqgan.callbacks.SetupCallback",
            "params": {
                "resume": resume,
                "now": now,
                "logdir": logdir,
                "ckptdir": ckptdir,
                "cfgdir": cfgdir,
                "config": config,
                "lightning_config": lightning_config,
            }
        },
        "image_logger": {
            "target": "loggers.ImageLogger",
            "params": {
                "batch_frequency": 750,
                "max_images": 4,
                "clamp": True
            }
        },
        "learning_rate_logger": {
            "target": "loggers.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
                # "log_momentum": True
            }
        },
        "cuda_callback": {
            "target": "vqgan.callbacks.CUDACallback"
        },
    }
