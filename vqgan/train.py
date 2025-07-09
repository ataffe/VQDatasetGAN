import datetime
import sys
import os
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from omegaconf import OmegaConf
import argparse
from utils import instantiate_from_config
from packaging import version
import pytorch_lightning as pl
from defaults import get_default_logger_cfgs, get_default_modelckpt_cfgs, get_default_callbacks_cfg
import torch
import signal
import glob
from utils import get_parser, nondefault_trainer_args

if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )

    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

        ckptdir = os.path.join(logdir, "checkpoints")
        cfgdir = os.path.join(logdir, "configs")
        seed_everything(opt.seed)
        trainer = None

        try:
            # init and save configs
            configs = [OmegaConf.load(cfg) for cfg in opt.base]
            cli = OmegaConf.from_dotlist(unknown)
            config = OmegaConf.merge(*configs, cli)
            lightning_config = config.pop("lightning", OmegaConf.create())
            # merge trainer cli with config
            trainer_config = lightning_config.get("trainer", OmegaConf.create())
            # default to ddp
            trainer_config["accelerator"] = "auto"
            # trainer_config["gpus"] = 1
            for k in nondefault_trainer_args(opt):
                trainer_config[k] = getattr(opt, k)
            if not "gpus" in trainer_config:
                del trainer_config["accelerator"]
                cpu = True
            else:
                gpuinfo = trainer_config["gpus"]
                print(f"Running on GPUs {gpuinfo}")
                cpu = False
            trainer_opt = argparse.Namespace(**trainer_config)
            lightning_config.trainer = trainer_config

            # model
            model = instantiate_from_config(config.model)

            # trainer and callbacks
            trainer_kwargs = dict()

            default_logger_cfgs = get_default_logger_cfgs(nowname, logdir, opt.debug)
            default_logger_cfg = default_logger_cfgs["tensorboard"]
            if "logger" in lightning_config:
                logger_cfg = lightning_config.logger
            else:
                logger_cfg = OmegaConf.create()
            logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
            trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

            default_modelckpt_cfg = get_default_modelckpt_cfgs(ckptdir)

            if hasattr(model, "monitor"):
                print(f"Monitoring {model.monitor} as checkpoint metric.")
                default_modelckpt_cfg["params"]["monitor"] = model.monitor
                default_modelckpt_cfg["params"]["save_top_k"] = 3

            if "modelcheckpoint" in lightning_config:
                modelckpt_cfg = lightning_config.modelcheckpoint
            else:
                modelckpt_cfg = OmegaConf.create()
            modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
            print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")
            if version.parse(pl.__version__) < version.parse('1.4.0'):
                trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

            # add callback which sets up log directory
            default_callbacks_cfg = get_default_callbacks_cfg(
                opt.resume, now, logdir, ckptdir, cfgdir, config, lightning_config
            )
            if version.parse(pl.__version__) >= version.parse('1.4.0'):
                default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

            if "callbacks" in lightning_config:
                callbacks_cfg = lightning_config.callbacks
            else:
                callbacks_cfg = OmegaConf.create()

            if 'metrics_over_trainsteps_checkpoint' in callbacks_cfg:
                print(
                    'Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
                default_metrics_over_trainsteps_ckpt_dict = {
                    'metrics_over_trainsteps_checkpoint':
                        {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                         'params': {
                             "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                             "filename": "{epoch:06}-{step:09}",
                             "verbose": True,
                             'save_top_k': -1,
                             'every_n_train_steps': 10000,
                             'save_weights_only': True
                         }
                         }
                }
                default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)
            callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
            if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'resume_from_checkpoint'):
                callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
            elif 'ignore_keys_callback' in callbacks_cfg:
                del callbacks_cfg['ignore_keys_callback']

            trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

            # trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
            trainer_args = {**trainer_kwargs, **trainer_config}
            if 'gpus' in trainer_args:
                del trainer_args['gpus']
            trainer = Trainer(**trainer_args)
            trainer.logdir = logdir  ###
            print(f'Training for {trainer.max_epochs} min epochs.')

            # data
            data = instantiate_from_config(config.data)
            # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
            # calling these ourselves should not be necessary but it is.
            # lightning still takes care of proper multiprocessing though
            data.prepare_data()
            data.setup()
            print("#### Data #####")
            for k in data.datasets:
                print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

                # configure learning rate
                bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
                if not cpu:
                    # ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
                    ngpu = torch.cuda.device_count()
                else:
                    ngpu = 1
                if 'accumulate_grad_batches' in lightning_config.trainer:
                    accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
                else:
                    accumulate_grad_batches = 1
                print(f"accumulate_grad_batches = {accumulate_grad_batches}")
                lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
                if opt.scale_lr:
                    model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
                    print(
                        "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                            model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
                else:
                    model.learning_rate = base_lr
                    print("++++ NOT USING LR SCALING ++++")
                    print(f"Setting learning rate to {model.learning_rate:.2e}")


                # allow checkpointing via USR1
                def melk(*args, **kwargs):
                    # run all checkpoint hooks
                    if trainer.global_rank == 0:
                        print("Summoning checkpoint.")
                        ckpt_path = os.path.join(ckptdir, "last.ckpt")
                        trainer.save_checkpoint(ckpt_path)


                def divein(*args, **kwargs):
                    if trainer.global_rank == 0:
                        import pudb;
                        pudb.set_trace()


                signal.signal(signal.SIGUSR1, melk)
                signal.signal(signal.SIGUSR2, divein)
                # run
                if opt.train:
                    try:
                        trainer.fit(model, data)
                    except Exception:
                        melk()
                        raise
                if not opt.no_test and not trainer.interrupted:
                    trainer.test(model, data)

        except Exception as e:
            print(f"Exception during training: {e}")
            raise e
            # if opt.debug and trainer is not None and trainer.global_rank == 0:
            #     try:
            #         import pudb as debugger
            #     except ImportError:
            #         import pdb as debugger
            #     debugger.post_mortem()
            # raise

