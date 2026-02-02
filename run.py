import argparse
import os
import torch
from configs import get_cfg
from util.net import init_training
from util.util import run_pre, init_checkpoint
from trainer import get_trainer
import warnings
warnings.filterwarnings("ignore")


def _run_once(cfg):
	init_checkpoint(cfg)
	trainer = get_trainer(cfg)
	trainer.run()


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--cfg_path', default='configs/mvad/mvad_realiad.py')
	parser.add_argument('-m', '--mode', default='train', choices=['train', 'test'])
	parser.add_argument('--sleep', type=int, default=-1)
	parser.add_argument('--memory', type=int, default=-1)
	parser.add_argument('--dist_url', default='env://', type=str, help='url used to set up distributed training')
	parser.add_argument('--logger_rank', default=0, type=int, help='GPU id to use.')
	parser.add_argument('opts', help='path.key=value', default=None, nargs=argparse.REMAINDER,)
	cfg_terminal = parser.parse_args()
	cfg = get_cfg(cfg_terminal)
	run_pre(cfg)
	init_training(cfg)
	if getattr(cfg.data, 'train_per_class', False) and isinstance(cfg.data.cls_names, list) and len(cfg.data.cls_names) > 0:
		orig_mode = cfg.mode
		cls_names = list(cfg.data.cls_names)
		for cls_name in cls_names:
			cfg.data.cls_names = [cls_name]
			cfg.trainer.logdir_sub = cls_name
			cfg.trainer.resume_dir = ''
			cfg.model.kwargs['checkpoint_path'] = ''
			cfg.trainer.__dict__.pop('metric_recorder', None)
			cfg.mode = orig_mode
			_run_once(cfg)
			if orig_mode == 'train':
				resume_dir = os.path.basename(cfg.logdir)
				cfg.trainer.resume_dir = resume_dir
				cfg.model.kwargs['checkpoint_path'] = ''
				cfg.trainer.__dict__.pop('metric_recorder', None)
				cfg.mode = 'test'
				_run_once(cfg)
				cfg.trainer.resume_dir = ''
		cfg.mode = orig_mode
	else:
		_run_once(cfg)


if __name__ == '__main__':
	main()
