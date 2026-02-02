import torch
from torch import optim as optim
from timm.optim.adafactor import Adafactor
from timm.optim.adahessian import Adahessian
from timm.optim.adamp import AdamP
try:
	from timm.optim.nadam import Nadam
except Exception:
	Nadam = getattr(optim, 'NAdam', None)
try:
	from timm.optim.radam import RAdam
except Exception:
	RAdam = getattr(optim, 'RAdam', None)
try:
	from timm.optim.rmsprop_tf import RMSpropTF
except Exception:
	RMSpropTF = None
try:
	from timm.optim.sgdp import SGDP
except Exception:
	SGDP = None
try:
	from timm.optim.lookahead import Lookahead
except Exception:
	Lookahead = None


def check_keywords_in_name(name, keywords=()):
	isin = False
	for keyword in keywords:
		if keyword in name:
			isin = True
	return isin


def add_weight_decay(model, weight_decay=1e-5, skip_list=(), skip_keywords=()):
	decay = []
	no_decay = []
	for name, param in model.named_parameters():
		if not param.requires_grad:
			continue  # frozen weights
		if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or check_keywords_in_name(name, skip_keywords):
			no_decay.append(param)
		else:
			decay.append(param)
	return [
		{'params': no_decay, 'weight_decay': 0.},
		{'params': decay, 'weight_decay': weight_decay}]


def get_optim(optim_kwargs, net, lr, betas=None, filter_bias_and_bn=True):
	kwargs = {k: v for k, v in optim_kwargs.items()}
	optim_split = kwargs.pop('name').lower().split('_')
	optim_name = optim_split[-1]
	optim_lookahead = optim_split[0] if len(optim_split) == 2 else None
	
	if kwargs.get('weight_decay', None) and filter_bias_and_bn:
		skip = {}
		skip_keywords = {}
		if hasattr(net, 'no_weight_decay'):
			skip = net.no_weight_decay()
		if hasattr(net, 'no_weight_decay_keywords'):
			skip_keywords = net.no_weight_decay_keywords()
		params = add_weight_decay(net, kwargs['weight_decay'], skip, skip_keywords)
		kwargs['weight_decay'] = 0.
	else:
		params = net.parameters()
		
	if kwargs.get('betas', None) and betas:
		kwargs['betas'] = betas
	
	optim_terms = {
		'sgd': optim.SGD,
		'adam': optim.Adam,
		'adamw': optim.AdamW,
		'adadelta': optim.Adadelta,
		'rmsprop': optim.RMSprop,
		'nadam': Nadam,
		'radam': RAdam,
		'adamp': AdamP,
		'sgdp': SGDP,
		'adafactor': Adafactor,
		'adahessian': Adahessian,
		'rmsproptf': RMSpropTF,
	}
	if optim_terms['nadam'] is None:
		optim_terms.pop('nadam')
	if optim_terms.get('radam') is None:
		optim_terms.pop('radam', None)
	if optim_terms.get('sgdp') is None:
		optim_terms.pop('sgdp', None)
	if optim_terms.get('rmsproptf') is None:
		optim_terms.pop('rmsproptf', None)
	optimizer = optim_terms[optim_name](params, lr=lr, **kwargs)
	if optim_lookahead and Lookahead is not None:
		optimizer = Lookahead(optimizer)
	return optimizer
