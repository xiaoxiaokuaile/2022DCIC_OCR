# Learner: 王振强
# Learn Time: 2022/2/15 18:00

from .lookahead import Lookahead
from .radam import RAdam


def Ranger(params, alpha=0.5, k=6, betas=(.95, 0.999), *args, **kwargs):
    radam = RAdam(params, betas=betas, *args, **kwargs)
    return Lookahead(radam, alpha, k)


















