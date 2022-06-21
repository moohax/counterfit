import numpy as np
from hyperopt import hp
from art.attacks.evasion import HopSkipJump
from counterfit.modules.algo.art.art import ArtEvasionAttack


class CFHopSkipJump(ArtEvasionAttack):
    attack_cls = HopSkipJump
    tags = ["image", "tabular"]
    category = "blackbox"

    parameters = {
        "default": {
            "targeted": False,
            "norm": 2,
            "max_iter": 50,
            "max_eval": 1000,
            "init_eval": 100,
            "init_size": 100,
        },
        "optimize": {
            "targeted": hp.choice("hsj_targeted", [False, True]),
            "norm": hp.choice("hsj_norm", [2, np.inf]),
            "max_iter": hp.quniform("hsj_maxiter", 10, 100, 1),
            "max_eval": hp.quniform("hsj_maxeval", 300, 1000, 1),
            "init_eval": hp.quniform("hsj_initeval", 10, 200, 1),
            "init_size": hp.quniform("hsj_initsize", 10, 200, 1),
        },
    }
