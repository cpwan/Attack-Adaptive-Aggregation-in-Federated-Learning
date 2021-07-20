from runx.logx import logx

from utils.allocateGPU import *

allocate_gpu()

import parser
import _main

if __name__ == "__main__":
    args = parser.parse_args()
    logx.initialize(logdir=args.experiment_name, coolname=True, tensorboard=True,
                hparams=vars(args))
    logx.msg("#" * 64)
    for i in vars(args):
        logx.msg(f"#{i:>40}: {str(getattr(args, i)):<20}#")
    logx.msg("#" * 64)
    _main.main(args)
