# **************** For CASIA-B ****************
# GaitSCM Ablation Experiments

# phase1: pretrain featrure extraction module
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 lib/main.py --cfgs ./config/gaitscm_pretrain_casiab.yaml --phase train

# phase2: featrure disentanglement module
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 lib/main.py --cfgs ./config/gaitscm_fdm_casiab.yaml --phase train

# phase3: backdoor adjustment module
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 lib/main.py --cfgs ./config/gaitscm_ba_casiab.yaml --phase train



# **************** For OUMVLP ****************
# GaitSCM Ablation Experiments

# phase1: pretrain featrure extraction module
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 lib/main.py --cfgs ./config/gaitscm_pretrain_oumvlp.yaml --phase train

# phase2: featrure disentanglement module
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 lib/main.py --cfgs ./config/gaitscm_fdm_oumvlp.yaml --phase train

# phase3: backdoor adjustment module
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 lib/main.py --cfgs ./config/gaitscm_ba_oumvlp.yaml --phase train
