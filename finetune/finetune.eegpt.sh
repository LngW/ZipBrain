#!/bin/bash

# We use uv to manage python version and venv, please install it if possible
# Please be aware of that modifications has to be made to source codes to run this script.

# CLI to kickoff a finetune on TUAB
(cd downstream_tueg && bash finetune_TUAB_EEGPT.sh)

# CLI to kickoff a finetune on TUEV
(cd downstream_tueg && bash finetune_TUEV_EEGPT.sh)

# CLI to kickoff a finetune on WorkLoad (EEGMAT)
(cd downstream && python finetune_EEGPT_EEGMAT.py)

# CLI to kickoff a finetune on EarEEG
(cd downstream && python finetune_EEGPT_EarEEG.py)

# CLI to kickoff a finetune on ISRUC
(cd downstream && python finetune_EEGPT_ISRUC.py)
