from .biot import apply_patch as biot
from .swag_vit import apply_patch as swag_vit
from .labram import apply_patch as labram
from .eegpt import apply_patch as eegpt
from .tfm_tokenizer import apply_patch as tfm_tokenizer

__all__ = [
    'biot', 'swag_vit', 'labram', 'eegpt', 'tfm_tokenizer'
]