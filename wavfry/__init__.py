"""WavFry acoustic feature extraction and classification toolkit."""

import warnings

# Suppress upstream PyTorch 2.6+ deprecation warning regarding WavLM's boolean key_padding_mask
# and floating-point gated_position_bias (handled internally by PyTorch auto-casting).
warnings.filterwarnings(
    "ignore",
    message=".*Support for mismatched key_padding_mask and attn_mask is deprecated.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*torchaudio.backend.common.AudioMetaData.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*No positive class found in y_true.*",
    category=UserWarning,
)
