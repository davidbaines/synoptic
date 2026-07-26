"""From-scratch encoder-decoder model construction.

A randomly-initialised MarianMT-style transformer. Transformer-big scale comes
from the YAML config; the smoke config shrinks every dimension for the 3090.
Special-token ids are taken from the trained SentencePiece model so the model,
collator and tokeniser agree: unk=0, bos=1, eos=2, pad=3.
"""

from __future__ import annotations

from .config import ModelConfig


def build_model(model_cfg: ModelConfig, sp, max_position_embeddings: int = 512):
    """Build a randomly-initialised MarianMTModel sized by ``model_cfg``."""
    from transformers import MarianConfig, MarianMTModel

    if model_cfg.arch != "marian":
        raise ValueError(
            f"Unsupported model arch {model_cfg.arch!r}: only a from-scratch "
            "'marian' model is implemented; there is no fine-tune path"
        )
    vocab_size = sp.get_piece_size()
    pad_id = sp.pad_id()
    eos_id = sp.eos_id()
    bos_id = sp.bos_id()

    config = MarianConfig(
        vocab_size=vocab_size,
        d_model=model_cfg.d_model,
        encoder_layers=model_cfg.encoder_layers,
        decoder_layers=model_cfg.decoder_layers,
        encoder_attention_heads=model_cfg.encoder_attention_heads,
        decoder_attention_heads=model_cfg.decoder_attention_heads,
        encoder_ffn_dim=model_cfg.encoder_ffn_dim,
        decoder_ffn_dim=model_cfg.decoder_ffn_dim,
        dropout=model_cfg.dropout,
        max_position_embeddings=max_position_embeddings,
        scale_embedding=True,
        share_encoder_decoder_embeddings=True,
        pad_token_id=pad_id,
        eos_token_id=eos_id,
        bos_token_id=bos_id,
        # Marian starts decoding from the pad token (no separate BOS in NMT).
        decoder_start_token_id=pad_id,
        forced_eos_token_id=eos_id,
    )
    model = MarianMTModel(config)
    return model
