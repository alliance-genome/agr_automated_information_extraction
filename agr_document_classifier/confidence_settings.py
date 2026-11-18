"""Per-model confidence threshold and rescaling (SCRUM-6482).

A document classifier calls a paper positive when its score reaches the model's
``ml_model.confidence_threshold`` (0.5 when unset), and ``ml_model.rescale_confidence`` says whether the
tag's confidence score is rescaled so that threshold maps to 0.5 or keeps the model's own score.
"""
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Models whose confidence settings are a curator decision rather than something the trainer tunes:
# (MOD, topic) -> (ml_model.confidence_threshold, ml_model.rescale_confidence). ZFIN molecular probe
# (SCRUM-5764): only papers scored 0.87 or more leave the acquisition workflow, and the tags show the
# model's own score. The trainer gives every new version of these models the default, and the classifier
# falls back to it if a model row lacks a threshold, because a probe positive is dropped from the
# acquisition workflow and never seen again.
MODEL_CONFIDENCE_DEFAULTS = {
    ("ZFIN", "ATP:0000370"): (0.87, False),
}


def model_confidence_settings(mod_abbreviation: str, topic: str,
                              model_meta_data: dict) -> Tuple[Optional[float], bool]:
    """(confidence_threshold, rescale_confidence) to classify with, from the model's ml_model metadata.

    A model in MODEL_CONFIDENCE_DEFAULTS with no threshold on its row gets its default, with an error logged:
    the row is missing what the ABC migration and the trainer give it."""
    threshold = model_meta_data.get("confidence_threshold")
    rescale = model_meta_data.get("rescale_confidence") is not False
    default = MODEL_CONFIDENCE_DEFAULTS.get((mod_abbreviation, topic))
    if threshold is None and default is not None:
        logger.error(f"{mod_abbreviation} {topic}: ml_model {model_meta_data.get('ml_model_id')} has no "
                     f"confidence_threshold; using the default {default[0]} (rescale_confidence={default[1]})")
        return default
    return (None if threshold is None else float(threshold)), rescale
