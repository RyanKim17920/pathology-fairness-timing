"""Disabled-by-default probe contract for the standalone public trainer.

The original training environment queued a separate internal benchmark suite.
Public recipes use the standalone evaluation scripts instead.  Keeping these
small adapters makes ``train.py`` self-contained while failing clearly if an
unsupported in-training probe is requested.
"""


def probe_enabled(cfg):
    enabled = bool((cfg.get("probe") or {}).get("enabled", False))
    if enabled:
        raise ValueError(
            "in-training probes are not bundled; set probe.enabled=false and "
            "run scripts/fairness_eval.py after pretraining"
        )
    return False


def prepare_probe_state(*_args, **_kwargs):
    return None


def collect_probe_results(*_args, **_kwargs):
    return None


def queue_probe_job(*_args, **_kwargs):
    return None


def completed_probe_summary(*_args, **_kwargs):
    return {}
