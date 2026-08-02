"""Disabled-by-default probe contract for the standalone trainer.

The published workflow uses the standalone evaluation scripts. These adapters
keep ``train.py`` self-contained and fail clearly if unsupported in-training
probe orchestration is requested.
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
