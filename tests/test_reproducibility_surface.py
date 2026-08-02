import csv
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts" / name
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_module(path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_metadata_builder_is_deterministic(tmp_path):
    source = tmp_path / "patients.csv"
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["patient_barcode", "cancer", "race", "age"]
        )
        writer.writeheader()
        writer.writerows([
            {"patient_barcode": "P2", "cancer": "B", "race": "White", "age": 60},
            {"patient_barcode": "P1", "cancer": "A", "race": "Black", "age": 40},
            {"patient_barcode": "P3", "cancer": "A", "race": "White", "age": 50},
        ])
    output = tmp_path / "fino_meta.json"
    subprocess.run([
        sys.executable,
        str(ROOT / "scripts" / "build_fino_metadata.py"),
        "--csv", str(source),
        "--out", str(output),
        "--discrete", "cancer", "race",
        "--continuous", "age",
    ], check=True)
    result = json.loads(output.read_text())
    assert result["categories"]["cancer"] == ["A", "B"]
    assert result["discrete"]["cancer"] == {"P1": 0, "P2": 1, "P3": 0}
    assert abs(sum(result["continuous"]["age"].values())) < 1e-8


def test_environment_receipt_is_machine_readable_and_path_free():
    environment = load_script("environment_receipt.py")
    receipt = environment.build_receipt(ROOT)
    assert receipt["schema"] == environment.SCHEMA
    assert receipt["source"]["git_commit"]
    assert receipt["packages"]
    serialized = json.dumps(receipt)
    assert str(ROOT) not in serialized


def test_public_configs_match_except_intervention():
    plain = yaml.safe_load((ROOT / "configs" / "pretrain_plain.yaml").read_text())
    fair = yaml.safe_load(
        (ROOT / "configs" / "pretrain_cancer_conditioned_race.yaml").read_text()
    )
    assert plain["data"] == fair["data"]
    assert plain["model"] == fair["model"]
    assert plain["train"] == fair["train"]
    assert plain["dino"] == fair["dino"]
    assert not plain["fino"]["enabled"]
    assert fair["fino"]["contrastive_condition_on"] == "cancer"
    assert plain["data"]["exclude_barcodes_file"] == \
        "data/metadata/downstream_holdout.txt"


def test_posthoc_supports_multiclass_conditioned_head():
    posthoc = load_script("post_hoc_debias.py")
    model = posthoc.build_head(
        8, 4, "race", method="dann", condition_on_label=True, n_conditions=3
    )
    assert len(model.adv_by_y["race"]) == 3


def test_public_tree_has_no_personal_absolute_paths():
    forbidden = ("/admin/" + "home/", "/data/personal-user", "personal-account")
    checked = [
        *ROOT.glob("scripts/*.py"),
        *ROOT.glob("pretraining/*.py"),
        *ROOT.glob("configs/*.yaml"),
        *ROOT.glob("docs/*.md"),
    ]
    for path in checked:
        text = path.read_text()
        assert not any(value in text for value in forbidden), path


def test_downloader_is_revision_pinned(monkeypatch, tmp_path):
    prepare = load_script("prepare_data.py")
    observed = {}

    def snapshot_download(**kwargs):
        observed.update(kwargs)

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", types.SimpleNamespace(
            snapshot_download=snapshot_download
        )
    )
    receipt = {
        "schema": prepare.RECEIPT_SCHEMA,
        "local": {"file_count": 200, "total_rows": 1},
    }
    monkeypatch.setattr(prepare, "validate_tiles", lambda *_args, **_kwargs: receipt)
    prepare.download_tiles(tmp_path, "pretraining", workers=3)
    assert observed["repo_id"] == "medarc/nanopath"
    assert observed["revision"] == prepare.PRETRAINING_REVISION
    assert observed["allow_patterns"] == ["shard-*.parquet"]
    assert json.loads((tmp_path / "DATASET_RECEIPT.json").read_text()) == receipt


def test_optional_hf_cohort_download_requires_revision(monkeypatch, tmp_path):
    tiles = load_script("hf_tiles.py")
    observed = {}

    def snapshot_download(**kwargs):
        observed.update(kwargs)

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", types.SimpleNamespace(
            snapshot_download=snapshot_download
        )
    )
    with pytest.raises(ValueError, match="revision"):
        tiles.pull("brca", tmp_path, repo="org/tiles")
    tiles.pull("brca", tmp_path, repo="org/tiles", revision="a" * 40)
    assert observed["repo_id"] == "org/tiles"
    assert observed["revision"] == "a" * 40
    assert observed["allow_patterns"] == ["brca/**"]


def test_tile_validation_checks_complete_schema(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    prepare = load_script("prepare_data.py")
    for index in range(prepare.PRETRAINING_SHARDS):
        pq.write_table(
            pa.table({"path": [f"TCGA-AA-{index:04d}/tile.jpg"], "jpeg": [b"jpeg"]}),
            tmp_path / f"shard-{index:05d}.parquet",
        )
    spec = prepare.DATASETS["pretraining"]
    monkeypatch.setitem(spec, "expected_rows", prepare.PRETRAINING_SHARDS)
    monkeypatch.setitem(
        spec, "expected_bytes",
        sum(path.stat().st_size for path in tmp_path.glob("*.parquet")),
    )
    monkeypatch.setitem(
        spec, "lfs_manifest_sha256",
        prepare.local_content_manifest(tmp_path, "pretraining"),
    )
    receipt = prepare.validate_tiles(tmp_path, "pretraining", deep=True)
    assert receipt["local"]["file_count"] == prepare.PRETRAINING_SHARDS
    assert receipt["local"]["total_rows"] == prepare.PRETRAINING_SHARDS
    (tmp_path / "shard-00199.parquet").unlink()
    try:
        prepare.validate_tiles(tmp_path, "pretraining")
    except ValueError as error:
        assert "incomplete" in str(error)
    else:
        raise AssertionError("incomplete shard set was accepted")


def test_downstream_validation_rejects_partial_mirror(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    prepare = load_script("prepare_data.py")
    cohort = tmp_path / "1"
    cohort.mkdir()
    pq.write_table(
        pa.table({"slide_path": ["TCGA-AA-0001.svs"], "image_bytes": [b"jpeg"]}),
        cohort / "00000000.parquet",
    )
    try:
        prepare.validate_tiles(tmp_path, "downstream")
    except ValueError as error:
        assert "incomplete" in str(error)
    else:
        raise AssertionError("partial downstream mirror was accepted")


def test_dataset_receipt_is_bound_to_current_local_inventory(tmp_path, monkeypatch):
    from pathology_fairness import data_contracts

    cohort = tmp_path / "1"
    cohort.mkdir()
    shard = cohort / "00000000.parquet"
    original_payload = b"parquet-placeholder"
    shard.write_bytes(original_payload)
    original_stat = shard.stat()
    inventory = data_contracts.local_inventory(tmp_path, "downstream")
    spec = data_contracts.DATASETS["downstream"]
    monkeypatch.setitem(spec, "expected_files", 1)
    monkeypatch.setitem(spec, "expected_rows", 7)
    monkeypatch.setitem(spec, "expected_bytes", inventory["total_bytes"])
    monkeypatch.setitem(spec, "manifest_sha256", inventory["manifest_sha256"])
    content_manifest = data_contracts.local_content_manifest(tmp_path, "downstream")
    monkeypatch.setitem(spec, "lfs_manifest_sha256", content_manifest)
    receipt = {
        "schema": data_contracts.RECEIPT_SCHEMA,
        "dataset": "downstream",
        "source": {
            "repo_id": spec["repo"],
            "revision": spec["revision"],
            "lfs_manifest_sha256": spec["lfs_manifest_sha256"],
        },
        "local": {
            **inventory,
            "total_rows": 7,
            "content_manifest_sha256": content_manifest,
        },
    }
    (tmp_path / "DATASET_RECEIPT.json").write_text(json.dumps(receipt))
    identity = data_contracts.validate_dataset_receipt(tmp_path, "downstream")
    assert identity["inventory_sha256"] == inventory["inventory_sha256"]

    contaminant = tmp_path / "extra" / "contaminant.parquet"
    contaminant.parent.mkdir()
    contaminant.write_bytes(b"not-part-of-the-pinned-snapshot")
    with pytest.raises(ValueError, match="outside the pinned manifest"):
        data_contracts.validate_dataset_receipt(tmp_path, "downstream")
    contaminant.unlink()

    shard.write_bytes(b"x" * len(original_payload))
    os.utime(
        shard,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    with pytest.raises(ValueError, match="local inventory"):
        data_contracts.validate_dataset_receipt(tmp_path, "downstream")


def test_cohort_receipt_records_labeled_tile_coverage(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pathology_fairness import data_contracts

    prepare = load_script("prepare_data.py")
    downstream = tmp_path / "downstream"
    metadata = tmp_path / "metadata"
    (downstream / "1").mkdir(parents=True)
    metadata.mkdir()
    pq.write_table(
        pa.table({"slide_path": ["TCGA-AA-0001-SLIDE.svs"]}),
        downstream / "1" / "00000000.parquet",
    )
    pq.write_table(
        pa.table({"slide_path": ["TCGA-AA-0003-SLIDE.svs"]}),
        downstream / "1" / "00000001.parquet",
    )
    demographics = metadata / "tcga_demographics.csv"
    demographics.write_text(
        "patient_barcode,label_brca,label_nsclc,label_glioma\n"
        "TCGA-AA-0001,0,,\nTCGA-AA-0002,1,,\nTCGA-AA-0003,,0,\n"
    )
    metadata_receipt = {
        "schema": prepare.RECEIPT_SCHEMA,
        "source": {"canonical_response_sha256": "a" * 64},
        "outputs": {
            "demographics_csv": {"sha256": prepare._sha256_file(demographics)}
        },
    }
    metadata_receipt_path = metadata / "METADATA_RECEIPT.json"
    metadata_receipt_path.write_text(json.dumps(metadata_receipt))
    monkeypatch.setattr(
        data_contracts, "validate_dataset_receipt",
        lambda *_args: {"receipt_sha256": "b" * 64},
    )
    receipt = prepare.prepare_cohort_receipt(downstream, metadata)
    assert receipt["tasks"]["brca"]["labeled_patients"] == 2
    assert receipt["tasks"]["brca"]["patients_with_tiles"] == 1
    assert receipt["tasks"]["brca"]["missing_patient_ids"] == ["TCGA-AA-0002"]
    assert receipt["tasks"]["nsclc"]["patients_with_tiles"] == 1
    assert (metadata / "COHORT_RECEIPT.json").is_file()


def test_gdc_metadata_and_folds_are_deterministic(tmp_path, monkeypatch):
    prepare = load_script("prepare_data.py")
    cases = []

    def add(project, index, diagnosis, race="white"):
        cases.append({
            "submitter_id": f"TCGA-{project[:2]}-{index:04d}",
            "project": {"project_id": f"TCGA-{project}"},
            "demographic": {
                "race": race,
                "sex_at_birth": "female",
                "days_to_birth": -18262,
            },
            "diagnoses": [{"primary_diagnosis": diagnosis}],
        })

    for index in range(10):
        race = "white" if index % 2 == 0 else "asian"
        add("LUAD", index, "Adenocarcinoma, NOS", race)
        add("LUSC", 100 + index, "Squamous cell carcinoma, NOS", race)
        add("LGG", 200 + index, "Glioma, NOS", race)
        add("GBM", 300 + index, "Glioblastoma", race)
        add("BRCA", 400 + index, "Infiltrating duct carcinoma, NOS", race)
        add("BRCA", 500 + index, "Infiltrating lobular carcinoma, NOS", race)
    add("KIRC", 900, "Clear cell adenocarcinoma", "american indian or alaska native")

    first = prepare.clinical_rows(cases)
    second = prepare.clinical_rows(list(reversed(cases)))
    assert prepare._canonicalize(cases) == prepare._canonicalize(list(reversed(cases)))
    for task in ("nsclc", "glioma", "brca"):
        prepare.assign_folds(first, task)
        prepare.assign_folds(second, task)
    assert first == second
    assert {row["race"] for row in first if row["race"]} == {"Asian", "White"}
    assert sorted({row["fold_brca"] for row in first if row["label_brca"] != ""}) \
        == [0, 1, 2, 3, 4]

    monkeypatch.setattr(prepare, "download_gdc_cases", lambda: cases)
    receipt = prepare.prepare_clinical(
        tmp_path / "metadata", "brca", tmp_path / "tiles" / "fino_meta.json"
    )
    assert len(receipt["source"]["canonical_response_sha256"]) == 64
    assert receipt["task_class_counts"]["brca"] == {"0": 10, "1": 10}
    assert receipt["brca_subtype_counts"]["ambiguous"] == 0
    assert receipt["race_availability_counts"]["all_tcga"]["unsupported_category"] == 1
    for artifact in receipt["outputs"].values():
        assert artifact["filename"]
        assert "path" not in artifact
        assert len(artifact["sha256"]) == 64

    monkeypatch.setattr(
        prepare, "download_gdc_cases",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected metadata refresh")),
    )
    reused = prepare.prepare_clinical(
        tmp_path / "metadata", "brca", tmp_path / "tiles" / "fino_meta.json"
    )
    assert reused == receipt
    with pytest.raises(ValueError, match="study contract"):
        prepare.prepare_clinical(
            tmp_path / "metadata", "nsclc", tmp_path / "tiles" / "fino_meta.json"
        )


def test_posthoc_reads_every_parquet_row_group(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    posthoc = load_script("post_hoc_debias.py")
    patient = "TCGA-AA-0001"
    pq.write_table(
        pa.table({
            "slide_path": [
                f"/slides/{patient}-01Z-00-DX1.svs",
                f"/slides/{patient}-01Z-00-DX1.svs",
                f"/slides/{patient}-01Z-00-DX1.svs",
            ],
            "image_bytes": [b"one", b"two", b"three"],
        }),
        tmp_path / "slide.parquet",
        row_group_size=1,
    )
    task, pool = posthoc.collect_tiles(
        str(tmp_path), {patient}, {patient: {"race": "White", "sex": "F"}},
        "race", "task_only", 10, 0, 0,
    )
    assert [payload for _, payload in task] == [b"one", b"two", b"three"]
    assert pool == []


def test_subgroup_support_requires_both_outcome_classes(monkeypatch):
    fairness = load_script("fairness_eval.py")
    monkeypatch.setattr(fairness, "_safe_auc", lambda _y, _p: 0.5)
    report = fairness.subgroup_report(
        [0] * 30 + [1], [0.1] * 30 + [0.9], ["Black"] * 31,
        min_n=15, min_class_n=5,
    )
    subgroup = report["subgroups"]["Black"]
    assert subgroup["n"] == 31
    assert subgroup["n_pos"] == 1
    assert subgroup["insufficient_support"]
    assert not subgroup["eligible_for_gap"]


def test_evaluation_requires_checkpoint_unless_smoke_is_explicit(tmp_path):
    fairness = load_script("fairness_eval.py")
    with pytest.raises(ValueError, match="checkpoint is required"):
        fairness.build_backbone(None, device=None)
    with pytest.raises(FileNotFoundError, match="checkpoint does not exist"):
        fairness.build_backbone(tmp_path / "missing.pt", device=None)


def test_training_refuses_to_overwrite_nonempty_output(tmp_path):
    trainer = load_module(ROOT / "pretraining" / "train.py")
    output = tmp_path / "run"
    output.mkdir()
    (output / "result.json").write_text("{}\n")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        trainer.prepare_output_dir(output, resume_path=None)
    assert (output / "result.json").is_file()


def test_pretraining_requires_clean_tracked_source(tmp_path, monkeypatch):
    trainer = load_module(ROOT / "pretraining" / "train.py")
    responses = iter([
        types.SimpleNamespace(returncode=0, stdout="a" * 40 + "\n"),
        types.SimpleNamespace(returncode=0),
        types.SimpleNamespace(returncode=0),
    ])
    monkeypatch.setattr(trainer.subprocess, "run", lambda *_args, **_kwargs: next(responses))
    assert trainer.clean_source_commit(tmp_path) == "a" * 40

    dirty = iter([
        types.SimpleNamespace(returncode=0, stdout="b" * 40 + "\n"),
        types.SimpleNamespace(returncode=1),
    ])
    monkeypatch.setattr(trainer.subprocess, "run", lambda *_args, **_kwargs: next(dirty))
    with pytest.raises(RuntimeError, match="uncommitted changes"):
        trainer.clean_source_commit(tmp_path)


def test_training_rejects_incomplete_pretraining_shards(tmp_path):
    trainer = load_module(ROOT / "pretraining" / "train.py")
    (tmp_path / "shard-00000.parquet").touch()
    with pytest.raises(FileNotFoundError, match="found=1 expected=200"):
        trainer.validate_pretraining_shards(tmp_path)


def test_training_binds_generated_input_receipts(tmp_path, monkeypatch):
    trainer = load_module(ROOT / "pretraining" / "train.py")
    tiles = tmp_path / "tiles"
    metadata = tmp_path / "metadata"
    tiles.mkdir()
    metadata.mkdir()
    holdout = metadata / "downstream_holdout.txt"
    holdout.write_text("TCGA-AA-0001\n")
    fino = tiles / "fino_meta.json"
    fino.write_text("{}\n")
    monkeypatch.setattr(
        trainer, "validate_dataset_receipt", lambda *_args: {
            "receipt_sha256": "1" * 64,
            "inventory_sha256": "2" * 64,
            "lfs_manifest_sha256": "3" * 64,
        },
    )
    metadata_receipt = {
        "schema": trainer.RECEIPT_SCHEMA,
        "fold_seed": 1337,
        "holdout_task": "brca",
        "outputs": {
            "holdout_file": {"sha256": trainer.sha256_file(holdout)},
            "fino_meta": {"sha256": trainer.sha256_file(fino)},
        }
    }
    (metadata / "METADATA_RECEIPT.json").write_text(json.dumps(metadata_receipt))
    cfg = {
        "data": {
            "dataset_dir": str(tiles),
            "exclude_barcodes_file": str(holdout),
            "holdout_task": "brca",
        },
        "fino": {"enabled": True},
    }
    identity = trainer.validate_input_receipts(cfg)
    assert identity["holdout_sha256"] == trainer.sha256_file(holdout)
    assert identity["holdout_task"] == "brca"
    assert identity["fino_meta_sha256"] == trainer.sha256_file(fino)

    cfg["data"]["holdout_task"] = "nsclc"
    with pytest.raises(ValueError, match="holdout task"):
        trainer.validate_input_receipts(cfg)

    first = {
        "project": {"output_dir": "one", "wandb_dir": "one"},
        "train": {"resume": None, "seed": 1},
    }
    second = {
        "project": {"output_dir": "two", "wandb_dir": "two"},
        "train": {"resume": "latest.pt", "seed": 1},
    }
    assert trainer.config_identity(first) == trainer.config_identity(second)
    second["train"]["seed"] = 2
    assert trainer.config_identity(first) != trainer.config_identity(second)


def test_reliable_runner_binds_downstream_coverage_and_checkpoint_holdout(
    tmp_path, monkeypatch
):
    import torch

    reliable = load_script("reliable_fairness_head.py")
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    demographics = metadata / "tcga_demographics.csv"
    demographics.write_text("patient_barcode,label_brca\nTCGA-AA-0001,0\n")
    metadata_receipt = {
        "schema": reliable.RECEIPT_SCHEMA,
        "fold_seed": 1337,
        "holdout_task": "brca",
        "task_patients": {"brca": 1},
        "outputs": {
            "demographics_csv": {"sha256": reliable.sha256_file(demographics)}
        },
    }
    metadata_receipt_path = metadata / "METADATA_RECEIPT.json"
    metadata_receipt_path.write_text(json.dumps(metadata_receipt))
    metadata_receipt_sha = reliable.sha256_file(metadata_receipt_path)
    downstream_identity = {
        "receipt_sha256": "d" * 64,
        "inventory_sha256": "e" * 64,
        "lfs_manifest_sha256": "f" * 64,
    }
    monkeypatch.setattr(
        reliable, "validate_dataset_receipt",
        lambda *_args: downstream_identity,
    )
    cohort_receipt = {
        "schema": "pathology-fairness-cohort/v1",
        "inputs": {
            "downstream_dataset_receipt_sha256": "d" * 64,
            "demographics_sha256": reliable.sha256_file(demographics),
            "metadata_receipt_sha256": metadata_receipt_sha,
        },
        "tasks": {
            "brca": {
                "labeled_patients": 1,
                "patients_with_tiles": 1,
                "missing_patients": 0,
            }
        },
    }
    (metadata / "COHORT_RECEIPT.json").write_text(json.dumps(cohort_receipt))
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({
        "input_identity": {
            "holdout_task": "brca",
            "metadata_receipt_sha256": metadata_receipt_sha,
            "pretraining_revision": "revision",
        }
    }, checkpoint)
    runtime = tmp_path / "runtime.json"
    runtime.write_text(json.dumps({
        "study": "test",
        "fold_seed": 1337,
        "head_architecture": "linear-relu-dropout-linear",
        "planned_head_seeds": [1],
        "primary_estimand": "posthoc_auc_gap_minus_pretraining_auc_gap",
        "utility_noninferiority_margin": 0.02,
        "sensitive": "race",
        "method": "contrastive",
        "condition_col": "cancer_type",
        "temperature": 0.2,
        "posthoc_lambda": 0.1,
    }))
    reliable.configure_cache_contract(
        study_task="brca", core_task="brca", hospital_fold=None,
        hospital_folds_csv=None, split_seed=1337, head_seed=1,
        runtime_contract=runtime, demographics_csv=demographics,
        tiles_dir=tmp_path / "downstream", checkpoint=checkpoint,
        execution_contract={
            "sensitive": "race", "method": "contrastive",
            "condition_col": "cancer_type", "temperature": 0.2,
            "lambda_adv": 0.0,
        },
    )
    assert reliable._CACHE_CONTRACT["cohort_receipt"]["coverage"][
        "patients_with_tiles"
    ] == 1
    assert reliable._CACHE_CONTRACT["checkpoint_training"]["holdout_task"] == "brca"


def test_base_embedding_cache_never_requires_pickle(tmp_path):
    import numpy as np

    posthoc = load_script("post_hoc_debias.py")
    tiles = [("TCGA-AA-0001", b"jpeg")]

    def embed(_tiles):
        return np.asarray([[1.0, 2.0]], dtype=np.float32), np.asarray([True])

    embeddings, barcodes = posthoc.cached_embed(
        "test|checkpoint=abc", tiles, embed, str(tmp_path), log=lambda *_: None
    )
    assert embeddings.tolist() == [[1.0, 2.0]]
    assert barcodes.tolist() == ["TCGA-AA-0001"]
    cache_path = next(tmp_path.glob("*.npz"))
    with np.load(cache_path, allow_pickle=False) as stored:
        assert stored["barcodes"].dtype.kind == "U"

    posthoc.cached_embed(
        "test|checkpoint=abc", [("TCGA-AA-0001", b"changed")], embed,
        str(tmp_path), log=lambda *_: None,
    )
    assert len(list(tmp_path.glob("*.npz"))) == 2


def test_auxiliary_pool_selection_is_balanced_and_deterministic():
    posthoc = load_script("post_hoc_debias.py")
    patients = [f"P{index:02d}" for index in range(16)]
    sens = {
        patient: {"race": "Black" if index % 2 else "White", "sex": "F"}
        for index, patient in enumerate(patients)
    }
    condition = {patient: index % 4 for index, patient in enumerate(patients)}
    first, receipt = posthoc.select_balanced_pool_patients(
        patients, sens, "race", condition, limit=8, seed=17
    )
    second, _ = posthoc.select_balanced_pool_patients(
        list(reversed(patients)), sens, "race", condition, limit=8, seed=17
    )
    assert first == second
    assert receipt["selected_patients"] == 8
    assert len(receipt["selected_strata"]) == 4
    assert set(receipt["selected_strata"].values()) == {2}


def test_zero_lambda_baseline_is_invariant_to_auxiliary_pool(monkeypatch):
    import numpy as np
    import torch

    posthoc = load_script("post_hoc_debias.py")
    monkeypatch.setattr(posthoc.fe, "_safe_auc", lambda _y, _p: 0.5)
    monkeypatch.setattr(
        posthoc.fe, "subgroup_report",
        lambda *_args, **_kwargs: {"subgroups": {}, "auc_delta": None},
    )
    task_patients = np.asarray([f"P{index}" for index in range(8)])
    pool_patients = np.asarray([f"Q{index}" for index in range(4)])
    embeddings = np.arange(32, dtype=np.float32).reshape(8, 4) / 32
    pool_embeddings = np.arange(16, dtype=np.float32).reshape(4, 4) / 16
    labels = {patient: (index // 2) % 2
              for index, patient in enumerate(task_patients)}
    folds = {patient: index % 2 for index, patient in enumerate(task_patients)}
    sens = {
        patient: {"race": "Black" if index % 2 else "White", "sex": "F"}
        for index, patient in enumerate([*task_patients, *pool_patients])
    }
    posthoc._DEMO_MAP = {
        patient: {"race": values["race"], "gender": "female", "age_years": "65"}
        for patient, values in sens.items()
    }

    common = dict(
        emb_task=embeddings, bc_task=task_patients, label_of=labels,
        fold_of=folds, sens=sens, sensitive="race", eval_fold=0, lambd=0.0,
        hidden=4, lr=1e-3, epochs=2, batch_size=2,
        device=torch.device("cpu"), method="dann", dump_records=True,
        log=lambda *_: None,
    )
    without_pool = posthoc.train_and_eval(
        emb_pool=np.zeros((0, 4), dtype=np.float32),
        bc_pool=np.asarray([], dtype=np.str_), **common,
    )
    with_pool = posthoc.train_and_eval(
        emb_pool=pool_embeddings, bc_pool=pool_patients, **common,
    )
    assert [row["y_score"] for row in without_pool["predictions"]] == \
        [row["y_score"] for row in with_pool["predictions"]]
    assert without_pool["n_adversary_tiles"] == 0
    assert with_pool["n_adversary_tiles"] == 0


def test_posthoc_conditional_objective_matches_shared_primitive():
    import torch

    from pathology_fairness.objectives import fair_supcon

    posthoc = load_script("post_hoc_debias.py")
    representation = torch.tensor([
        [1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9],
    ], requires_grad=True)
    sensitive = torch.tensor([0, 1, 0, 1])
    condition = torch.tensor([0, 0, 1, 1])
    production = posthoc._fair_supcon(
        representation, {"race": sensitive}, ["race"], 0.2,
        torch, torch.nn.functional, task_labels=condition,
    )
    reference = fair_supcon(
        representation, sensitive, 0.2,
        relation="same_condition_different", condition=condition,
    )
    assert torch.allclose(production, reference, atol=1e-6)
    production_gradient = torch.autograd.grad(
        production, representation, retain_graph=True
    )[0]
    reference_gradient = torch.autograd.grad(reference, representation)[0]
    assert torch.allclose(production_gradient, reference_gradient, atol=1e-6)


def test_timing_analysis_is_paired_and_bootstrapped(tmp_path):
    analysis = load_script("analyze_timing.py")
    paths = {arm: tmp_path / f"{arm}.jsonl"
             for arm in ("control", "pretraining", "posthoc")}
    handles = {arm: path.open("w") for arm, path in paths.items()}
    try:
        for group in ("A", "B"):
            for label in (0, 1):
                for index in range(10):
                    patient = f"{group}-{label}-{index}"
                    scores = {
                        "control": (0.9 if label else 0.1) if group == "A"
                                   else (0.1 if label else 0.9),
                        "pretraining": 0.9 if label else 0.1,
                        "posthoc": (0.9 if label else 0.1) if group == "A" else 0.5,
                    }
                    for arm, handle in handles.items():
                        handle.write(json.dumps({
                            "patient_id": patient, "y_true": label,
                            "y_score": scores[arm], "race": group,
                            "sex": "F", "age": 65, "outer_fold": index % 5,
                        }) + "\n")
    finally:
        for handle in handles.values():
            handle.close()

    data = analysis.load_arms(
        [paths["control"]], [paths["pretraining"]], [paths["posthoc"]],
        "race", 65,
    )
    groups = analysis.eligible_groups(data["y"], data["groups"], 15, 5)
    assert groups == ["A", "B"]
    per_run = analysis._run_summary(
        data["y"], data["groups"], data["scores"], groups
    )
    assert per_run[0]["contrasts"]["pretraining_vs_posthoc"][
        "fairness_advantage"
    ] == pytest.approx(0.5)
    bootstrap = analysis.bootstrap_timing(
        data["y"], data["groups"], data["scores"], groups, 200, 7
    )
    assert bootstrap["fairness_advantage"]["ci_95"][0] > 0

    runtime = tmp_path / "runtime.json"
    runtime_declaration = {
        "study": "test",
        "fold_seed": 1337,
        "head_architecture": "linear-relu-dropout-linear",
        "planned_head_seeds": [1],
        "primary_estimand": "posthoc_auc_gap_minus_pretraining_auc_gap",
        "utility_noninferiority_margin": 0.02,
        "sensitive": "race",
        "method": "contrastive",
        "condition_col": "cancer_type",
        "temperature": 0.2,
        "posthoc_lambda": 0.1,
    }
    runtime.write_text(json.dumps(runtime_declaration))
    result_paths = {arm: tmp_path / f"{arm}-result.json" for arm in paths}
    for arm in paths:
        fairness_enabled = arm == "pretraining"
        result = {
            "lambda_adv": 0.1 if arm == "posthoc" else 0.0,
            "reliable_fairness": {
                "schema": "reliable-fairness-head/v2",
                "core_task": "brca",
                "split_seed": 1337,
                "head_seed": 1,
                "outer_fold_count": 5,
                "prediction_artifact": {
                    "sha256": analysis.sha256_file(paths[arm]),
                    "record_count": 40,
                    "required_outer_folds": list(range(5)),
                },
                "checkpoint_identity": {
                    "sha256": "fair" if fairness_enabled else "plain"
                },
                "study_cache_contract": {
                    "runtime_contract": {
                        "sha256": analysis.sha256_file(runtime)
                    },
                    "checkpoint_training": {
                        "fairness_intervention": {
                            "enabled": fairness_enabled,
                            "objective": (
                                "contrastive-two-condition"
                                if fairness_enabled else None
                            ),
                            "method": "contrastive" if fairness_enabled else None,
                            "condition_on": "cancer" if fairness_enabled else None,
                            "temperature": 0.2 if fairness_enabled else None,
                        }
                    },
                },
            },
        }
        result_paths[arm].write_text(json.dumps(result))

    prediction_lists = {arm: [path] for arm, path in paths.items()}
    result_lists = {arm: [path] for arm, path in result_paths.items()}
    provenance = analysis.validate_run_provenance(
        prediction_lists, result_lists, runtime, runtime_declaration
    )
    assert provenance["control"][0]["checkpoint_sha256"] == "plain"

    duplicate_predictions = dict(prediction_lists)
    duplicate_predictions["pretraining"] = [paths["control"]]
    with pytest.raises(ValueError, match="same prediction file"):
        analysis.validate_run_provenance(
            duplicate_predictions, result_lists, runtime, runtime_declaration
        )

    swapped_results = dict(result_lists)
    swapped_results["control"] = [result_paths["pretraining"]]
    with pytest.raises(ValueError, match="prediction digest mismatch"):
        analysis.validate_run_provenance(
            prediction_lists, swapped_results, runtime, runtime_declaration
        )

    posthoc_result = json.loads(result_paths["posthoc"].read_text())
    posthoc_result["reliable_fairness"]["head_seed"] = 2
    result_paths["posthoc"].write_text(json.dumps(posthoc_result))
    with pytest.raises(ValueError, match="head seed is out of order"):
        analysis.validate_run_provenance(
            prediction_lists, result_lists, runtime, runtime_declaration
        )
