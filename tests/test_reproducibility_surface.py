import csv
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts" / name
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
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


def test_tile_validation_checks_complete_schema(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    prepare = load_script("prepare_data.py")
    for index in range(prepare.PRETRAINING_SHARDS):
        pq.write_table(
            pa.table({"path": [f"TCGA-AA-{index:04d}/tile.jpg"], "jpeg": [b"jpeg"]}),
            tmp_path / f"shard-{index:05d}.parquet",
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

    first = prepare.clinical_rows(cases)
    second = prepare.clinical_rows(list(reversed(cases)))
    for task in ("nsclc", "glioma", "brca"):
        prepare.assign_folds(first, task)
        prepare.assign_folds(second, task)
    assert first == second
    assert {row["race"] for row in first} == {"Asian", "White"}
    assert sorted({row["fold_brca"] for row in first if row["label_brca"] != ""}) \
        == [0, 1, 2, 3, 4]

    monkeypatch.setattr(prepare, "download_gdc_cases", lambda: cases)
    receipt = prepare.prepare_clinical(
        tmp_path / "metadata", "brca", tmp_path / "tiles" / "fino_meta.json"
    )
    assert len(receipt["source"]["canonical_response_sha256"]) == 64
    for artifact in receipt["outputs"].values():
        assert Path(artifact["path"]).is_file()
        assert len(artifact["sha256"]) == 64


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
