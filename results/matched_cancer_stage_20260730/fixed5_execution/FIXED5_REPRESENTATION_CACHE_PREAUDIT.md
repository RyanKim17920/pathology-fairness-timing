# Fixed-five representation cache pre-audit

Completed read-only on `2026-08-01`, before the representation audit was
launched or any representation-audit metric was computed.

- Accepted FM-seed ancestry: `5/5` passed.
- Completion-receipt/checkpoint bindings: `25/25` passed.
- Existing final B/P/H cache entries: `30/30` passed.
- Cache rows independently checked: `22,895,445`.
- Cache bytes read: `19,988,086,305`.
- Failures: `0`.

Every cache had the exact member/schema topology, finite float32 128D rows,
per-tile L2 normalization, barcode/keep-mask alignment, independently
reproduced entry digest, completion/checkpoint identity, and encoder/adapter
state provenance.

Shared evidence by cancer:

| cancer | caches | patients | input = valid tiles | ordered-tile SHA-256 | aggregate evidence SHA-256 |
| --- | ---: | ---: | ---: | --- | --- |
| BRCA | 15 | 328 | 778,468 | `931f88c954e7649cbc7982cc86a0d92877a2e7d64c9d6426035396e1db1e434e` | `f07860176138ce5a0f9e5c5e7e7658bea270866e3032a9b1daef244b713ad727` |
| LUAD | 15 | 281 | 747,895 | `ed2fca91922eecc7611aabf2c0283540d7eff86f656112a1d0fc2990cb515f95` | `acd1264928783176c8814374dbed8cfd86de4c8cdb9b39b5d51c5abc5d73c102` |

The aggregate semantic-summary SHA-256 over the production pre-audit was
`8e759779049ba35156b06660cf5e019f4d1ccfdc05d052b7dcadb927505e25e5`.

This pre-audit is corroborative only. The production pipeline must repeat all
cache, ancestry, and shared-tile checks inside the bound `main_1gpu`
allocation; it may not trust this document as an input.
