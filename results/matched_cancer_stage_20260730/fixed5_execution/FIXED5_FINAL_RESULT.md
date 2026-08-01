# Fixed-five final result

Finalized on `2026-08-01` after Slurm job `368758` (`main_1gpu`)
completed with exit code `0:0`.  This report records the single permitted
scientific look for FM seeds `32001..32005`; it does not authorize additional
TP53 seeds.

## Provenance and exact topology

- one finalization attempt and one analyzer invocation;
- analyzer and independent-verifier semantic reports exactly equal;
- all 32 completion-receipt identities matched byte counts and SHA-256s;
- 182,700 prediction rows, 120 seed/arm/cancer/head cells, and 2,250 nested
  threshold audits;
- no FM seed outside `32001..32005` entered the collection; and
- the production job ran serially in one one-GPU allocation.

The canonical sealed artifacts are under
`/data/ryan.kim/nanopath/reruns/matched_cancer_fixed48_20260730/finalization/attempt_01`.

## Placement contrast

The preregistered contrast is

`theta = iEO(P) - iEO(H)`,

where positive values nominally favor post-hoc placement `H` because lower
iEO is better.

| FM seed | theta |
| --- | ---: |
| 32001 | -0.00044355 |
| 32002 | +0.00211872 |
| 32003 | +0.00759718 |
| 32004 | -0.00141999 |
| 32005 | +0.00178136 |

The mean was `+0.00192675`, median `+0.00178136`, and SD `0.00350110`.
The 90% CI was `[-0.00141117, +0.00526466]`; the 95% CI was
`[-0.00242044, +0.00627393]`.  The paired result was
`t(4)=1.23057`, two-sided `p=0.28590`; signs were 3 positive and 2 negative,
with exact sign-test `p=1.0`.  With only five independent FM seeds, the
smallest possible two-sided sign-test p-value is `0.0625`.

All leave-one-seed-out means were positive but tiny.  Cancer means were
`+0.00012466` for BRCA and `+0.00372883` for LUAD.  The prespecified head
halves disagreed: heads `42001/42002` gave `+0.00190705`, while heads
`42003/42004` gave `-0.00227547`.

## Locked classification

The result is `small_across_five_tested_seeds`, not meaningful/stable and not
inconclusive:

- `|mean(theta)| < 0.02`;
- every `|theta_seed| < 0.03`; and
- the 90% CI lies strictly inside `+/-0.03`.

This is a bounded practical-equivalence result for placement timing in this
tested system.  It is not a population claim and is not evidence that both
fairness interventions are useful.

## Why the apparent timing effect disappeared

Both fairness arms were nearly unchanged from baseline.  Relative to `B`,
overall AUROC changed by `-0.00149543` for `P` and `-0.00009656` for `H`.
Mean iEO changes inferred from the sealed cancer contrasts were also tiny:

| Arm versus B | BRCA | LUAD |
| --- | ---: | ---: |
| P | +0.00065500 | +0.00192443 |
| H | +0.00053033 | -0.00180440 |

The favored-arm harm gate passed, but its absolute utility gate failed:
`H` overall AUROC was `0.58382`, and LUAD AUROC was `0.51434`.  Baseline was
already similarly weak (`0.58392` overall and `0.51454` for LUAD).  Thus the
old apparent pre/post difference was unstable because the matched
interventions barely moved either fairness or utility on a weak downstream
LUAD endpoint.

## Consequence

Do not add TP53 seeds and do not claim useful timing equivalence.  The next
step is the prespecified zero-new-seed representation audit over the five
existing checkpoint sets.  Only a representation-level effect that passes
the frozen leakage and preservation gates may justify one targeted mechanism
seed; only a passing mechanism package may advance to three serial seeds in a
genuinely unseen scenario.
