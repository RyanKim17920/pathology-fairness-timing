#!/usr/bin/env python3

import inspect
import unittest

import torch
from torch import nn

from tools.matched_cancer_stage_20260730.objectives import (
    BLACK_RACE_ID,
    HIDDEN_DIM,
    INPUT_DIM,
    OUTPUT_DIM,
    TEMPERATURE,
    WHITE_RACE_ID,
    StageAdapter,
    cancer_stage_loss,
    fair_supcon,
)
from tools.matched_stage_union_20260730.objectives import (
    fair_supcon as tested_fair_supcon,
)


class StageAdapterTest(unittest.TestCase):
    def test_canonical_shape_and_layers(self) -> None:
        adapter = StageAdapter()
        self.assertEqual((adapter.lin1.in_features, adapter.lin1.out_features),
                         (INPUT_DIM, HIDDEN_DIM))
        self.assertEqual((adapter.lin2.in_features, adapter.lin2.out_features),
                         (HIDDEN_DIM, OUTPUT_DIM))
        self.assertIsInstance(adapter.act1, nn.GELU)
        self.assertIsInstance(adapter.norm2, nn.LayerNorm)
        self.assertFalse(hasattr(adapter, "act2"))
        self.assertFalse(hasattr(adapter, "norm1"))
        output = adapter(torch.randn(7, INPUT_DIM))
        self.assertEqual(output.shape, (7, OUTPUT_DIM))
        torch.testing.assert_close(
            output.norm(dim=1), torch.ones(7), rtol=0, atol=1e-6
        )
        with self.assertRaisesRegex(ValueError, "shape"):
            adapter(torch.randn(7, INPUT_DIM - 1))

    def test_dedicated_initialization_is_deterministic_and_rng_isolated(self) -> None:
        torch.manual_seed(91)
        state = torch.random.get_rng_state()
        first = StageAdapter(init_seed=123)
        torch.testing.assert_close(torch.random.get_rng_state(), state, rtol=0, atol=0)

        _ = torch.randn(19)
        second = StageAdapter(init_seed=123)
        third = StageAdapter(init_seed=124)
        for left, right in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertFalse(torch.equal(first.lin1.weight, third.lin1.weight))


class CancerStageLossTest(unittest.TestCase):
    @staticmethod
    def _reference_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = torch.tensor(
            [
                [1.0, 0.0, 0.1],
                [0.8, 0.2, 0.0],
                [0.0, 1.0, 0.1],
                [0.1, 0.8, 0.2],
                [0.6, 0.6, 0.1],
                [0.5, 0.4, 0.2],
            ],
            requires_grad=True,
        )
        cancer = torch.tensor([0, 0, 1, 1, 0, 0])
        race = torch.tensor(
            [
                BLACK_RACE_ID,
                WHITE_RACE_ID,
                BLACK_RACE_ID,
                WHITE_RACE_ID,
                BLACK_RACE_ID,
                WHITE_RACE_ID,
            ]
        )
        return h, cancer, race

    def test_reuses_mathematically_tested_fair_supcon(self) -> None:
        self.assertIs(fair_supcon, tested_fair_supcon)

    def test_frozen_reference_value_and_gradient_are_finite_nonzero(self) -> None:
        h, cancer, race = self._reference_batch()
        result = cancer_stage_loss(h, cancer, race, fair_weight=0.75)
        gradient = torch.autograd.grad(result.total, h)[0]

        self.assertTrue(torch.isfinite(result.total))
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(result.cancer.detach()), 0.0)
        self.assertGreater(float(result.fair.detach()), 0.0)
        # Frozen after independent direct evaluation of the shared fair_supcon
        # primitive; these constants detect relation, masking, or weight drift.
        torch.testing.assert_close(
            result.cancer.detach(), torch.tensor(1.0516749620), rtol=0, atol=1e-6
        )
        torch.testing.assert_close(
            result.fair.detach(), torch.tensor(0.9380502105), rtol=0, atol=1e-6
        )
        torch.testing.assert_close(
            result.total.detach(), torch.tensor(1.7552125454), rtol=0, atol=1e-6
        )
        expected_gradient = torch.tensor(
            [
                [0.0171486363, -0.3653747141, -0.1714865565],
                [0.0462863930, -0.1851455718, 0.0636168718],
                [0.4164080322, 0.0069575459, -0.0695757121],
                [0.9998795390, -0.1714308262, 0.1857833564],
                [-0.8283824921, 0.8011631966, 0.1633159518],
                [-0.7650977373, 0.9448731542, 0.0229978841],
            ]
        )
        torch.testing.assert_close(
            gradient, expected_gradient, rtol=0, atol=1e-6
        )

    def test_api_has_no_downstream_diagnosis_input(self) -> None:
        parameters = inspect.signature(cancer_stage_loss).parameters
        self.assertEqual(
            list(parameters),
            ["h", "cancer_id", "race_id", "fair_weight"],
        )
        forbidden = {"tp53", "diagnosis", "outcome", "task_label", "y"}
        self.assertTrue(forbidden.isdisjoint(name.lower() for name in parameters))

    def test_missing_and_out_of_scope_races_are_excluded_from_fair_only(self) -> None:
        h = torch.tensor(
            [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.0, 1.0],
                [0.2, 0.8],
                [0.5, 0.5],
            ],
            requires_grad=True,
        )
        cancer = torch.tensor([0, 0, 1, 1, 0])
        race = torch.tensor([BLACK_RACE_ID, WHITE_RACE_ID, -1, 0, 99])
        result = cancer_stage_loss(h, cancer, race)
        expected_cancer = tested_fair_supcon(
            h, cancer, TEMPERATURE, relation="same"
        )
        expected_fair = tested_fair_supcon(
            h[:2],
            race[:2],
            TEMPERATURE,
            relation="same-condition-different",
            condition=cancer[:2],
        )
        torch.testing.assert_close(result.cancer, expected_cancer)
        torch.testing.assert_close(result.fair, expected_fair)

    def test_missing_cancer_and_anchors_without_positives_are_safe(self) -> None:
        h = torch.randn(5, 4, requires_grad=True)
        cancer = torch.tensor([-1, 0, 1, 2, 3])
        race = torch.tensor(
            [-1, BLACK_RACE_ID, WHITE_RACE_ID, BLACK_RACE_ID, WHITE_RACE_ID]
        )
        result = cancer_stage_loss(h, cancer, race)
        self.assertEqual(float(result.cancer.detach()), 0.0)
        self.assertEqual(float(result.fair.detach()), 0.0)
        gradient = torch.autograd.grad(result.total, h)[0]
        torch.testing.assert_close(gradient, torch.zeros_like(h))

    def test_fair_weight_scales_only_fair_component(self) -> None:
        h, cancer, race = self._reference_batch()
        default = cancer_stage_loss(h, cancer, race)
        first = cancer_stage_loss(h, cancer, race, fair_weight=0.0)
        second = cancer_stage_loss(h, cancer, race, fair_weight=2.5)
        torch.testing.assert_close(first.cancer, second.cancer)
        torch.testing.assert_close(first.fair, second.fair)
        torch.testing.assert_close(
            default.total, default.cancer + 0.1 * default.fair
        )
        torch.testing.assert_close(first.total, first.cancer)
        torch.testing.assert_close(
            second.total, second.cancer + 2.5 * second.fair
        )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            cancer_stage_loss(h, cancer, race, fair_weight=-0.1)


if __name__ == "__main__":
    unittest.main()
