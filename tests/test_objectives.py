from __future__ import annotations

import inspect
import unittest

import torch

from pathology_fairness.objectives import (
    condition_preserving_fairness_loss,
    fair_supcon,
    relation_consistent_mask,
)


class ObjectiveTests(unittest.TestCase):
    def test_all_pairs_objective_is_finite_and_differentiable(self) -> None:
        representations = torch.tensor(
            [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]],
            requires_grad=True,
        )
        condition = torch.tensor([0, 0, 1, 1])
        sensitive = torch.tensor([0, 1, 0, 1])
        loss = condition_preserving_fairness_loss(
            representations, condition, sensitive, objective="all_pairs"
        )
        gradient = torch.autograd.grad(loss.total, representations)[0]
        self.assertTrue(torch.isfinite(loss.total))
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.norm()), 0.0)

    def test_relation_consistent_mask_uses_only_matched_relations(self) -> None:
        representations = torch.tensor(
            [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.0, 1.0],
                [0.2, 0.8],
            ],
            requires_grad=True,
        )
        condition = torch.tensor([0, 0, 1, 1])
        sensitive = torch.tensor([0, 1, 0, 1])
        value = relation_consistent_mask(
            representations, condition, sensitive, temperature=0.2
        )
        gradient = torch.autograd.grad(value, representations)[0]
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.norm()), 0.0)

    def test_missing_labels_and_missing_pairs_return_differentiable_zero(self) -> None:
        representations = torch.randn(4, 3, requires_grad=True)
        condition = torch.tensor([0, -1, 1, -1])
        sensitive = torch.tensor([0, -1, 0, -1])
        value = relation_consistent_mask(representations, condition, sensitive)
        gradient = torch.autograd.grad(value, representations)[0]
        self.assertEqual(float(value.detach()), 0.0)
        torch.testing.assert_close(gradient, torch.zeros_like(gradient))

    def test_weighted_supcon_rejects_negative_weights(self) -> None:
        representations = torch.eye(3)
        labels = torch.tensor([0, 1, 1])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            fair_supcon(
                representations,
                labels,
                anchor_weights=torch.tensor([1.0, -1.0, 1.0]),
            )

    def test_public_objective_api_has_no_downstream_outcome_input(self) -> None:
        forbidden = {"diagnosis", "outcome", "task_label", "tp53", "y"}
        for function in (
            condition_preserving_fairness_loss,
            fair_supcon,
            relation_consistent_mask,
        ):
            names = {name.lower() for name in inspect.signature(function).parameters}
            self.assertTrue(forbidden.isdisjoint(names))


if __name__ == "__main__":
    unittest.main()
