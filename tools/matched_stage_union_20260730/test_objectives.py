#!/usr/bin/env python3

import unittest
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from objectives import fair_supcon


class FairSupConTest(unittest.TestCase):
    def test_marginal_matches_frozen_reference_value_and_gradient(self) -> None:
        z = torch.tensor(
            [
                [1.0, 0.0, 0.1],
                [0.8, 0.2, 0.0],
                [0.0, 1.0, 0.1],
                [0.1, 0.8, 0.2],
            ],
            requires_grad=True,
        )
        race = torch.tensor([0, 0, 1, 1])
        value = fair_supcon(z, race, 0.2, relation="different")
        gradient = torch.autograd.grad(value, z)[0]
        expected_gradient = torch.tensor(
            [
                [0.0621666908, -1.8042422533, -0.6216676235],
                [0.8119281530, -3.2477126122, -0.1903342754],
                [-2.0793266296, -0.0248348713, 0.2483493239],
                [-3.0475666523, 0.4741818905, -0.3729422688],
            ]
        )
        torch.testing.assert_close(
            value.detach(), torch.tensor(3.9829883575), rtol=0, atol=1e-6
        )
        torch.testing.assert_close(
            gradient, expected_gradient, rtol=0, atol=1e-6
        )

    def test_marginal_value_and_gradient_are_deterministic(self) -> None:
        base = torch.tensor(
            [
                [1.0, 0.0, 0.1],
                [0.8, 0.2, 0.0],
                [0.0, 1.0, 0.1],
                [0.1, 0.8, 0.2],
            ],
            requires_grad=True,
        )
        race = torch.tensor([0, 0, 1, 1])
        first = fair_supcon(base, race, 0.2, relation="different")
        first_gradient = torch.autograd.grad(first, base)[0]
        clone = base.detach().clone().requires_grad_(True)
        second = fair_supcon(clone, race, 0.2, relation="different")
        second_gradient = torch.autograd.grad(second, clone)[0]
        self.assertTrue(torch.isfinite(first))
        self.assertGreater(float(first.detach()), 0.0)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(first_gradient, second_gradient, rtol=0, atol=0)

    def test_no_downstream_label_is_needed_for_primary_relation(self) -> None:
        z = torch.eye(4, requires_grad=True)
        race = torch.tensor([0, 1, 0, 1])
        value = fair_supcon(z, race, 0.2, relation="different")
        self.assertTrue(torch.isfinite(value))

    def test_conditional_relation_is_explicitly_separate(self) -> None:
        z = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.1, 0.9],
                [0.0, 1.0],
            ],
            requires_grad=True,
        )
        race = torch.tensor([0, 1, 0, 1])
        outcome = torch.tensor([0, 0, 1, 1])
        marginal = fair_supcon(z, race, 0.2, relation="different")
        conditional = fair_supcon(
            z,
            race,
            0.2,
            relation="same-condition-different",
            condition=outcome,
        )
        self.assertNotEqual(
            float(marginal.detach()), float(conditional.detach())
        )


if __name__ == "__main__":
    unittest.main()
