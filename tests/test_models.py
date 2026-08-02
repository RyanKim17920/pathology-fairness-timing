from __future__ import annotations

import unittest

import torch

from pathology_fairness.models import (
    PostHocDANNHead,
    StageAdapter,
    dann_loss,
    gradient_reverse,
)


class ModelTests(unittest.TestCase):
    def test_adapter_is_reproducible_and_does_not_consume_global_rng(self) -> None:
        torch.manual_seed(91)
        state = torch.random.get_rng_state()
        first = StageAdapter(8, 6, 4, init_seed=7)
        torch.testing.assert_close(torch.random.get_rng_state(), state, rtol=0, atol=0)
        second = StageAdapter(8, 6, 4, init_seed=7)
        for left, right in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        output = first(torch.randn(5, 8))
        torch.testing.assert_close(
            output.norm(dim=1), torch.ones(5), rtol=0, atol=1e-6
        )

    def test_gradient_reverse_changes_only_the_backward_direction(self) -> None:
        value = torch.tensor([1.0, 2.0], requires_grad=True)
        output = gradient_reverse(value, 0.5)
        torch.testing.assert_close(output, value)
        output.sum().backward()
        torch.testing.assert_close(value.grad, torch.tensor([-0.5, -0.5]))

    def test_posthoc_head_and_loss_support_multiple_sensitive_attributes(self) -> None:
        head = PostHocDANNHead(
            input_dim=8,
            hidden_dim=5,
            sensitive_classes={"race": 3, "sex": 2},
            dropout=0.0,
        )
        output = head(torch.randn(6, 8), reversal_weight=0.2)
        self.assertEqual(output.task_logits.shape, (6,))
        self.assertEqual(output.sensitive_logits["race"].shape, (6, 3))
        self.assertEqual(output.sensitive_logits["sex"].shape, (6, 2))
        loss = dann_loss(
            output,
            torch.tensor([0, 1, 0, 1, 0, 1]),
            {
                "race": torch.tensor([0, 1, 2, 0, -1, 1]),
                "sex": torch.tensor([0, 1, 0, 1, 0, -1]),
            },
            sensitive_weight=0.1,
        )
        loss.total.backward()
        self.assertTrue(torch.isfinite(loss.total))
        self.assertGreater(float(loss.sensitive.detach()), 0.0)


if __name__ == "__main__":
    unittest.main()
