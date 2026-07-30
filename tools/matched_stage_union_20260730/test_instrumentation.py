#!/usr/bin/env python3

import copy
import math
import unittest

import torch

from tools.matched_stage_union_20260730.instrumentation import (
    estimated_step_flops,
    forward_flop_meter,
    gradient_dose_diagnostic,
    training_schedule_state,
    uses_fino_prototypes,
)


class TinyTwoHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trunk = torch.nn.Linear(4, 3)
        self.main_head = torch.nn.Linear(3, 1)
        self.fair_head = torch.nn.Linear(3, 1)

    def losses(
        self, x: torch.Tensor, main_target: torch.Tensor, fair_target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shared = torch.tanh(self.trunk(x))
        main = torch.nn.functional.mse_loss(self.main_head(shared), main_target)
        fair = torch.nn.functional.mse_loss(self.fair_head(shared), fair_target)
        return main, fair


def run_two_steps(
    initial_model: TinyTwoHead,
    *,
    dose_logging: bool,
) -> tuple[TinyTwoHead, torch.optim.AdamW, list[int], list[dict[str, float | bool]]]:
    model = copy.deepcopy(initial_model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.04)
    x = torch.tensor(
        [
            [0.2, -0.4, 0.8, 0.1],
            [0.7, 0.3, -0.2, 0.5],
            [-0.1, 0.6, 0.4, -0.7],
        ]
    )
    main_target = torch.tensor([[0.3], [-0.2], [0.7]])
    fair_target = torch.tensor([[-0.4], [0.6], [0.1]])
    train_flops = 0
    measured_flops = None
    flop_history: list[int] = []
    schedules: list[dict[str, float | bool]] = []

    for step in range(2):
        schedule = training_schedule_state(
            train_flops=train_flops,
            max_train_flops=10_000_000,
            examples_seen=step * len(x),
            max_train_samples=6,
            warmup_train_samples=1,
            base_lr=0.01,
            min_lr=0.001,
            warmup_fraction=0.1,
            freeze_last_layer_fraction=0.01,
            lr_key=None,
            reg_key=None,
        )
        schedules.append(schedule)
        for group in optimizer.param_groups:
            group["lr"] = float(schedule["lr"])
            group["weight_decay"] = float(schedule["wd"])

        meter = forward_flop_meter(measured_flops is None)
        with meter:
            main_loss, fair_loss = model.losses(x, main_target, fair_target)
            total_loss = main_loss + 0.1 * fair_loss

        before = [
            None if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in model.parameters()
        ]
        if dose_logging:
            dose = gradient_dose_diagnostic(
                main_loss, 0.1 * fair_loss, model.trunk.parameters()
            )
            self_numeric = [
                value for value in dose.values() if not isinstance(value, bool)
            ]
            if not all(math.isfinite(value) for value in self_numeric):
                raise AssertionError("nonfinite diagnostic in regression fixture")
            if float(dose["dose_fair_grad_norm"]) <= 0:
                raise AssertionError("zero fair gradient in regression fixture")
        after = [
            None if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in model.parameters()
        ]
        for old, new in zip(before, after):
            if old is None or new is None:
                if old is not new:
                    raise AssertionError("diagnostic changed a None gradient")
            else:
                torch.testing.assert_close(old, new, rtol=0, atol=0)

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        if measured_flops is None:
            measured_flops = estimated_step_flops(meter)
        train_flops += measured_flops
        flop_history.append(measured_flops)
    return model, optimizer, flop_history, schedules


class InstrumentationNeutralityTest(unittest.TestCase):
    def test_primary_faircon_allocates_no_unused_random_prototypes(self) -> None:
        self.assertFalse(uses_fino_prototypes("contrastive", True))
        self.assertFalse(uses_fino_prototypes("dann", True))
        self.assertFalse(uses_fino_prototypes("fino", False))
        self.assertTrue(uses_fino_prototypes("fino", True))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_primary_setup_preserves_next_cuda_random_draw(self) -> None:
        torch.cuda.manual_seed_all(418)
        plain_next = torch.randn(16, device="cuda")
        torch.cuda.manual_seed_all(418)
        if uses_fino_prototypes("contrastive", True):
            torch.randn(5, 384, device="cuda")
        fair_next = torch.randn(16, device="cuda")
        torch.testing.assert_close(plain_next, fair_next, rtol=0, atol=0)

    def test_dose_logging_is_schedule_and_update_neutral(self) -> None:
        torch.manual_seed(90210)
        initial = TinyTwoHead()
        off_model, off_optimizer, off_flops, off_schedules = run_two_steps(
            initial, dose_logging=False
        )
        on_model, on_optimizer, on_flops, on_schedules = run_two_steps(
            initial, dose_logging=True
        )
        self.assertEqual(off_flops, on_flops)
        self.assertEqual(off_schedules, on_schedules)
        for off, on in zip(off_model.parameters(), on_model.parameters()):
            torch.testing.assert_close(off, on, rtol=0, atol=0)

        off_state = off_optimizer.state_dict()
        on_state = on_optimizer.state_dict()
        self.assertEqual(off_state["param_groups"], on_state["param_groups"])
        self.assertEqual(off_state["state"].keys(), on_state["state"].keys())
        for key in off_state["state"]:
            self.assertEqual(
                off_state["state"][key].keys(), on_state["state"][key].keys()
            )
            for field in off_state["state"][key]:
                off_value = off_state["state"][key][field]
                on_value = on_state["state"][key][field]
                if torch.is_tensor(off_value):
                    torch.testing.assert_close(off_value, on_value, rtol=0, atol=0)
                else:
                    self.assertEqual(off_value, on_value)


if __name__ == "__main__":
    unittest.main()
