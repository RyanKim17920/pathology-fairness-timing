#!/usr/bin/env python3

from pathlib import Path
import re
import unittest


class SubmissionFileTests(unittest.TestCase):
    def test_driver_is_exactly_one_named_gpu_task(self) -> None:
        package = Path(__file__).parent
        driver = (package / "serial_audit.sbatch").read_text()
        self.assertEqual(driver.count("#SBATCH --job-name=main_1gpu"), 1)
        self.assertEqual(driver.count("#SBATCH --gpus-per-task=1"), 1)
        self.assertEqual(driver.count("#SBATCH --ntasks=1"), 1)
        self.assertEqual(driver.count("#SBATCH --mem=64G"), 1)
        self.assertEqual(driver.count("#SBATCH --comment=matched_cancer_rep_audit_20260801"), 1)
        self.assertNotRegex(driver, r"(?im)^#SBATCH\s+--array")
        self.assertNotRegex(driver, r"(?im)^#SBATCH\s+--dependency")
        self.assertNotIn("srun", driver)
        self.assertEqual(
            driver.count(
                '"$PY" -m tools.matched_cancer_representation_audit_20260801.pipeline run'
            ),
            1,
        )

    def test_submitter_has_queue_lock_preflight_and_one_sbatch(self) -> None:
        package = Path(__file__).parent
        submit = (package / "safe_submit.sh").read_text()
        self.assertIn("fixed48_execution/submission/safe_submit.lock", submit)
        self.assertIn("-t PENDING,RUNNING", submit)
        self.assertIn('"$job_name" == main_1gpu', submit)
        self.assertIn("pipeline preflight", submit)
        self.assertEqual(submit.count('"$SBATCH" --parsable'), 1)
        self.assertNotIn("--array", submit)
        self.assertNotIn("--dependency", submit)
        self.assertRegex(submit, re.compile(r"OUTPUT_ROOT=.*/attempt_01"))


if __name__ == "__main__":
    unittest.main()
