"""Robotics & control domain prompt adapter.

Provides domain-specific prompt blocks for robotics experiments
(control policies, RL-based manipulation, locomotion, sim-to-real).
"""

from __future__ import annotations

from typing import Any

from researchclaw.domains.prompt_adapter import PromptAdapter, PromptBlocks


class RoboticsPromptAdapter(PromptAdapter):
    """Adapter for robotics and control domains."""

    def get_code_generation_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        domain = self.domain
        libs = (
            ", ".join(domain.core_libraries)
            if domain.core_libraries
            else "gymnasium, stable-baselines3, torch"
        )

        return PromptBlocks(
            compute_budget=domain.compute_budget_guidance
            or self._default_compute_budget(),
            dataset_guidance=domain.dataset_guidance
            or self._default_dataset_guidance(),
            hp_reporting=domain.hp_reporting_guidance
            or self._default_hp_reporting(),
            code_generation_hints=domain.code_generation_hints
            or self._default_code_hints(),
            output_format_guidance=self._output_format(),
        )

    def get_experiment_design_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        domain = self.domain

        design_context = (
            f"This is a **{domain.display_name}** experiment.\n"
            f"Paradigm: {domain.experiment_paradigm}\n\n"
            "Key principles for robotics experiments:\n"
            "1. Use standardized environments (Gymnasium, MuJoCo) for "
            "reproducibility\n"
            "2. Report mean ± std of episode return over multiple seeds\n"
            "3. Include learning curves (return vs. training steps)\n"
            "4. Compare against established RL baselines (PPO, SAC, TD3)\n"
            "5. Report success rate for goal-conditioned tasks\n"
        )

        if domain.standard_baselines:
            design_context += (
                f"\nStandard baselines: "
                f"{', '.join(domain.standard_baselines)}\n"
            )

        stats = (
            ", ".join(domain.statistical_tests)
            if domain.statistical_tests
            else "paired t-test"
        )

        return PromptBlocks(
            experiment_design_context=design_context,
            statistical_test_guidance=(
                f"Use {stats} across random seeds to assess significance. "
                "Report results over at least 5 seeds. Include confidence "
                "intervals on learning curves."
            ),
        )

    def get_result_analysis_blocks(self, context: dict[str, Any]) -> PromptBlocks:
        return PromptBlocks(
            result_analysis_hints=self.domain.result_analysis_hints
            or (
                "Robotics result analysis:\n"
                "- Episode return: mean ± std across seeds and evaluation "
                "episodes\n"
                "- Success rate: fraction of episodes reaching goal\n"
                "- Sample efficiency: return at fixed training step count\n"
                "- Learning curves: smoothed return vs. environment steps\n"
                "- Wall-clock time if comparing algorithm efficiency"
            ),
            statistical_test_guidance=(
                "Use paired t-test or Welch's t-test across seeds. "
                "Report 95% confidence intervals on all metrics."
            ),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _default_code_hints(self) -> str:
        return (
            "Robotics/control code:\n"
            "1. Create Gymnasium environment (use built-in envs or define "
            "custom wrappers)\n"
            "2. Implement or instantiate RL agent (PPO, SAC, TD3 via "
            "stable-baselines3)\n"
            "3. Train for a fixed number of environment steps\n"
            "4. Evaluate over 100 episodes, record returns\n"
            "5. Repeat across multiple seeds (>= 5)\n"
            "6. Output results.json with per-seed evaluation metrics\n"
        )

    def _default_compute_budget(self) -> str:
        return (
            "Time budget for robotics experiments:\n"
            "- Use simple envs (CartPole, Pendulum) for fast iteration\n"
            "- Limit training to 100k-500k steps for benchmarks\n"
            "- MuJoCo envs are heavier: reduce training budget\n"
            "- Evaluate over 100 episodes per seed for stable metrics"
        )

    def _default_dataset_guidance(self) -> str:
        return (
            "Robotics experiments use simulation environments:\n"
            "- Use Gymnasium built-in envs (CartPole, Pendulum, HalfCheetah)\n"
            "- Define custom environments via Gymnasium API if needed\n"
            "- Do NOT download external datasets\n"
            "- All training data is generated through environment interaction"
        )

    def _default_hp_reporting(self) -> str:
        return (
            "Report training hyperparameters:\n"
            "HYPERPARAMETERS: {'env': ..., 'algorithm': ..., 'lr': ..., "
            "'gamma': ..., 'total_timesteps': ..., 'n_seeds': ..., "
            "'eval_episodes': ...}"
        )

    def _output_format(self) -> str:
        return (
            "Output robotics experiment results to results.json:\n"
            '{"conditions": {"algorithm": {"mean_return": 195.2, '
            '"std_return": 12.3, "success_rate": 0.95}},\n'
            ' "metadata": {"domain": "robotics_control", "env": "...", '
            '"total_timesteps": ...}}'
        )
