"""Model roles must reach actual stage clients and cannot degrade to templates."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig, validate_config
from researchclaw.llm.routing import prepare_routed_client, resolve_stage_config, stage_model
from researchclaw.pipeline import executor
from researchclaw.pipeline.stages import Stage, StageStatus

EXECUTION_STAGES = (4, 6, 10, 11, 12, 13, 17, 19, 21, 22)


def config_data():
    return {
        "project": {"name": "model-routing", "mode": "docs-first"},
        "research": {"topic": "bounded test"},
        "runtime": {"timezone": "UTC"},
        "notifications": {"channel": "console"},
        "knowledge_base": {"root": "docs/kb", "backend": "markdown"},
        "llm": {
            "provider": "acp", "primary_model": "gpt-6-astra",
            "acp": {"agent": "codex", "model_routing": True,
                    "reasoning_effort": "ultra", "execution_model": "gpt-6.1-sol",
                    "stage_models": {stage: "gpt-6.1-sol" for stage in EXECUTION_STAGES}},
        },
        "experiment": {"mode": "simulated"},
    }


def config():
    return RCConfig.from_dict(config_data(), check_paths=False)


class SelectedClient:
    def __init__(self, cfg):
        self.config = cfg.llm
        self.session_selection = {}
        self.selection_error = None

    def prepare(self):
        self.session_selection = {
            "verified": True, "selected_model": self.config.primary_model,
            "selected_reasoning_effort": "ultra",
        }


@pytest.mark.parametrize("stage", list(Stage))
def test_all_stage_roles_and_original_config_unchanged(stage):
    original = config()
    resolved = resolve_stage_config(original, int(stage), "run-one")
    expected = "gpt-6.1-sol" if int(stage) in EXECUTION_STAGES else "gpt-6-astra"
    assert resolved.llm.primary_model == expected
    assert resolved.llm.acp.reasoning_effort == "ultra"
    assert original.llm.primary_model == "gpt-6-astra"
    assert original.llm.acp.session_name == "researchclaw"


def test_sessions_isolate_runs_stages_and_figure_execution():
    cfg = config()
    main = resolve_stage_config(cfg, 14, "one")
    drawing = resolve_stage_config(cfg, 14, "one", execution=True, purpose="figure-code")
    other_run = resolve_stage_config(cfg, 14, "two")
    code = resolve_stage_config(cfg, 10, "one")
    assert drawing.llm.primary_model == "gpt-6.1-sol"
    assert len({c.llm.acp.session_name for c in (main, drawing, other_run, code)}) == 4
    assert main == resolve_stage_config(cfg, 14, "one")
    discussion = resolve_stage_config(cfg, 10, "one", purpose="copilot", model=cfg.llm.primary_model)
    assert discussion.llm.primary_model == "gpt-6-astra"


def test_legacy_config_unchanged_and_round_trip():
    from researchclaw.config import _parse_llm_config
    cfg = config()
    assert _parse_llm_config(cfg.to_dict()["llm"]) == cfg.llm
    legacy = replace(cfg, llm=replace(cfg.llm, acp=replace(cfg.llm.acp, model_routing=False)))
    assert resolve_stage_config(legacy, 10, "one") is legacy


@pytest.mark.parametrize("key,value", [
    ("stage_models", {0: "a"}), ("stage_models", {24: "a"}),
    ("stage_models", {True: "a"}), ("stage_models", {"wrong": "a"}),
    ("stage_models", {10: ""}), ("stage_models", ["a"]),
    ("stage_models", {10: "a", "10": "b"}),
    ("reasoning_effort", "automatic"), ("reasoning_effort", ""),
    ("model_routing", "false"), ("execution_model", 123),
])
def test_invalid_routing_config_rejected(key, value):
    data = config_data()
    data["llm"]["acp"][key] = value
    assert not validate_config(data, check_paths=False).ok


def test_stage_models_cannot_be_silently_ignored():
    data = config_data()
    data["llm"]["acp"]["model_routing"] = False
    assert not validate_config(data, check_paths=False).ok
    with pytest.raises(ValueError):
        stage_model(config(), 24)


def test_stage_receives_effective_config_and_selected_client(tmp_path, monkeypatch):
    cfg = config()
    stage = Stage.CODE_GENERATION
    monkeypatch.setattr(executor, "create_llm_client", SelectedClient)
    monkeypatch.setitem(executor.CONTRACTS, stage, replace(executor.CONTRACTS[stage], input_files=(), output_files=()))
    seen = []

    def execute(stage_dir, run_dir, stage_config, adapters, *, llm, **kwargs):
        seen.append((stage_config.llm.primary_model, llm.config.primary_model))
        return executor.StageResult(stage=stage, status=StageStatus.DONE, artifacts=())

    monkeypatch.setitem(executor._STAGE_EXECUTORS, stage, execute)
    result = executor.execute_stage(stage, run_dir=tmp_path, run_id="one", config=cfg, adapters=AdapterBundle())
    assert result.status == StageStatus.DONE
    assert seen == [("gpt-6.1-sol", "gpt-6.1-sol")]
    evidence = json.loads((tmp_path / "stage-10/llm_selection.json").read_text())
    assert evidence["verified"] and evidence["selected_model"] == "gpt-6.1-sol"


@pytest.mark.parametrize("late_failure", [False, True])
def test_unavailable_model_cannot_become_template_or_approval(tmp_path, monkeypatch, late_failure):
    cfg = config()
    stage = Stage.LITERATURE_SCREEN
    monkeypatch.setitem(executor.CONTRACTS, stage, replace(executor.CONTRACTS[stage], input_files=(), output_files=()))
    client = SelectedClient(cfg)
    if not late_failure:
        def failed_prepare():
            raise RuntimeError("model unavailable")
        client.prepare = failed_prepare
    monkeypatch.setattr(executor, "create_llm_client", lambda c: client)
    ran = []

    def swallowed_failure(stage_dir, run_dir, config, adapters, *, llm, **kwargs):
        ran.append(True)
        llm.selection_error = "model changed unexpectedly"
        return executor.StageResult(stage=stage, status=StageStatus.DONE, artifacts=())

    monkeypatch.setitem(executor._STAGE_EXECUTORS, stage, swallowed_failure)
    result = executor.execute_stage(stage, run_dir=tmp_path, run_id="one", config=cfg, adapters=AdapterBundle())
    assert result.status == StageStatus.FAILED
    assert bool(ran) == late_failure
    assert not json.loads((tmp_path / "stage-05/llm_selection.json").read_text())["verified"]


def test_review_records_writer_sol_and_judge_astra(tmp_path):
    from researchclaw.pipeline.stage_impls._review_publish import _build_reviewer_or_generator
    cfg = config()
    judge = SelectedClient(resolve_stage_config(cfg, 18, "one"))
    client, author, reviewer = _build_reviewer_or_generator(cfg, judge, tmp_path)
    assert client is judge
    assert (author, reviewer) == ("gpt-6.1-sol", "gpt-6-astra")
    path = tmp_path / "stage-19/llm_selection.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"verified": True, "selected_model": "recorded-writer"}))
    assert _build_reviewer_or_generator(cfg, judge, tmp_path)[1] == "gpt-6.1-sol"
    assert _build_reviewer_or_generator(cfg, judge, tmp_path, writer_stage=19)[1] == "recorded-writer"
    (path.parent / "paper_author.json").write_text(json.dumps({"source_stage": 17}))
    assert _build_reviewer_or_generator(cfg, judge, tmp_path, writer_stage=19)[1] == "gpt-6.1-sol"


def test_unverified_selection_is_saved_as_failure(tmp_path):
    client = SimpleNamespace(prepare=lambda: None, session_selection={})
    path = tmp_path / "selection.json"
    with pytest.raises(RuntimeError, match="not verified"):
        prepare_routed_client(client, config(), path)
    assert not json.loads(path.read_text())["verified"]


def test_late_nested_failure_invalidates_saved_selection(tmp_path):
    from researchclaw.llm.routing import check_selection_error
    cfg = config()
    client = SelectedClient(cfg)
    path = tmp_path / "figure_selection.json"
    prepare_routed_client(client, cfg, path)
    client.selection_error = "selected wrong reasoning effort"
    with pytest.raises(RuntimeError, match="wrong reasoning effort"):
        check_selection_error(client, path)
    assert not json.loads(path.read_text())["verified"]
