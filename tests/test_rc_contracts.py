import re

import pytest

from researchclaw.pipeline.contracts import CONTRACTS, StageContract
from researchclaw.pipeline.stages import GATE_STAGES, STAGE_SEQUENCE, Stage


def test_contracts_dict_has_exactly_23_entries():
    assert len(CONTRACTS) == 23


def test_every_stage_has_matching_contract_entry():
    assert set(CONTRACTS.keys()) == set(Stage)


@pytest.mark.parametrize("stage", STAGE_SEQUENCE)
def test_each_stage_member_resolves_to_stage_contract(stage: Stage):
    assert isinstance(CONTRACTS[stage], StageContract)


@pytest.mark.parametrize("stage,contract", tuple(CONTRACTS.items()))
def test_contract_stage_field_matches_dict_key(stage: Stage, contract: StageContract):
    assert contract.stage is stage


@pytest.mark.parametrize("contract", tuple(CONTRACTS.values()))
def test_output_files_is_non_empty_for_all_contracts(contract: StageContract):
    assert contract.output_files


@pytest.mark.parametrize("stage,contract", tuple(CONTRACTS.items()))
def test_error_code_starts_with_e_and_contains_stage_number(
    stage: Stage, contract: StageContract
):
    assert contract.error_code.startswith("E")
    assert f"{int(stage):02d}" in contract.error_code
    assert re.match(r"^E\d{2}_[A-Z0-9_]+$", contract.error_code)


@pytest.mark.parametrize("contract", tuple(CONTRACTS.values()))
def test_max_retries_is_non_negative_for_all_contracts(contract: StageContract):
    assert contract.max_retries >= 0


def test_gate_stages_have_expected_max_retries():
    assert CONTRACTS[Stage.LITERATURE_SCREEN].max_retries == 0
    assert CONTRACTS[Stage.EXPERIMENT_DESIGN].max_retries == 0
    assert CONTRACTS[Stage.QUALITY_GATE].max_retries == 0


@pytest.mark.parametrize("stage", tuple(GATE_STAGES))
def test_gate_stage_contracts_are_never_retried(stage: Stage):
    assert CONTRACTS[stage].max_retries == 0


def test_topic_init_contract_has_expected_input_output_files():
    contract = CONTRACTS[Stage.TOPIC_INIT]

    assert contract.input_files == ()
    assert contract.output_files == ("goal.md", "hardware_profile.json")


def test_export_publish_contract_has_expected_outputs():
    contract = CONTRACTS[Stage.EXPORT_PUBLISH]

    assert contract.output_files == ("paper_final.md", "code/")


@pytest.mark.parametrize("contract", tuple(CONTRACTS.values()))
def test_dod_is_non_empty_string_for_all_contracts(contract: StageContract):
    assert isinstance(contract.dod, str)
    assert contract.dod.strip()


@pytest.mark.parametrize("contract", tuple(CONTRACTS.values()))
def test_input_files_is_tuple_of_strings(contract: StageContract):
    assert isinstance(contract.input_files, tuple)
    assert all(isinstance(path, str) and path for path in contract.input_files)


@pytest.mark.parametrize("contract", tuple(CONTRACTS.values()))
def test_output_files_is_tuple_of_strings(contract: StageContract):
    assert isinstance(contract.output_files, tuple)
    assert all(isinstance(path, str) and path for path in contract.output_files)


def test_error_codes_are_unique_across_contracts():
    all_codes = [contract.error_code for contract in CONTRACTS.values()]
    assert len(all_codes) == len(set(all_codes))


def test_contracts_follow_stage_sequence_order():
    assert tuple(CONTRACTS.keys()) == STAGE_SEQUENCE


@pytest.mark.parametrize("stage", STAGE_SEQUENCE)
def test_contract_stage_int_matches_stage_enum_value(stage: Stage):
    assert int(CONTRACTS[stage].stage) == int(stage)


def test_stage_output_satisfies_next_input():
    # Valida la CONTINUIDAD declarativa output->input a lo largo de STAGE_SEQUENCE:
    # recorriendo los stages en orden, cada archivo de input_files debe haber sido
    # declarado como output_files por algun stage ANTERIOR.
    #
    # Notas de diseño:
    # (a) El estado del pipeline fluye via un run_dir compartido (los stages leen/escriben
    #     archivos en ese directorio), no por paso explicito de argumentos entre stages.
    #     Por eso este test verifica la COHERENCIA DECLARATIVA de los contratos, no el
    #     wiring runtime real.
    # (b) Stages con input_files=() son casos aceptados: TOPIC_INIT (entrada del pipeline,
    #     no consume nada previo) y KNOWLEDGE_ARCHIVE (opera sobre run_dir, no sobre
    #     archivos declarados).
    # (c) Si apareciera un input no cubierto por outputs previos, NO se debe parchear el
    #     pipeline: documentarlo aqui en ALLOWED_UNSATISFIED con el motivo. Hoy esta VACIO
    #     porque la cadena no tiene gaps.
    ALLOWED_UNSATISFIED: set[tuple[Stage, str]] = set()

    produced: set[str] = set()
    for stage in STAGE_SEQUENCE:
        for path in CONTRACTS[stage].input_files:
            if (stage, path) in ALLOWED_UNSATISFIED:
                continue
            assert path in produced, (
                f"{stage.name}: input file {path!r} no fue producido por ningun stage previo"
            )
        produced.update(CONTRACTS[stage].output_files)
