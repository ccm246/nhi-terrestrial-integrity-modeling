"""Fit reference-condition models using ALL stack variables as predictors.

The stock ``fit_grassland_integrity_parameters`` fits each ecological response
(d02-d19) from the environmental predictors (d20-d39) only. This variant keeps
the same sample, spatial folds, spline/ridge settings, and outputs, but also lets
every other ecological-response band (d02-d19) enter as a continuous predictor.
A response is never used to predict itself.

It wraps the stock module without editing it, so the two runs are directly
comparable. Same command-line arguments:

    python -m scripts.fit_all_variables_models CONFIG.toml SAMPLE.parquet \
        --output-directory outputs/integrity_parameters/<name>_all_variables

Models fitted here are for cross-validated comparison; the apply step expects the
stock predictor set and is not run on them.
"""
from dataclasses import replace

from . import fit_grassland_integrity_parameters as stock

_original_prepare = stock.prepare_reference_condition_data
_original_fit = stock.fit_response_gam
_response_columns: tuple[str, ...] = ()


def _prepare_with_responses(sample_table, configuration, analysis_configuration):
    global _response_columns
    prepared = _original_prepare(sample_table, configuration, analysis_configuration)
    _response_columns = tuple(
        analysis_configuration.columns_with_role(
            sample_table.columns, "response"
        ).values()
    )
    usable = tuple(
        name
        for name in _response_columns
        if sample_table[name].notna().mean() >= configuration.minimum_predictor_coverage
    )
    return replace(
        prepared,
        continuous_predictor_names=(*prepared.continuous_predictor_names, *usable),
    )


def _fit_excluding_self(
    training_table,
    response_name,
    continuous_predictor_names,
    categorical_predictor_name,
    imputation_values,
    configuration,
    analysis_configuration,
):
    return _original_fit(
        training_table,
        response_name,
        tuple(n for n in continuous_predictor_names if n != response_name),
        categorical_predictor_name,
        imputation_values,
        configuration,
        analysis_configuration,
    )


stock.prepare_reference_condition_data = _prepare_with_responses
stock.fit_response_gam = _fit_excluding_self

if __name__ == "__main__":
    stock.main()
