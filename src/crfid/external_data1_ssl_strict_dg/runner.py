"""Source-only fold runner and final-model builder.

Every unit of work is one ``(treatment, objective, fold, seed)`` cell evaluated on the
fold's held-out *source* domain. No unit in this module can read the target: it only
ever indexes the P1-P3 corpus.

Self-supervised trunks are cached by a digest of ``(corpus digests, pretrain config,
seed)``. The cache is a pure compute optimisation -- a cache hit is bit-identical to a
recomputation, which the Data1-only treatment relies on because its trunk does not
depend on the Paper4 fold.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from . import carrier, data1_corpus, metrics, model as model_module, objectives
from . import paths, pretrain, readout, source_data, supervised, treatments
from .integrity import array_sha256, canonical_json_sha256

TRUNK_CACHE_RELATIVE = ("10_logs", "ssl_trunk_cache")


@dataclass
class UnitResult:
    treatment_id: str
    objective_id: str | None
    fold_id: str
    seed: int
    selected_inner_epoch: int
    inner_validation_macro_f1: float
    head_metrics: dict[str, Any]
    ncm_metrics: dict[str, Any]
    resources: dict[str, Any]
    state_sha256: str

    def row(self) -> dict[str, Any]:
        return {
            "treatment_id": self.treatment_id,
            "objective_id": self.objective_id or "",
            "fold_id": self.fold_id,
            "seed": self.seed,
            "selected_inner_epoch": self.selected_inner_epoch,
            "inner_validation_macro_f1": self.inner_validation_macro_f1,
            "outer_held_accuracy": self.head_metrics["accuracy"],
            "outer_held_macro_f1": self.head_metrics["macro_f1"],
            "outer_held_worst_class_recall": self.head_metrics["worst_class_recall"],
            "outer_held_zero_recall_class_count": self.head_metrics["zero_recall_class_count"],
            "outer_held_ncm_accuracy": self.ncm_metrics["accuracy"],
            "outer_held_ncm_macro_f1": self.ncm_metrics["macro_f1"],
            "state_sha256": self.state_sha256,
            "ssl_seconds": self.resources.get("ssl_seconds", 0.0),
            "supervised_seconds": self.resources.get("supervised_seconds", 0.0),
            "ssl_optimizer_steps": self.resources.get("ssl_optimizer_steps", 0),
            "supervised_optimizer_steps": self.resources.get("supervised_optimizer_steps", 0),
        }


def _trunk_cache_path(key: str):
    return paths.branch_output(*TRUNK_CACHE_RELATIVE, f"{key}.pt")


def _build_corpora(
    treatment: treatments.Treatment, paper4_rows: np.ndarray
) -> dict[str, pretrain.SslCorpus]:
    corpus = source_data.load_source_corpus()
    built: dict[str, pretrain.SslCorpus] = {}
    if pretrain.PAPER4_CORPUS in treatment.corpus_fractions:
        built[pretrain.PAPER4_CORPUS] = pretrain.build_ssl_corpus(
            pretrain.PAPER4_CORPUS, corpus.signals[paper4_rows]
        )
    if pretrain.DATA1_CORPUS in treatment.corpus_fractions:
        built[pretrain.DATA1_CORPUS] = pretrain.build_ssl_corpus(
            pretrain.DATA1_CORPUS, data1_corpus.load_corpus()
        )
    return built


def obtain_trunk(
    treatment: treatments.Treatment,
    objective_id: str,
    paper4_rows: np.ndarray,
    seed: int,
    pretrain_config: pretrain.PretrainConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(trunk_state, resources)``, reusing an identical cached trunk."""

    corpora = _build_corpora(treatment, paper4_rows)
    corpus_digests = {
        name: array_sha256(corpus.difference) for name, corpus in sorted(corpora.items())
    }
    key = canonical_json_sha256(
        {
            "corpus_digests": corpus_digests,
            "pretrain": pretrain_config.as_record(),
            "seed": int(seed),
            "objective_id": objective_id,
        }
    )
    cached = paths.BRANCH_OUTPUT_ROOT.joinpath(*TRUNK_CACHE_RELATIVE, f"{key}.pt")
    if cached.is_file():
        payload = torch.load(cached, map_location="cpu", weights_only=False)
        return payload["trunk_state"], {
            "ssl_seconds": float(payload["wall_clock_seconds"]),
            "ssl_optimizer_steps": int(payload["total_steps"]),
            "ssl_trainable_parameters": int(payload["trainable_parameters"]),
            "ssl_trunk_sha256": payload["trunk_sha256"],
            "ssl_final_epoch_loss": float(payload["epoch_losses"][-1]),
            "ssl_cache_hit": True,
            "ssl_domain_accuracy": payload.get("final_domain_accuracy"),
        }

    result = pretrain.pretrain(corpora, pretrain_config, seed=seed)
    torch.save(
        {
            "trunk_state": result.trunk_state,
            "trunk_sha256": result.trunk_sha256,
            "epoch_losses": list(result.epoch_losses),
            "total_steps": result.total_steps,
            "wall_clock_seconds": result.wall_clock_seconds,
            "trainable_parameters": result.trainable_parameters,
            "final_domain_accuracy": result.final_domain_accuracy,
            "cache_key": key,
        },
        _trunk_cache_path(key),
    )
    return result.trunk_state, {
        "ssl_seconds": result.wall_clock_seconds,
        "ssl_optimizer_steps": result.total_steps,
        "ssl_trainable_parameters": result.trainable_parameters,
        "ssl_trunk_sha256": result.trunk_sha256,
        "ssl_final_epoch_loss": float(result.epoch_losses[-1]),
        "ssl_cache_hit": False,
        "ssl_domain_accuracy": result.final_domain_accuracy,
    }


def run_fold_unit(
    *,
    treatment: treatments.Treatment,
    objective_id: str | None,
    fold_id: str,
    seed: int,
    pretrain_config: pretrain.PretrainConfig | None,
    supervised_config: supervised.SupervisedConfig,
) -> UnitResult:
    """Train one source fold unit and score it on the held-out source domain."""

    corpus = source_data.load_source_corpus()
    partitions = source_data.load_fold_partitions()[fold_id]
    inner_train = partitions["inner_train"]
    inner_validation = partitions["inner_validation"]
    outer_held = partitions["outer_held"]
    outer_development = np.sort(np.concatenate([inner_train, inner_validation]))

    inner_standardizer = carrier.fit_supervised_standardizer(corpus.signals[inner_train])
    inner_train_inputs = carrier.prepare_supervised(corpus.signals[inner_train], inner_standardizer)
    inner_validation_inputs = carrier.prepare_supervised(
        corpus.signals[inner_validation], inner_standardizer
    )

    resources: dict[str, Any] = {"ssl_seconds": 0.0, "ssl_optimizer_steps": 0}
    trunk_state = None
    if treatment.uses_ssl:
        if pretrain_config is None or objective_id is None:
            raise ValueError(f"{treatment.treatment_id} requires an SSL configuration")
        # The fold's unlabelled Paper4 corpus is its ``inner_train`` partition only, so
        # neither the inner-validation rows nor the held-out domain are ever seen -- not
        # even without labels -- before the selection they inform.
        trunk_state, ssl_resources = obtain_trunk(
            treatment, objective_id, inner_train, seed, pretrain_config
        )
        resources.update(ssl_resources)

    started = time.perf_counter()
    inner = supervised.train_with_early_stopping(
        train_inputs=inner_train_inputs,
        train_labels=corpus.labels[inner_train],
        validation_inputs=inner_validation_inputs,
        validation_labels=corpus.labels[inner_validation],
        seed=seed,
        config=supervised_config,
        trunk_state=trunk_state,
        stage_offset=supervised.INNER_STAGE_OFFSET,
    )

    # Canonical outer refit: a fresh model from the same seed, trained on the full
    # development partition for exactly the selected number of epochs.
    outer_standardizer = carrier.fit_supervised_standardizer(corpus.signals[outer_development])
    outer_train_inputs = carrier.prepare_supervised(
        corpus.signals[outer_development], outer_standardizer
    )
    held_inputs = carrier.prepare_supervised(corpus.signals[outer_held], outer_standardizer)
    outer = supervised.train_fixed_epochs(
        train_inputs=outer_train_inputs,
        train_labels=corpus.labels[outer_development],
        epochs=inner.selected_epoch,
        seed=seed,
        config=supervised_config,
        trunk_state=trunk_state,
        stage_offset=supervised.OUTER_STAGE_OFFSET,
    )
    resources["supervised_seconds"] = time.perf_counter() - started
    resources["supervised_optimizer_steps"] = inner.optimizer_steps + outer.optimizer_steps
    resources["inner_epochs_executed"] = inner.epochs_executed
    resources["outer_refit_epochs"] = outer.epochs_executed
    resources["trainable_parameters"] = model_module.CANONICAL_PARAMETER_COUNT_SEVEN_CLASS

    scored = model_module.initialize_backbone(seed)
    scored.load_state_dict(outer.state)
    logits = model_module.logits_numpy(scored, held_inputs)
    head_metrics = metrics.classification_metrics(
        corpus.labels[outer_held], metrics.argmax_predictions(logits)
    )
    head_metrics["mean_prediction_entropy"] = metrics.prediction_entropy(metrics.softmax(logits))
    development_logits = model_module.logits_numpy(scored, outer_train_inputs)
    head_metrics["outer_development_accuracy"] = metrics.classification_metrics(
        corpus.labels[outer_development], metrics.argmax_predictions(development_logits)
    )["accuracy"]

    prototypes = readout.NearestClassMean.fit(
        model_module.embed_numpy(scored, outer_train_inputs),
        corpus.labels[outer_development],
        fitted_on=f"{fold_id}_outer_development",
    )
    ncm_metrics = metrics.classification_metrics(
        corpus.labels[outer_held],
        prototypes.predict(model_module.embed_numpy(scored, held_inputs)),
    )

    return UnitResult(
        treatment_id=treatment.treatment_id,
        objective_id=objective_id,
        fold_id=fold_id,
        seed=seed,
        selected_inner_epoch=inner.selected_epoch,
        inner_validation_macro_f1=inner.best_validation_macro_f1,
        head_metrics=head_metrics,
        ncm_metrics=ncm_metrics,
        resources=resources,
        state_sha256=outer.state_sha256,
    )


def build_final_model(
    *,
    treatment: treatments.Treatment,
    objective_id: str | None,
    seed: int,
    epochs: int,
    pretrain_config: pretrain.PretrainConfig | None,
    supervised_config: supervised.SupervisedConfig,
) -> dict[str, Any]:
    """Train the final all-source model for one seed and return its frozen payload."""

    corpus = source_data.load_source_corpus()
    rows = source_data.all_source_indices()
    standardizer = carrier.fit_supervised_standardizer(corpus.signals[rows])
    train_inputs = carrier.prepare_supervised(corpus.signals[rows], standardizer)

    resources: dict[str, Any] = {"ssl_seconds": 0.0, "ssl_optimizer_steps": 0}
    trunk_state = None
    if treatment.uses_ssl:
        if pretrain_config is None or objective_id is None:
            raise ValueError(f"{treatment.treatment_id} requires an SSL configuration")
        trunk_state, ssl_resources = obtain_trunk(
            treatment, objective_id, rows, seed, pretrain_config
        )
        resources.update(ssl_resources)

    started = time.perf_counter()
    fitted = supervised.train_fixed_epochs(
        train_inputs=train_inputs,
        train_labels=corpus.labels[rows],
        epochs=epochs,
        seed=seed,
        config=supervised_config,
        trunk_state=trunk_state,
        stage_offset=supervised.FINAL_STAGE_OFFSET,
    )
    resources["supervised_seconds"] = time.perf_counter() - started
    resources["supervised_optimizer_steps"] = fitted.optimizer_steps

    scored = model_module.initialize_backbone(seed)
    scored.load_state_dict(fitted.state)
    prototypes = readout.NearestClassMean.fit(
        model_module.embed_numpy(scored, train_inputs),
        corpus.labels[rows],
        fitted_on="all_source_P1_P2_P3",
    )
    return {
        "treatment_id": treatment.treatment_id,
        "objective_id": objective_id,
        "seed": seed,
        "epochs": epochs,
        "state": fitted.state,
        "state_sha256": fitted.state_sha256,
        "ssl_trunk_sha256": resources.get("ssl_trunk_sha256"),
        "standardizer_mean": standardizer.mean,
        "standardizer_scale": standardizer.scale,
        "standardizer_sha256": canonical_json_sha256(
            {
                "mean_sha256": array_sha256(standardizer.mean),
                "scale_sha256": array_sha256(standardizer.scale),
                "fit_sample_count": int(rows.size),
                "fit_domains": ["P1", "P2", "P3"],
            }
        ),
        "ncm_prototypes": prototypes.prototypes,
        "ncm_record": prototypes.as_record(),
        "epoch_losses": list(fitted.epoch_losses),
        "resources": resources,
    }


def sweep(
    *,
    treatment: treatments.Treatment,
    objective_id: str | None,
    seeds: tuple[int, ...],
    folds: tuple[str, ...],
    pretrain_config: pretrain.PretrainConfig | None,
    supervised_config: supervised.SupervisedConfig,
    progress: Any = None,
) -> list[UnitResult]:
    results: list[UnitResult] = []
    for fold_id in folds:
        for seed in seeds:
            result = run_fold_unit(
                treatment=treatment,
                objective_id=objective_id,
                fold_id=fold_id,
                seed=seed,
                pretrain_config=pretrain_config,
                supervised_config=supervised_config,
            )
            results.append(result)
            if progress is not None:
                progress(result)
    return results


def aggregate_source_side(results: list[UnitResult]) -> dict[str, Any]:
    """Source-side aggregate using the canonical selector's primary criterion."""

    by_fold: dict[str, list[float]] = {}
    for item in results:
        by_fold.setdefault(item.fold_id, []).append(item.head_metrics["macro_f1"])
    fold_means = {fold: float(np.mean(values)) for fold, values in sorted(by_fold.items())}
    all_macro = [item.head_metrics["macro_f1"] for item in results]
    all_accuracy = [item.head_metrics["accuracy"] for item in results]
    worst_recall = [item.head_metrics["worst_class_recall"] for item in results]
    ncm_macro = [item.ncm_metrics["macro_f1"] for item in results]
    by_seed: dict[int, list[float]] = {}
    for item in results:
        by_seed.setdefault(item.seed, []).append(item.head_metrics["macro_f1"])
    return {
        "unit_count": len(results),
        "fold_mean_macro_f1": fold_means,
        "worst_fold_macro_f1": float(min(fold_means.values())),
        "mean_macro_f1": float(np.mean(all_macro)),
        "mean_accuracy": float(np.mean(all_accuracy)),
        "macro_f1_population_standard_deviation": float(np.std(all_macro, ddof=0)),
        "mean_worst_class_recall": float(np.mean(worst_recall)),
        "zero_recall_unit_count": int(
            sum(1 for item in results if item.head_metrics["zero_recall_class_count"] > 0)
        ),
        "mean_ncm_macro_f1": float(np.mean(ncm_macro)),
        "across_seed_standard_deviation": float(
            np.std([float(np.mean(values)) for values in by_seed.values()], ddof=0)
        ),
        "total_ssl_seconds": float(sum(item.resources.get("ssl_seconds", 0.0) for item in results)),
        "total_supervised_seconds": float(
            sum(item.resources.get("supervised_seconds", 0.0) for item in results)
        ),
    }
