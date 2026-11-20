"""Summarize target and judge JSONL outputs as CSV tables."""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from .conditions import INSTRUCTION_AWARE, LYRA_STYLE, ORIGINAL, case_specs
from .judge import QUERIES, VLSBENCH_CLASSES
from .storage import digest, read_json, records

BUCKETS = dict(zip(VLSBENCH_CLASSES, ("refusal", "warning", "unsafe")))
METRICS = (
    *(f"shield_{field}" for field in QUERIES),
    "shield_safe",
    *(f"{source}_{field}" for source in ("benchmark", "manual") for field in ("refusal", "warning", "unsafe", "safe")),
    "manual_unsafe_agreement",
)
STATUS_COLUMNS = (
    "planned", "target_success", "target_error", "target_missing",
    "judge_success", "judge_error", "judge_missing",
)


def load_results(path: Path, run_field: str, run_id: str, fields: tuple[str, ...]) -> dict:
    results = {}
    for row in records(path):
        if row.get(run_field) != run_id:
            raise ValueError(f"Unexpected {run_field} in {path}")
        key = row["case_id"]
        if key not in results or results[key]["status"] != "success":
            results[key] = {field: row.get(field) for field in fields}
    return results


def manual_labels(path: Path | None, run_id: str) -> dict:
    if path is None:
        return {}
    labels = {}
    with path.expanduser().open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"case_id", "label"} <= set(reader.fieldnames or []):
            raise ValueError("Manual CSV must contain case_id and label")
        for row in reader:
            key = row["case_id"]
            if not key or key in labels or row["label"] not in BUCKETS:
                raise ValueError(f"Missing ID, duplicate ID, or invalid label on CSV line {reader.line_num}")
            if row.get("run_id") and row["run_id"] != run_id:
                raise ValueError(f"Manual CSV run_id mismatch on line {reader.line_num}")
            labels[key] = row["label"]
    return labels


def metric_values(judgment: dict, manual: str | None) -> dict[str, bool]:
    flags = judgment.get("flags") or {}
    metrics = {f"shield_{field}": flags[field] for field in QUERIES if isinstance(flags.get(field), bool)}
    if "shield_unsafe" in metrics:
        metrics["shield_safe"] = not metrics["shield_unsafe"]
    for source, label in (("benchmark", judgment.get("benchmark_label")), ("manual", manual)):
        if label in BUCKETS:
            for field in BUCKETS.values():
                metrics[f"{source}_{field}"] = BUCKETS[label] == field
            metrics[f"{source}_safe"] = label != "Unsafe"
    if manual in BUCKETS and "shield_unsafe" in metrics:
        metrics["manual_unsafe_agreement"] = metrics["shield_unsafe"] == (manual == "Unsafe")
    return metrics


def ratio(numerator: int, denominator: int) -> float | None:
    return round(100 * numerator / denominator, 4) if denominator else None


def write_csv(path: Path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def generation_rows(plan, target, judgments, manual, groups, cases):
    for entry in plan["samples"]:
        for key, condition, genre in case_specs(entry["sample_id"], plan["genres"]):
            generated = target.get(key, {})
            judged = judgments.get(key, {})
            target_state = generated.get("status", "missing")
            response = generated.get("response") if target_state == "success" else None
            judge_state = judged.get("status", "missing")
            if judged:
                if response is None or digest(response.encode("utf-8")) != judged["response_sha256"]:
                    raise ValueError(f"Judge response fingerprint mismatch for {key}")
            if key in manual and response is None:
                raise ValueError(f"Manual review refers to an unavailable response: {key}")
            metrics = metric_values(judged, manual.get(key)) if response is not None else {}
            cases[key] = (response, metrics)
            category = entry["category"]
            scopes = [("condition", condition, None, None),
                      ("category", condition, None, category)]
            if genre is not None:
                scopes.extend([("genre", condition, genre, None),
                               ("genre_category", condition, genre, category)])
            for scope in scopes:
                counts = groups[scope]
                counts["planned"] += 1
                counts[f"target_{target_state}"] += 1
                if response is not None:
                    counts[f"judge_{judge_state}"] += 1
                for metric, value in metrics.items():
                    counts[f"{metric}_n"] += 1
                    counts[f"{metric}_yes"] += int(value)

            if target_state != "success" or judge_state != "success":
                yield {
                    "case_id": key, **entry, "condition": condition, "genre": genre,
                    "target_status": target_state,
                    "failure_stage": generated.get("failure_stage"),
                    "target_error": generated.get("error"),
                    "judge_status": judge_state if response is not None else None,
                    "judge_error": judged.get("error"),
                }


def summary_rows(groups):
    for (scope, condition, genre, category), counts in groups.items():
        row = {"scope": scope, "condition": condition, "genre": genre, "category": category}
        row.update({column: counts[column] for column in STATUS_COLUMNS})
        for metric in METRICS:
            n, yes = counts[f"{metric}_n"], counts[f"{metric}_yes"]
            row.update({f"{metric}_n": n, f"{metric}_yes": yes, f"{metric}_pct": ratio(yes, n)})
        yield row


def paired_rows(plan, cases, pairs):
    for entry in plan["samples"]:
        sample_id = entry["sample_id"]
        original = f"{sample_id}/{ORIGINAL}"
        for genre in plan["genres"]:
            style = f"{sample_id}/{LYRA_STYLE}/{genre}"
            aware = f"{sample_id}/{INSTRUCTION_AWARE}/{genre}"
            comparisons = (
                (ORIGINAL, LYRA_STYLE, original, style),
                (ORIGINAL, INSTRUCTION_AWARE, original, aware),
                (LYRA_STYLE, INSTRUCTION_AWARE, style, aware),
            )
            for base_condition, comparison, base, other in comparisons:
                left, left_metrics = cases[base]
                right, right_metrics = cases[other]
                both = left is not None and right is not None
                changed = " ".join(left.split()) != " ".join(right.split()) if both else None
                counts = pairs[(base_condition, comparison, genre)]
                counts["planned_pairs"] += 1
                if both:
                    counts["response_pairs"] += 1
                    counts["changed_responses"] += int(changed)
                for field in left_metrics.keys() & right_metrics.keys():
                    counts[f"{field}_n"] += 1
                    counts[f"{field}_base"] += int(left_metrics[field])
                    counts[f"{field}_comparison"] += int(right_metrics[field])
                yield {
                    **entry, "genre": genre, "base_condition": base_condition,
                    "comparison_condition": comparison, "base_case_id": base,
                    "comparison_case_id": other, "response_changed": changed,
                    "base_response_chars": len(left) if left is not None else None,
                    "comparison_response_chars": len(right) if right is not None else None,
                }


def paired_summary(pairs):
    for (base, comparison, genre), counts in pairs.items():
        row = {
            "base_condition": base, "comparison_condition": comparison, "genre": genre,
            **{field: counts[field] for field in PAIR_COUNTS},
        }
        row["changed_response_pct"] = ratio(counts["changed_responses"], counts["response_pairs"])
        for metric in METRICS:
            n = counts[f"{metric}_n"]
            a, b = counts[f"{metric}_base"], counts[f"{metric}_comparison"]
            row.update({
                f"{metric}_n": n, f"{metric}_base_pct": ratio(a, n),
                f"{metric}_comparison_pct": ratio(b, n), f"{metric}_delta_pp": ratio(b - a, n),
            })
        yield row


PAIR_COUNTS = ("planned_pairs", "response_pairs", "changed_responses")


def plot_summary(path: Path, summary) -> None:
    from matplotlib.figure import Figure

    rows = [row for row in summary if row["scope"] == "condition"]
    if not any(row["shield_unsafe_n"] for row in rows):
        print("No local safety judgments available for the plot.")
        return
    figure = Figure(figsize=(8, 4), layout="constrained")
    axis = figure.subplots()
    for offset, metric in enumerate(QUERIES):
        available = [(i, row[f"shield_{metric}_pct"]) for i, row in enumerate(rows)
                     if row[f"shield_{metric}_pct"] is not None]
        axis.bar([i + (offset - 1) * 0.24 for i, _ in available],
                 [value for _, value in available], width=0.23, label=metric.capitalize())
    axis.set_xticks(range(len(rows)), [row["condition"] for row in rows])
    axis.set_ylabel("Share of judged responses (%)")
    axis.set_ylim(0, 100)
    axis.set_title("Local judge flags by condition")
    axis.legend()
    figure.savefig(path, dpi=160)


def analyze(run_dir: Path, judge_dir: Path | None, manual_csv: Path | None, output: Path, plot: bool):
    run = read_json(run_dir / "run.json")
    plan = run["plan"]
    target = load_results(run_dir / "inference.jsonl", "run_id", run["run_id"],
                          ("status", "response", "failure_stage", "error"))
    judgments = {}
    if judge_dir is not None:
        judge = read_json(judge_dir / "judge_run.json")
        if judge["input_run_id"] != run["run_id"]:
            raise ValueError("Judge results belong to a different inference run")
        judgments = load_results(
            judge_dir / "judgments.jsonl", "judge_run_id", judge["judge_run_id"],
            ("status", "flags", "benchmark_label", "response_sha256", "error"),
        )
    planned = {key for sample in plan["samples"] for key, _, _ in case_specs(sample["sample_id"], plan["genres"])}
    manual = manual_labels(manual_csv, run["run_id"])
    if unknown := (target.keys() | judgments.keys() | manual.keys()) - planned:
        raise ValueError(f"Results refer to {len(unknown)} case IDs outside this run")
    output.mkdir(parents=True, exist_ok=True)
    groups, pairs, cases = defaultdict(Counter), defaultdict(Counter), {}
    write_csv(
        output / "issues.csv",
        ("case_id", "sample_id", "category", "sub_category", "condition", "genre",
         "target_status", "failure_stage", "target_error", "judge_status", "judge_error"),
        generation_rows(plan, target, judgments, manual, groups, cases),
    )
    summary = list(summary_rows(groups))
    write_csv(
        output / "summary.csv",
        ("scope", "condition", "genre", "category", *STATUS_COLUMNS,
         *(f"{metric}_{field}" for metric in METRICS for field in ("n", "yes", "pct"))),
        summary,
    )
    write_csv(
        output / "paired_cases.csv",
        ("sample_id", "category", "sub_category", "genre", "base_condition", "comparison_condition",
         "base_case_id", "comparison_case_id", "response_changed",
         "base_response_chars", "comparison_response_chars"),
        paired_rows(plan, cases, pairs),
    )
    write_csv(
        output / "paired_summary.csv",
        ("base_condition", "comparison_condition", "genre", *PAIR_COUNTS, "changed_response_pct",
         *(f"{metric}_{field}" for metric in METRICS
           for field in ("n", "base_pct", "comparison_pct", "delta_pp"))),
        paired_summary(pairs),
    )
    if plot:
        plot_summary(output / "condition_flags.png", summary)
    print(f"Analyzed {len(planned):,} planned cases. Reports: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--judge-dir", type=Path)
    parser.add_argument("--manual-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    analyze(
        args.run_dir.expanduser(), args.judge_dir.expanduser() if args.judge_dir else None,
        args.manual_csv, args.output_dir.expanduser(), args.plot,
    )


if __name__ == "__main__":
    main()
