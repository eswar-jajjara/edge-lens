# Phase 2 — deployment configuration search (0.6.0)

EdgeLens extends the existing desktop tool and CLI with bounded ONNX
configuration search. It uses the standard PyTorch exporter and ONNX Runtime
quantizer, then measures calibration variants and selective operation exclusions.
It can preserve accuracy more closely, retain FP32, or find no feasible candidate.
An improvement must come from the actual report; the tool does not promise one.

## Run in the desktop tool

1. Close the old app and launch `start-desktop.cmd` from current source. Do not
   open an old `release/win-unpacked/EdgeLens.exe`. Teammates use
   [Windows source setup](TEAM-SETUP.md).
2. Select or upload a trusted `.pt2` or **FP32 single-file ONNX** classifier.
   Declare its exact input shape, class order and preprocessing. The current
   contract is one fixed batch-one floating-point image input and one
   `[1, classes]` output. A weights-only `.pth` must be exported first.
3. Upload three disjoint labelled ZIPs: **calibration**, **validation** and a
   **held-out test**. Select them in their respective fields. Archive equality,
   duplicate preprocessed images and inconsistent class mappings are rejected.
4. Choose ONNX and **Optimize deployment · validation search**. The calibration
   and weight controls define the initial INT8 control. Search also explores
   MinMax/Entropy/Percentile with per-channel/per-tensor weights.
5. Choose an objective: smallest file, lowest median host latency, highest
   validation accuracy, or trade-offs only. Set maximum accuracy loss relative
   to validation FP32, maximum MiB and/or maximum median host latency. Blank
   limits mean no constraint. A 1 pp accuracy limit is a useful initial policy;
   it is not a guarantee about test accuracy.
6. Start the experiment. Read candidate outcomes, constraint rejection reasons,
   Pareto points, sensitivity controls and layer diagnostics. Export HTML/CSV/JSON
   and download the exact selected ONNX artifact by candidate ID and SHA256.

The ESP32/Raspberry Pi choice is a deployment goal. Phase 2 times this computer's
CPU. It does not run ONNX on an ESP32 or measure either board's latency.

## Run with the same engine from PowerShell

From the repository root, replace the input paths with your own files:

```powershell
backend/.venv/Scripts/python.exe backend/cli.py `
  --model "inputs/model.pt2" --spec "inputs/spec.json" `
  --calibration "inputs/calibration.zip" `
  --validation "inputs/validation.zip" --dataset "inputs/test.zip" `
  --format onnx --target raspberry_pi --strategy deployment_search `
  --objective size --max-accuracy-loss-pp 1 `
  --max-candidates 16 --sensitivity-probes 4 --diagnostic-samples 3 `
  --search-seconds 600 --output ".cache/my-phase2-experiment" `
  --trust-own-model
```

The CLI writes SQLite, model artifacts, calibration statistics and
`runs/<run-id>/report.{html,csv,json}`. It uses the same builder, evaluator,
constraint selector and report exporter as the desktop. Source dependencies and
model input contracts are unchanged from Phase 1.

## What the search measures

- **Static candidates:** the original FP32 graph plus standard S8S8 QDQ INT8
  candidates. Calibration labels and test inputs do not choose quantization
  ranges. Matching-method range caches are reused only inside the same experiment
with the same baseline and calibration inputs.
- **Activation diagnostics:** compare up to 128 eligible Conv/Gemm/MatMul
  operation outputs on 1–8 calibration images, eight operations per capture.
  Match only unique preserved ONNX node names with the same operation type and
  single output. Report MAE, maximum absolute error, normalized RMSE, sample
  count, tolerance failures, split and mapping provenance. Unmapped, ambiguous,
  unsupported and failed captures remain unavailable.
- **Sensitivity controls:** start from the successful INT8 candidate with highest
  validation accuracy, breaking ties by output MAE and ID. Rank its measured
  operation drift by NRMSE, then exclude one operation at a time while holding
  other quantizer settings and calibration ranges fixed. Record recovery in
  validation accuracy and output MAE against that named parent, including
  negative and zero changes. Try cumulative exclusions only for individually
  improving operations (accuracy, then MAE on equal accuracy).
- **Selective precision:** excluded operations retain FP32 weights where
  verifiable. Adjacent activations can still pass through QDQ. These are selective
  quantization exclusions, not proof of entirely FP32 execution or integer-only
  hardware kernels. Intermediate drift includes upstream error propagation;
  controlled recovery is validation-specific evidence, not universal root cause.
- **Deployment evidence:** candidates run with the same optimized host runtime,
  preprocessing, threads and validation images. Shared timing code uses warm-ups,
  rotating profile order and raw samples on one fixed validation image. Diagnostic
  sessions disable graph optimization and are never timed as deployment artifacts.

Known diagnostic output sizes are guarded at 128 MiB per graph chunk; observed
outputs across both graphs are guarded at 256 MiB. Guards do not establish process
peak RAM. Unsupported captures do not erase the candidate's other measurements.

## Selection and held-out evaluation

Every candidate must have successful validation accuracy, serialized size and
median host latency. Hard constraints are checked separately, with explicit
reasons for rejection. No composite score combines unlike units.

| Objective | Selection order after constraints |
|---|---|
| Size | Smaller bytes → higher accuracy → lower median latency → ID |
| Latency | Lower median latency → higher accuracy → smaller bytes → ID |
| Accuracy | Higher accuracy → smaller bytes → lower median latency → ID |
| Trade-offs | No automatic selection |

Pareto membership uses validation accuracy (maximize), bytes (minimize) and median
host latency (minimize) across successfully measured candidates, including those
rejected by a requested constraint. A Pareto point is not automatically feasible.

Selection is frozen **before** held-out inference. The PyTorch reference (for
PT2), FP32, initial INT8 control and selected candidate receive final test metrics.
Other candidates retain validation evidence only. A failed final test is reported;
the tool does not choose a replacement using test data. If all candidates violate
the limits, the result is **no feasible candidate**. FP32 can be selected.

If a prior test set was already inspected while changing the strategy, reserve a
fresh untouched test set for confirmation. Repeat independent host timings before
claiming a tiny latency difference. With 500 test images, one changed prediction
is 0.2 pp; a 0.001 pp accuracy claim cannot be supported by that sample.

## Budgets, storage and compatibility

Default search: at most 16 candidates including the two controls, four operation
probes, three diagnostic images and 600 seconds. API limits are 24 candidates,
eight probes, eight diagnostic images and 1,800 seconds. The time budget starts
after initial controls are built and is checked between operations; an in-progress
build/evaluation finishes. Validation timing and selection finish for built
candidates, so this is a **soft budget**, not a hard timeout. Failed candidates
remain visible with their build/runtime reason.

Set operation probes to zero to retain calibration diagnostics without building
exclusion candidates. A two-candidate limit keeps the FP32 and initial INT8
controls. Diagnostic capture can be unavailable if the soft budget is exhausted.

SQLite migration 4 adds `sensitivity_results` and `deployment_selections` while
preserving earlier reports, candidates and dataset-role records. Public exports
omit internal filesystem paths. The `layer_diagnostics` JSON artifact preserves
the complete diagnostic/control evidence. Do not downgrade or delete the database
to open an old EXE; rebuild the portable app or use current source.

Phase 2 adds no quantization-aware training, TFLite optimization, peak host RAM,
energy measurement or new hardware runner. Existing imported TFLite and ESP32
package/report workflows remain available separately.
