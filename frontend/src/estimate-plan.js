'use strict';
(function () {
  window.EdgeLensEstimate = Object.freeze({
    eligibleArtifacts(report) {
      const dataset = report?.datasets?.test || report?.dataset || {};
      return (report?.artifacts || []).map((artifact, index) => ({ artifact, index })).filter(({artifact}) =>
        ['onnx', 'tflite'].includes(artifact.format) && /^[a-f0-9]{64}$/.test(artifact.sha256) &&
        /^[a-f0-9]{64}$/.test(dataset.sha256) && (report.metrics || []).some(metric =>
          metric.artifact_sha256 === artifact.sha256 && metric.dataset_sha256 === dataset.sha256 &&
          metric.evaluation_split === 'test' && metric.sample_count > 0 && metric.sample_count === dataset.image_count &&
          typeof metric.accuracy_pct === 'number' && Number.isFinite(metric.accuracy_pct)))
        .filter((entry, index, all) => all.findIndex(other => other.artifact.sha256 === entry.artifact.sha256 && other.artifact.format === entry.artifact.format) === index);
    },
    afterBenchmark(report, choice) {
      if (!choice?.enabled) return { action: 'skip' };
      if (report?.source !== 'measured') return { action: 'unavailable', reason: 'Complete the local benchmark first.' };
      const artifacts = window.EdgeLensEstimate.eligibleArtifacts(report);
      if (!artifacts.length) return { action: 'unavailable', reason: 'This test has no ONNX or TFLite artifact with complete held-out evaluation. Its laptop report is saved; re-run evaluation before requesting an estimate.' };
      if (artifacts.length > 1) return { action: 'select', reason: 'Choose which evaluated ONNX or TFLite artifact to upload, then request its estimate.' };
      return { action: 'upload', artifact_index: artifacts[0].index };
    },
  });
})();
