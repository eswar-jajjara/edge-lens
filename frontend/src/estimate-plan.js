'use strict';
(function () {
  window.EdgeLensEstimate = Object.freeze({
    afterBenchmark(report, choice) {
      if (!choice?.enabled) return { action: 'skip' };
      if (report?.source !== 'measured') return { action: 'unavailable', reason: 'Complete the local benchmark first.' };
      const artifacts = (report.artifacts || []).map((artifact, index) => ({ artifact, index })).filter(({artifact}) => artifact.format === 'tflite' && /^[a-f0-9]{64}$/.test(artifact.sha256));
      if (!artifacts.length) return { action: 'unavailable', reason: 'This test has no evaluated TFLite artifact. Its laptop report is saved. Import and benchmark TFLite to request an estimate; ONNX-only results cannot supply TFLite accuracy.' };
      if (artifacts.length > 1) return { action: 'select', reason: 'Choose which evaluated TFLite artifact to upload, then request its estimate.' };
      return { action: 'upload', artifact_index: artifacts[0].index };
    },
  });
})();
