'use strict';
// Fictional offline fixtures, never persisted as measurements.
window.EdgeLensDemoReport = Object.freeze({
  source: 'illustrative', summary: 'Illustrative example: accuracy is unchanged, while the configured conversion has a different server inference time. This is not measured evidence.',
  model: { id: 'mobilenet_v2', name: 'MobileNetV2' }, dataset: { name: 'Example image set', image_count: 100, class_count: 10 },
  environment: { device: 'Illustrative server CPU', threads: 1 },
  target: { id: 'raspberry_pi', name: 'Raspberry Pi', status: 'Not tested on device', latency_ms: null, notes: ['These example timings do not describe a Raspberry Pi.', 'An on-device run is required to measure real latency and memory.'] },
  metrics: [
    { profile: 'pytorch', label: 'PyTorch reference', accuracy_pct: 80, accuracy_delta_pp: 0, agreement_pct: 100, latency_p50_ms: 19.2, latency_p95_ms: 21.6, conversion_seconds: null, size_bytes: 14155776, output_mae: 0, output_max_abs: 0 },
    { profile: 'standard', label: 'Standard ONNX conversion', accuracy_pct: 80, accuracy_delta_pp: 0, agreement_pct: 100, latency_p50_ms: 11.6, latency_p95_ms: 13.1, conversion_seconds: 2.8, size_bytes: 13841203, output_mae: 0.00012, output_max_abs: 0.0017 },
    { profile: 'dashboard', label: 'Configured ONNX conversion', accuracy_pct: 80, accuracy_delta_pp: 0, agreement_pct: 100, latency_p50_ms: 10.9, latency_p95_ms: 12.8, conversion_seconds: 3.4, size_bytes: 13841203, output_mae: 0.00012, output_max_abs: 0.0017 },
  ],
  layers: [
    { name: 'features.0', operation: 'Conv / BatchNorm / ReLU6', profile: 'standard', status: 'unmapped', mae: null, max_abs: null, detail: 'Illustrative fused boundary: no reliable one-to-one intermediate mapping. No numerical pass is claimed.' },
    { name: 'features.1.conv.0', operation: 'Depthwise convolution', profile: 'standard', status: 'pass', mae: 0.00001, max_abs: 0.00008, detail: 'Illustrative aligned output is within the configured tolerance.' },
    { name: 'features.2.conv.2', operation: 'Pointwise convolution', profile: 'dashboard', status: 'review', mae: 0.0003, max_abs: 0.003, detail: 'Illustrative tensor exceeds tolerance. Check matching boundaries and preprocessing; this alone does not establish lower classification accuracy.' },
    { name: 'classifier.1', operation: 'Linear classifier', profile: 'dashboard', status: 'pass', mae: 0.00012, max_abs: 0.0017, detail: 'Illustrative final logits retain the same top-1 predictions in this example.' },
  ], artifacts: [],
  methodology: ['Use the same labelled images, preprocessing and class mapping for every profile.', 'Top-1 accuracy uses ground-truth labels. Agreement compares predictions with PyTorch.', 'Compare signed accuracy differences in percentage points; improvement is not assumed.', 'Time conversion separately from inference. Warm up before repeated server CPU inference.'],
  limitations: ['All figures on this demo are fabricated for interface review.', 'Four example layer rows illustrate report states; they are not a complete model trace.', 'No model was executed and no target device was benchmarked.'],
});
