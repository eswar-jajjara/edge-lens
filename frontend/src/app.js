'use strict';
const $ = selector => document.querySelector(selector);
const api = window.EdgeLensAPI;
const desktop = Boolean(window.EDGELENS_CONFIG?.desktop);
const workerName = desktop ? 'this computer' : 'the server CPU';
const state = { mode: 'demo', page: 'overview', report: window.EdgeLensDemoReport, run: null, capabilities: null, datasets: [], models: [], edgeRecords: [], history: [], connected: false, creating: false, uploading: false, pollToken: 0, connectionToken: 0 };
const pages = { edge: ['Edge hardware', 'Measure the real board. Keep estimates separate.', 'Edge hardware'], overview: ['Validation overview', 'See what conversion changes. Keep the evidence.', 'Overview'], diagnostics: ['Layer diagnostics', 'Find the differences behind the final predictions.', 'Layer diagnostics'], report: ['Reports & history', 'Keep the complete record of each experiment.', 'Reports & history'] };
const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]);
const numeric = value => typeof value === 'number' && Number.isFinite(value);
const number = (value, places = 2) => numeric(value) ? value.toLocaleString('en-US', { maximumFractionDigits: places, minimumFractionDigits: places }) : '—';
const delta = value => numeric(value) ? `${value > 0 ? '+' : ''}${number(value, 6)}` : '—';
const scientific = value => numeric(value) ? (value === 0 ? '0' : Math.abs(value) < 0.001 ? value.toExponential(2) : number(value, 5)) : '—';
const human = value => String(value ?? 'Not available').replaceAll('_', ' ');
const list = values => (values || []).map(value => `<li>${escapeHtml(value)}</li>`).join('');
const summaryText = value => typeof value === 'string' ? value : value?.conclusion || '';
const displayName = value => typeof value === 'string' ? value : value?.name || value?.id || 'Not available';
const isActive = run => run && ['queued', 'running'].includes(run.status);
let toastTimer;
function toast(message) { clearTimeout(toastTimer); $('#toast').textContent = message; $('#toast').hidden = false; toastTimer = setTimeout(() => { $('#toast').hidden = true; }, 5000); }
function error(message) { $('#workspace-error').textContent = message || ''; $('#workspace-error').hidden = !message; }
function describeError(cause) { return cause?.message === 'Failed to fetch' ? 'Cannot reach the benchmark API. Start the backend and check its API address, then reconnect.' : cause?.message || 'The request failed. Please try again.'; }
function showPage(page) {
  if (!pages[page]) page = 'overview';
  state.page = page;
  document.querySelectorAll('.page-panel').forEach(panel => { panel.hidden = panel.id !== `${page}-page`; });
  document.querySelectorAll('[data-page]').forEach(button => { const selected = button.dataset.page === page; button.classList.toggle('active', selected); if (selected) button.setAttribute('aria-current', 'page'); else button.removeAttribute('aria-current'); });
  $('#page-title').textContent = pages[page][0]; $('#page-description').textContent = pages[page][1]; $('#breadcrumb').textContent = pages[page][2];
}
function navigate(page) { location.hash = page; showPage(page); }
function layerGroup(status) { const value = String(status).toLowerCase(); if (['pass', 'within_tolerance', 'matched'].includes(value)) return 'pass'; if (['drift', 'nonfinite', 'review', 'fail', 'failed', 'warning', 'exceeds_tolerance', 'mismatch'].includes(value)) return 'review'; return 'unmapped'; }
function layers() { return state.report?.layers || []; }
function renderLayers() {
  const search = $('#layer-search').value.trim().toLowerCase(), filter = $('#layer-filter').value;
  const rows = layers().filter(layer => (filter === 'all' || layerGroup(layer.status) === filter) && `${layer.name} ${layer.operation} ${layer.profile}`.toLowerCase().includes(search));
  $('#layer-count').textContent = `${rows.length} / ${layers().length} reported layers`;
  $('#layer-list').innerHTML = rows.length ? rows.map(layer => {
    const group = layerGroup(layer.status), label = group === 'pass' ? 'Within tolerance' : group === 'review' ? 'Needs review' : 'Not compared';
    return `<article class="layer-row"><div class="layer-row-heading"><div><code>${escapeHtml(layer.name)}</code><p>${escapeHtml(layer.operation || 'Operation not available')} <span>· ${escapeHtml(human(layer.profile))}</span></p></div><span class="status ${group === 'pass' ? 'pass' : group === 'review' ? 'warning' : 'neutral'}">${label}</span></div><div class="layer-errors"><span>Mean absolute error <strong>${scientific(layer.mae)}</strong></span><span>Maximum absolute error <strong>${scientific(layer.max_abs)}</strong></span></div><details><summary>Comparison details</summary><p>${escapeHtml(layer.detail || 'No additional diagnostic context was recorded.')}${layer.max_error_index ? `<br>Largest error at [${escapeHtml(layer.max_error_index.join(', '))}] · expected ${scientific(layer.expected_value)} · actual ${scientific(layer.actual_value)}` : ''}</p></details></article>`;
  }).join('') : '<div class="empty-inline">No layers to show. Run a benchmark or change the layer filters.</div>';
}
function renderChart(metrics) {
  const maximum = Math.max(1, ...metrics.map(metric => numeric(metric.latency_p50_ms) ? metric.latency_p50_ms : 0));
  $('#performance-chart').innerHTML = metrics.map((metric, index) => `<div class="latency-row"><div><strong>${escapeHtml(metric.label || human(metric.profile))}</strong><span>${number(metric.latency_p50_ms, 3)} ms</span></div><div class="latency-track"><div class="latency-bar tone-${index % 3}" style="width:${numeric(metric.latency_p50_ms) ? Math.max(1, metric.latency_p50_ms / maximum * 100) : 0}%"></div></div></div>`).join('');
}
function renderReport() {
  const report = state.report, demo = state.mode === 'demo';
  $('#result-overview').hidden = !report; $('#empty-results').hidden = Boolean(report); $('#report-page .report-panel').hidden = !report;
  document.querySelectorAll('[data-export]').forEach(button => { button.disabled = !report; });
  if (!report) { renderLayers(); renderDeveloperReport(); return; }
  const metrics = report.metrics || [], reference = metrics.find(metric => ['pytorch', 'reference', 'baseline'].includes(metric.profile)) || metrics[0];
  const standard = metrics.find(metric => ['fp32', 'standard', 'default', 'standard_conversion'].includes(metric.profile)) || metrics[1];
  const configured = metrics.find(metric => ['static_int8', 'dashboard', 'configured', 'dashboard_configured'].includes(metric.profile)) || metrics[2];
  const difference = numeric(configured?.accuracy_pct) && numeric(standard?.accuracy_pct) ? configured.accuracy_pct - standard.accuracy_pct : null;
  const compared = layers().filter(layer => numeric(layer.mae) && numeric(layer.max_abs)).length;
  $('#reference-accuracy').innerHTML = `${number(reference?.accuracy_pct)}<span class="metric-unit">%</span>`;
  $('#accuracy-difference').innerHTML = `${delta(difference)}<span class="metric-unit">pp</span>`;
  $('#configured-latency').innerHTML = `${number((report.experiment_type ? configured : configured || reference)?.latency_p50_ms, 3)}<span class="metric-unit">ms</span>`;
  $('#layer-coverage').innerHTML = `${compared}<span class="metric-unit">/ ${layers().length}</span>`;
  $('#layer-coverage-note').textContent = demo ? 'Compared / reported · illustrative sample only' : 'Numerically compared / reported layer entries';
  $('#active-model').textContent = displayName(report.model); $('#active-dataset').textContent = displayName(report.dataset);
  $('#active-location').textContent = demo ? 'Illustrative CPU values' : `Measured on ${workerName}`;
  const candidateFailures = (report.candidates || []).filter(candidate => candidate.status === 'failed').length;
  $('#run-status').textContent = demo ? 'Illustrative' : candidateFailures ? 'Completed with failures' : 'Measured'; $('#run-status').className = `status ${demo ? 'neutral' : candidateFailures ? 'warning' : 'pass'}`;
  $('#run-id').textContent = demo ? 'DEMO · NOT MEASURED' : state.run?.id || 'Saved report';
  $('#chart-label').textContent = demo ? 'Illustrative CPU values' : desktop ? 'This computer · CPU' : 'Server CPU only'; renderChart(metrics);
  $('#comparison-body').innerHTML = metrics.map((metric, index) => `<tr><td><div class="framework-cell"><span class="framework-logo ${['torch', 'onnx-logo', 'lite'][index % 3]}">${index + 1}</span><div><strong>${escapeHtml(metric.label || human(metric.profile))}</strong><small>${metric.profile === 'imported' ? 'Uploaded model; no PyTorch reference' : index === 0 ? 'Original reference' : index === 1 ? 'Standard settings' : 'Configured settings'}</small></div></div></td><td>${number(metric.accuracy_pct)}${numeric(metric.accuracy_pct) ? '%' : ''}</td><td>${delta(metric.accuracy_delta_pp)}</td><td>${number(metric.agreement_pct)}${numeric(metric.agreement_pct) ? '%' : ''}</td><td>${number(metric.latency_p50_ms, 3)} / ${number(metric.latency_p95_ms, 3)}</td><td>${number(metric.conversion_seconds)}</td><td>${number(numeric(metric.size_bytes) ? metric.size_bytes / 1048576 : null)}</td><td>${scientific(metric.output_mae)}</td></tr>`).join('');
  $('#target-name').textContent = report.target?.name || human(report.target?.id); $('#target-status').textContent = human(report.target?.status);
  $('#target-notes').innerHTML = list(report.target?.notes?.length ? report.target.notes : ['No target-device benchmark has been recorded.']);
  $('#result-summary').textContent = summaryText(report.summary) || 'Review the measured metrics and layer evidence before deciding whether this conversion meets your requirements.';
  $('#report-model').textContent = displayName(report.model); $('#report-run-id').textContent = $('#run-id').textContent; $('#report-source').textContent = demo ? 'Illustrative data · not evidence' : desktop ? 'Measured · saved locally' : 'Measured · saved on server';
  $('#report-summary').textContent = summaryText(report.summary);
  const environment = report.environment || {}, facts = [ ['Dataset', displayName(report.dataset)], ['Images / classes', `${number(report.dataset?.image_count, 0)} / ${number(report.dataset?.class_count, 0)}`], ['Execution location', demo ? 'Illustrative CPU values' : workerName], ['Platform', environment.platform || environment.system || environment.device || 'See full report'], ['CPU threads', environment.settings?.threads ?? state.run?.request?.settings?.threads ?? '—'], ['Deployment goal', report.target?.name || human(report.target?.id)], ['Device reports', String((report.edge_results || []).filter(x => x.kind === 'hardware').length)] ];
  $('#report-facts').innerHTML = facts.map(([key, value]) => `<dt>${escapeHtml(key)}</dt><dd>${escapeHtml(value)}</dd>`).join('');
  $('#report-methodology').innerHTML = list(report.methodology); $('#report-limitations').innerHTML = list(report.limitations?.length ? report.limitations : ['Server measurements do not establish performance on the target edge device.']);
  $('#model-artifacts').innerHTML = !demo && report.artifacts?.length ? report.artifacts.map((artifact, index) => `<button class="button secondary" data-artifact="${index}">Save ${escapeHtml(human(artifact.profile))}${artifact.role === 'calibration_statistics' ? '' : ' model'} ↓</button>`).join('') : '<p class="field-help">Converted model downloads appear after a measured run.</p>';
  renderLayers(); renderDeveloperReport();
}
function renderHistory() {
  $('#run-history').innerHTML = state.history.length ? state.history.map(run => `<button class="history-row ${state.run?.id === run.id ? 'selected' : ''}" data-run-id="${escapeHtml(run.id)}"><span><strong>${escapeHtml((state.models || []).find(model => model.id === run.request?.model_id)?.name || human(run.request?.model_id || run.model_id || 'Image classifier'))}</strong><small>${escapeHtml(run.id)} · ${escapeHtml(run.created_at ? new Date(run.created_at).toLocaleString() : 'Date not recorded')}</small></span><span class="status ${run.status === 'completed' ? 'pass' : run.status === 'failed' ? 'warning' : 'neutral'}">${escapeHtml(human(run.status))}</span></button>`).join('') : '<div class="empty-inline">No saved experiments yet. Start a benchmark to create your first record.</div>';
}
function renderDatasets(selected) {
  const previous = selected || $('#dataset-select').value;
  $('#dataset-select').innerHTML = '<option value="">Select a labelled dataset</option>' + state.datasets.map(dataset => `<option value="${escapeHtml(dataset.id)}">${escapeHtml(dataset.name)} · ${escapeHtml(dataset.image_count)} images</option>`).join('');
  if (state.datasets.some(dataset => dataset.id === previous)) $('#dataset-select').value = previous;
  renderCalibration(); updateDatasetDescription();
}
function updateDatasetDescription() {
  const dataset = state.datasets.find(item => item.id === $('#dataset-select').value);
  $('#dataset-description').textContent = dataset ? `${dataset.image_count} images · ${dataset.class_count} classes · ${desktop ? 'saved locally' : 'saved on server'}` : 'Accuracy needs labelled images matching your model’s output class order.';
  updateReadiness();
}
function updateTarget() {
  $('#target-hint').textContent = $('#run-target').value === 'esp32' ? 'ESP32 is a constrained microcontroller target. Full MobileNetV2 and ResNet18 are not assumed to fit or run. The report records feasibility limits; device timings require real hardware.' : `Raspberry Pi is the deployment goal. This experiment measures ${workerName}. Raspberry Pi performance still needs a hardware test.`;
}
function updateReadiness() {
  const model = (state.models || []).find(x => x.id === $('#run-model').value);
  const runtime = model?.format === 'tflite' ? {available:state.capabilities?.imported_tflite_execution,reason:'Install ai-edge-litert for imported TFLite inference.'} : state.capabilities?.runtimes?.[$('#run-format').value];
  const quantizing = $('#run-strategy').value === 'quantization_compare';
  const splitIds = [$('#calibration-select').value, $('#validation-select').value, $('#dataset-select').value];
  const calibrationReady = quantizing ? splitIds.every(Boolean) && new Set(splitIds).size === 3 : $('#run-strategy').value !== 'fidelity_search' || ($('#calibration-select').value && $('#calibration-select').value !== $('#dataset-select').value);
  const ready = state.connected && calibrationReady && runtime?.available && $('#dataset-select').value && !state.creating && !state.uploading && !isActive(state.run);
  $('#start-benchmark').disabled = !ready;
  $('#start-benchmark').textContent = state.creating ? 'Submitting…' : isActive(state.run) ? 'Benchmark in progress…' : desktop ? 'Run local benchmark →' : 'Run server benchmark →';
  $('#upload-dataset').disabled = !state.connected || state.uploading;
  $('#upload-model').disabled = !state.connected || !state.capabilities?.custom_model_uploads;
  $('#runtime-note').textContent = !state.connected ? 'Connect to the backend to upload a dataset and start a run.' : !calibrationReady ? (quantizing ? 'Choose three separate calibration, validation and test datasets.' : 'Choose a separate calibration dataset to enable fidelity search.') : runtime?.available ? (quantizing ? 'Measures FP32 and static INT8 on this CPU. Unsupported candidates are recorded with their failure reasons.' : model ? 'Your exported model runs locally with the declared image preprocessing. Unsupported signatures fail with an explanation.' : 'Sample presets use pretrained weights. The first run may download them.') : runtime?.reason || 'This runtime is unavailable. Install its optional backend dependencies before starting this format.';
}
async function connect() {
  const token = ++state.connectionToken; error(''); $('#server-status').textContent = 'Connecting to the benchmark server…'; $('#connect-server').disabled = true;
  const results = await Promise.allSettled([api.capabilities(), api.datasets(), api.runs(), api.models()]);
  if (token !== state.connectionToken || state.mode !== 'real') return;
  $('#connect-server').disabled = false;
  state.connected = results[0].status === 'fulfilled';
  state.capabilities = state.connected ? results[0].value : null;
  if (results[1].status === 'fulfilled') { state.datasets = results[1].value; renderDatasets(); }
  if (results[2].status === 'fulfilled') { state.history = results[2].value; renderHistory(); }
  if (results[3].status === 'fulfilled') { state.models = results[3].value; renderModels(); renderHistory(); }
  $('#server-status').textContent = state.connected ? `Connected · benchmarks run on ${workerName} · reports saved in SQLite` : 'Engine unavailable · reconnect to check the status';
  const failed = results.find(result => result.status === 'rejected'); if (failed) error(describeError(failed.reason));
  updateReadiness();
}
function switchMode(mode) {
  if (mode === state.mode) return;
  state.mode = mode; state.pollToken++; state.connectionToken++; state.run = null; state.edgeRecords = []; state.report = mode === 'demo' ? window.EdgeLensDemoReport : null; error('');
  $('#demo-mode').setAttribute('aria-pressed', String(mode === 'demo')); $('#real-mode').setAttribute('aria-pressed', String(mode === 'real'));
  $('#real-setup').hidden = mode !== 'real'; $('#history-panel').hidden = mode !== 'real'; $('#run-progress').hidden = true; $('#resume-poll').hidden = true;
  $('#environment-label').textContent = mode === 'demo' ? 'Illustrative demo' : desktop ? 'Desktop · local CPU' : 'Server CPU workspace';
  $('#mode-notice').innerHTML = `<span class="info-icon" aria-hidden="true">i</span><span>${mode === 'demo' ? '<strong>Illustrative data.</strong> All values are fabricated examples. No models are executed or saved.' : '<strong>Server benchmarks, edge deployment goals.</strong> Timings describe the machine running your backend. Device measurements appear only after importing or capturing a matching hardware report in Edge hardware.'}</span>`;
  if (desktop && mode === 'real') $('#mode-notice').innerHTML = '<span class="info-icon" aria-hidden="true">i</span><span><strong>Local measurements, edge deployment goals.</strong> Timings describe this computer. Device measurements appear only after importing or capturing a matching hardware report in Edge hardware.</span>';
  renderReport(); navigate('overview'); if (mode === 'real') connect();
}
async function refreshHistory() { state.history = await api.runs(); renderHistory(); }
function showRun(run) {
  if (state.run?.id !== run.id) {
    state.edgeRecords = [];
    if (run.request) {
      const request = run.request;
      $('#run-model').value = request.model_id;
      $('#run-format').value = request.format;
      $('#run-target').value = request.target;
      $('#run-strategy').value = request.strategy || 'fixed_profiles';
      $('#dataset-select').value = request.dataset_id;
      $('#calibration-select').value = request.calibration_dataset_id || '';
      $('#validation-select').value = request.validation_dataset_id || '';
      $('#quant-calibration').value = request.quantization?.calibration_method || 'MinMax';
      $('#quant-per-channel').value = String(request.quantization?.per_channel ?? true);
      for (const [key, id] of Object.entries({warmup_runs:'warmup-runs',measured_runs:'measured-runs',threads:'cpu-threads',atol:'atol',rtol:'rtol'})) {
        if (request.settings?.[key] !== undefined) document.getElementById(id).value = request.settings[key];
      }
      modelChanged(); updateTarget(); updateDatasetDescription();
    }
  }
  state.run = run;
  if (run.report && run.report.source !== 'measured') throw new Error('The backend report is not marked as measured. It cannot be displayed as benchmark evidence.');
  state.report = run.report || null;
  const active = isActive(run); $('#run-progress').hidden = !active; $('#resume-poll').hidden = true;
  $('#progress-title').textContent = run.status === 'queued' ? 'Benchmark queued' : `Benchmark running on ${workerName}`;
  $('#progress-detail').textContent = 'Loading your model, checking its signature and preprocessing images. Conversions and calibration search can take several minutes.';
  if (run.status === 'failed') error(typeof run.error === 'string' ? run.error : run.error?.message || 'The benchmark failed. Open the saved run again after checking the backend logs.');
  if (run.status === 'completed' && !run.report) error('This run completed without an attached report. Refresh the saved run to check again.');
  renderReport(); renderHistory(); updateReadiness(); if (run.status === "completed") refreshEdgeRecords();
}
async function loadRun(id) {
  const token = ++state.pollToken; error('');
  try { const run = await api.run(id); if (token !== state.pollToken || state.mode !== 'real') return; showRun(run); if (isActive(run)) pollRun(id, token); navigate('overview'); }
  catch (cause) { if (token === state.pollToken) error(describeError(cause)); }
}
async function pollRun(id, token) {
  const started = Date.now();
  while (token === state.pollToken && state.mode === 'real') {
    if (Date.now() - started > 15 * 60 * 1000) { $('#progress-detail').textContent = 'Automatic status checks paused after 15 minutes. The server may still be working. Check the status again when ready.'; $('#resume-poll').hidden = false; return; }
    await new Promise(resolve => setTimeout(resolve, 2000));
    if (token !== state.pollToken || state.mode !== 'real') return;
    try {
      const run = await api.run(id); if (token !== state.pollToken || state.mode !== 'real') return;
      showRun(run);
      if (!isActive(run)) { await refreshHistory(); if (run.status === 'completed') toast('Measured report saved on the server.'); return; }
    } catch (cause) { if (token !== state.pollToken || state.mode !== 'real') return; error(describeError(cause)); $('#progress-detail').textContent = 'Status checks paused. The server run may continue; reconnect or check its status again.'; $('#resume-poll').hidden = false; return; }
  }
}
async function startBenchmark(event) {
  event.preventDefault(); if ($('#start-benchmark').disabled) return; error(''); state.creating = true; updateReadiness();
  const token = ++state.pollToken;
  const strategy = $('#run-strategy').value;
  const payload = { model_id: $('#run-model').value, format: $('#run-format').value, target: $('#run-target').value, strategy, calibration_dataset_id: strategy !== 'fixed_profiles' ? $('#calibration-select').value : null, validation_dataset_id: strategy === 'quantization_compare' ? $('#validation-select').value : null, quantization: { calibration_method: $('#quant-calibration').value, per_channel: $('#quant-per-channel').value === 'true' }, dataset_id: $('#dataset-select').value, settings: { warmup_runs: Number($('#warmup-runs').value), measured_runs: Number($('#measured-runs').value), threads: Number($('#cpu-threads').value), atol: Number($('#atol').value), rtol: Number($('#rtol').value) } };
  try { const run = await api.createRun(payload); if (state.mode !== 'real' || token !== state.pollToken) return; showRun(run); await refreshHistory(); if (isActive(run)) pollRun(run.id, token); }
  catch (cause) { if (state.mode === 'real' && token === state.pollToken) error(describeError(cause)); }
  finally { state.creating = false; updateReadiness(); }
}
async function uploadDataset() {
  const file = $('#dataset-file').files[0]; if (!file) { error('Choose a dataset ZIP file first.'); return; }
  if (!file.name.toLowerCase().endsWith('.zip') || file.size > 50 * 1024 * 1024) { error('Choose a ZIP file no larger than 50 MB.'); return; }
  error(''); state.uploading = true; updateReadiness(); $('#upload-status').textContent = 'Uploading and validating the image labels…';
  try { const dataset = await api.uploadDataset(file, file.name.replace(/\.zip$/i, '')); state.datasets = await api.datasets(); renderDatasets(dataset.id); $('#upload-status').textContent = `${dataset.image_count} images validated and stored.`; $('#dataset-file').value = ''; }
  catch (cause) { error(describeError(cause)); $('#upload-status').textContent = 'Upload not completed.'; }
  finally { state.uploading = false; updateReadiness(); }
}
function download(filename, content, type) { const url = URL.createObjectURL(new Blob([content], { type })), anchor = document.createElement('a'); anchor.href = url; anchor.download = filename; document.body.append(anchor); anchor.click(); anchor.remove(); setTimeout(() => URL.revokeObjectURL(url), 2000); }
async function exportReport(format) {
  if (!state.report) return;
  if (state.mode === 'real') {
    try {
      if (format === 'json') download(`EdgeLens-${state.run.id}.json`, JSON.stringify(state.report, null, 2), 'application/json');
      else {
        const response = await fetch(api.reportUrl(state.run.id, format), { signal: AbortSignal.timeout(30000) });
        if (!response.ok) throw new Error(`Report export failed (${response.status}).`);
        download(`EdgeLens-${state.run.id}.${format}`, await response.blob(), response.headers.get('content-type'));
      }
      toast('Report ready to save.');
    } catch (cause) { error(describeError(cause)); }
    return;
  }
  const report = state.report;
  if (format === 'json') download('EdgeLens-ILLUSTRATIVE-DEMO.json', JSON.stringify(report, null, 2), 'application/json');
  else if (format === 'csv') { const rows = [['ILLUSTRATIVE DEMO - FABRICATED VALUES - NOT MEASURED'], ['Profile', 'Accuracy %', 'Agreement %', 'Latency median ms', 'Latency p95 ms', 'Conversion seconds'], ...report.metrics.map(metric => [metric.label, metric.accuracy_pct, metric.agreement_pct, metric.latency_p50_ms, metric.latency_p95_ms, metric.conversion_seconds ?? ''])]; download('EdgeLens-ILLUSTRATIVE-DEMO.csv', rows.map(row => row.map(value => `"${String(value).replaceAll('"', '""')}"`).join(',')).join('\r\n'), 'text/csv;charset=utf-8'); }
  else download('EdgeLens-ILLUSTRATIVE-DEMO.html', `<!doctype html><html lang="en"><meta charset="utf-8"><title>EdgeLens illustrative report</title><body style="font-family:system-ui;max-width:900px;margin:40px auto;padding:20px"><h1>EdgeLens — Illustrative demo</h1><p><strong>All values are fabricated. No models were executed.</strong></p><h2>${escapeHtml(displayName(report.model))}</h2><p>${escapeHtml(summaryText(report.summary))}</p><pre style="white-space:pre-wrap">${escapeHtml(JSON.stringify(report, null, 2))}</pre></body></html>`, 'text/html;charset=utf-8');
  toast('Illustrative file prepared. It is not a measured report.');
}
document.querySelectorAll('[data-page]').forEach(button => button.addEventListener('click', () => navigate(button.dataset.page)));
document.querySelectorAll('[data-goto]').forEach(button => button.addEventListener('click', () => navigate(button.dataset.goto)));
document.querySelectorAll('[data-export]').forEach(button => button.addEventListener('click', () => exportReport(button.dataset.export)));
$('#demo-mode').addEventListener('click', () => switchMode('demo')); $('#real-mode').addEventListener('click', () => switchMode('real'));
$('#connect-server').addEventListener('click', connect); $('#benchmark-form').addEventListener('submit', startBenchmark); $('#upload-dataset').addEventListener('click', uploadDataset);
$('#run-target').addEventListener('change', updateTarget); $('#run-format').addEventListener('change', modelChanged); $('#dataset-select').addEventListener('change', updateDatasetDescription);
$('#layer-search').addEventListener('input', renderLayers); $('#layer-filter').addEventListener('change', renderLayers);
$('#refresh-history').addEventListener('click', async () => { try { error(''); await refreshHistory(); } catch (cause) { error(describeError(cause)); } });
$('#run-history').addEventListener('click', event => { const button = event.target.closest('[data-run-id]'); if (button) loadRun(button.dataset.runId); });
$('#resume-poll').addEventListener('click', () => { if (state.run?.id) loadRun(state.run.id); });
$('#model-artifacts').addEventListener('click', async event => {
  const button = event.target.closest('[data-artifact]'); if (!button || state.mode !== 'real') return;
  const index = Number(button.dataset.artifact), artifact = state.report?.artifacts?.[index], id = state.run?.id;
  if (!artifact || !id) return;
  button.disabled = true;
  try {
    const response = await fetch(api.artifactUrl(id, index), { signal: AbortSignal.timeout(120000) });
    if (!response.ok) throw new Error(`Model download failed (${response.status}).`);
    download(`${artifact.profile}.${artifact.format === 'pytorch_state_dict' ? 'pt' : artifact.format}`, await response.blob(), 'application/octet-stream');
  } catch (cause) { error(describeError(cause)); }
  finally { button.disabled = false; }
});
window.addEventListener('hashchange', () => showPage(location.hash.slice(1)));
const initialPage = location.hash.slice(1) || 'overview';
updateTarget(); renderReport();
if (desktop) {
  document.title = 'EdgeLens — Model Conversion Studio';
  $('#real-mode').textContent = 'Local benchmark';
  document.querySelector('.setup-panel .panel-heading p').textContent = 'Runs on this computer. Choose an edge device as your deployment goal.';
  document.querySelector('#history-panel .panel-heading p').textContent = 'Saved on this computer using SQLite.';
  document.querySelector('#run-progress .status').textContent = 'This computer · CPU';
  switchMode('real');
}
navigate(initialPage);


function renderModels(selected) {
  const previous = selected || $('#run-model').value;
  $('#run-model').innerHTML = '<option value="mobilenet_v2">MobileNetV2 · sample preset</option><option value="resnet18">ResNet18 · sample preset</option>' + (state.models || []).map(model => `<option value="${escapeHtml(model.id)}">${escapeHtml(model.name)} · ${escapeHtml(model.format.toUpperCase())}</option>`).join('');
  if ([...$('#run-model').options].some(x => x.value === previous)) $('#run-model').value = previous;
  modelChanged();
}
function modelChanged() {
  const model = (state.models || []).find(x => x.id === $('#run-model').value);
  if (model && model.format !== 'pt2') $('#run-format').value = model.format;
  $('#run-format').disabled = Boolean(model && model.format !== 'pt2');
  const searchSupported = model?.format === 'pt2' && $('#run-format').value === 'onnx';
  const quantSupported = ['pt2', 'onnx'].includes(model?.format) && $('#run-format').value === 'onnx';
  $('#run-strategy').querySelector('[value="fidelity_search"]').disabled = !searchSupported;
  $('#run-strategy').querySelector('[value="quantization_compare"]').disabled = !quantSupported;
  $('#run-strategy').disabled = !searchSupported && !quantSupported;
  if ((!searchSupported && $('#run-strategy').value === 'fidelity_search') || (!quantSupported && $('#run-strategy').value === 'quantization_compare')) $('#run-strategy').value = 'fixed_profiles';
  $('#model-detail').textContent = model ? `${model.input_shape.join(' × ')} · ${model.layout} · ${model.class_count} classes · SHA256 ${model.sha256.slice(0, 16)}…` : 'Optional pretrained preset. Upload your exported classifier to use your own architecture and class order.';
  const quantizing = $('#run-strategy').value === 'quantization_compare';
  $('#calibration-field').hidden = $('#run-strategy').value === 'fixed_profiles';
  $('#validation-field').hidden = !quantizing;
  $('#quantization-controls').hidden = !quantizing;
  document.querySelector('label[for="dataset-select"]').textContent = quantizing ? 'Held-out test images · final evaluation' : 'Labelled image dataset';
  $('#strategy-help').textContent = quantizing ? 'Upload a PT2 or FP32 ONNX classifier. INT8 uses calibration images only. Validation and held-out test metrics are recorded separately; this experiment does not select a winner.' : 'Fidelity search requires an uploaded PT2 model, ONNX output and separate calibration images. It may select the standard converter when no better numerical match is found.';
  updateReadiness();
}
function renderCalibration() {
  for (const selector of ['#calibration-select', '#validation-select']) {
    const previous = $(selector).value;
    $(selector).innerHTML = '<option value="">Choose a separate dataset</option>' + state.datasets.map(x => `<option value="${escapeHtml(x.id)}">${escapeHtml(x.name)} · ${x.image_count} images</option>`).join('');
    if (state.datasets.some(x => x.id === previous)) $(selector).value = previous;
  }
}
async function uploadModel() {
  const file = $('#model-file').files[0];
  if (!file) return error('Choose an exported .pt2, .onnx or .tflite model.');
  const format = file.name.split('.').pop().toLowerCase();
  if (!['pt2', 'onnx', 'tflite'].includes(format) || file.size > 100 * 1024 * 1024) return error('Supported exports: .pt2, .onnx, .tflite, up to 100 MiB.');
  if (!$('#model-trusted').checked) return error('Confirm this is your own trusted model before uploading.');
  const numbers = id => $(id).value.split(',').map(x => Number(x.trim()));
  const spec = { name: $('#model-name').value.trim() || file.name, format, input_shape: numbers('#model-shape'), layout: $('#model-layout').value,
    class_count: Number($('#model-classes').value), scale: Number($('#model-scale').value), mean: numbers('#model-mean'), std: numbers('#model-std'),
    resize: $('#model-resize').value, resize_shorter: $('#model-resize').value === 'shortest_center_crop' ? Number($('#model-shorter').value) : null, trusted_source: true };
  $('#upload-model').disabled = true; error(''); $('#model-upload-status').textContent = 'Saving exported model and preprocessing settings…';
  try {
    const saved = await api.uploadModel(file, spec); state.models = await api.models(); renderModels(saved.id);
    $('#model-upload-status').textContent = `Saved ${saved.name}. Signature and runtime compatibility are checked when you run it.`;
    $('#model-file').value = ''; $('#model-trusted').checked = false;
  } catch (cause) { error(describeError(cause)); $('#model-upload-status').textContent = 'Upload not completed.'; }
  finally { $('#upload-model').disabled = false; }
}
function developerTable(rows, keys) {
  if (!rows.length) return '<p class="field-help">No evidence recorded for this section.</p>';
  return `<div class="table-scroll"><table><thead><tr>${keys.map(k => `<th>${escapeHtml(human(k))}</th>`).join('')}</tr></thead><tbody>${rows.map(row => `<tr>${keys.map(k => `<td>${escapeHtml(typeof row[k] === 'number' ? (Number.isInteger(row[k]) ? row[k] : row[k].toPrecision(7)) : row[k] ?? '—')}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
}
function renderDeveloperReport() {
  const report = state.report;
  $('#precision-candidates').hidden = !report?.candidates?.length || state.mode === 'demo';
  $('#developer-evidence').hidden = !report || state.mode === 'demo';
  if (report && state.mode !== 'demo') {
    const selection = report.selection;
    $('#selection-summary').textContent = report.experiment_type ? 'Fixed FP32/INT8 comparison. No automatic selection and no test-based tuning. Calibration, validation and test evidence are retained separately.' : selection ? `${selection.algorithm}: selected ${human(selection.selected)}. ${selection.objective} Total search: ${number(selection.total_strategy_seconds, 3)} s.` : 'No calibration-based search in this run. Imported models need a PyTorch reference to establish conversion loss.';
    $('#selection-evidence').innerHTML = developerTable(selection?.candidates || [], ['id', 'tolerance_failure_count', 'output_mae', 'output_max_abs', 'export_optimize', 'runtime_optimize']);
    const resolution = report.accuracy_resolution_pp ?? (report.dataset?.image_count ? 100 / report.dataset.image_count : null);
    $('#accuracy-resolution').textContent = `One changed prediction = ${number(resolution, 6)} percentage points on this test set. Output MAE is a separate numerical metric. Small timing differences require repeated runs.`;
    $('#prediction-evidence').innerHTML = developerTable((report.predictions || []).filter(x => x.profile !== 'pytorch').slice(0, 20), ['profile', 'image', 'label', 'prediction', 'reference_prediction', 'max_abs', 'within_tolerance']);
    $('#candidate-evidence').innerHTML = developerTable((report.candidates || []).map(candidate => ({ name: candidate.name, status: candidate.status, validation_accuracy_pct: candidate.validation_metrics?.accuracy_pct, test_accuracy_pct: candidate.test_metrics?.accuracy_pct, latency_p50_ms: candidate.test_metrics?.latency_p50_ms, qdq_operations: `${candidate.inventory?.qdq_operation_count ?? '—'} / ${candidate.inventory?.eligible_operation_count ?? '—'}`, failure: candidate.failure ? `${candidate.failure.stage}: ${candidate.failure.message}` : '—' })), ['name', 'status', 'validation_accuracy_pct', 'test_accuracy_pct', 'latency_p50_ms', 'qdq_operations', 'failure']);
    $('#reproduce-evidence').textContent = JSON.stringify({ model_sha256: report.model?.sha256, dataset_sha256: report.dataset?.sha256, datasets: report.datasets, candidates: report.candidates, settings: report.settings, preprocessing: report.model, versions: report.environment?.versions }, null, 2);
  }
  renderEdge();
}
function renderEdge() {
  const active = state.mode === 'real' && state.run?.status === 'completed' && state.report;
  $('.device-latency strong').textContent = 'Not measured';
  $('#edge-workspace').hidden = !active;
  $('#edge-empty').hidden = Boolean(active);
  if (!active) return;
  $('#edge-run-label').textContent = `${displayName(state.report.model)} · ${state.run.id}`;
  const previous = $('#edge-artifact').value;
  $('#edge-artifact').innerHTML = '<option value="">Select a TFLite artifact</option>' + (state.report.artifacts || []).map((x, i) => x.format === 'tflite' ? `<option value="${i}">${escapeHtml(human(x.profile))} · ${number(x.size_bytes / 1024, 1)} KiB · ${escapeHtml(x.sha256.slice(0, 12))}…</option>` : '').join('');
  if ([...$('#edge-artifact').options].some(x => x.value === previous)) $('#edge-artifact').value = previous;
  $('#prepare-edge').disabled = !state.report.artifacts?.some(x => x.format === 'tflite');
  $('#submit-ei').disabled = $('#prepare-edge').disabled;
  $('#edge-format-note').textContent = $('#prepare-edge').disabled ? 'This run has no TFLite artifact. Import and benchmark a TFLite classifier to prepare ESP32 firmware or request Edge Impulse analysis.' : 'Choose the exact TFLite artifact to associate with device and provider evidence.';
  const oldPackage = $('#edge-package').value;
  const records = state.edgeRecords || [];
  $('#edge-package').innerHTML = '<option value="">Select a prepared firmware package</option>' + records.filter(x => x.kind === 'package').map(x => `<option value="${escapeHtml(x.id)}">${escapeHtml(human(x.profile))} · ${escapeHtml(x.id.slice(0, 16))} · ${x.arena_capacity_bytes / 1024} KiB arena</option>`).join('');
  if (records.some(x => x.id === oldPackage)) $('#edge-package').value = oldPackage;
  const oldJob = $('#ei-job').value;
  $('#ei-job').innerHTML = '<option value="">Select a submitted profile job</option>' + records.filter(x => x.kind === 'edge_impulse_job').map(x => `<option value="${escapeHtml(x.id)}">${escapeHtml(x.device)} · job ${x.job_id}</option>`).join('');
  if (records.some(x => x.id === oldJob)) $('#ei-job').value = oldJob;
  const results = state.report.edge_results || [];
  const hardware = results.filter(x => x.kind === 'hardware');
  $('#hardware-evidence').innerHTML = developerTable(hardware.map(x => ({ ...x, chip: x.device_report?.chip })), ['profile', 'chip', 'source', 'latency_p50_ms', 'latency_p95_ms', 'arena_used_bytes', 'output_max_abs', 'within_tolerance']);
  $('#ei-evidence').innerHTML = results.filter(x => x.kind === 'edge_impulse_result').map(x => `<article class="provider-result"><strong>Edge Impulse · provider analysis · ${escapeHtml(x.device)}</strong><pre>${escapeHtml(JSON.stringify(x.provider_result, null, 2))}</pre><p>Provider estimates are separate from your board measurements.</p></article>`).join('') || '<p class="field-help">No Edge Impulse analysis recorded. An account and explicit model upload are required.</p>';
  const latest = hardware.at(-1);
  $('.device-latency strong').textContent = latest ? `${number(latest.latency_p50_ms, 3)} ms · ${latest.device_report.chip}` : 'Not measured';
}
async function refreshEdgeRecords() {
  const id = state.run?.id;
  if (!id || state.run?.status !== 'completed' || state.mode !== 'real') return;
  try { const records = await api.edgeRecords(id); if (state.run?.id !== id || state.mode !== 'real') return; state.edgeRecords = records; renderEdge(); }
  catch (cause) { if (state.run?.id === id) error(describeError(cause)); }
}
async function edgeAction(button, action) {
  if (state.mode !== 'real' || state.run?.status !== 'completed') return error('Open a completed real benchmark first.');
  error(''); button.disabled = true; $('#edge-status').textContent = 'Working…';
  try { const message = await action(); $('#edge-status').textContent = message; }
  catch (cause) { error(describeError(cause)); $('#edge-status').textContent = 'Action not completed.'; }
  finally { button.disabled = false; }
}
async function reloadEdgeReport() { const id = state.run.id; const run = await api.run(id); if (state.run?.id === id) { state.report = run.report; renderReport(); await refreshEdgeRecords(); } }
function selectedArtifact() { if ($('#edge-artifact').value === '') throw new Error('Select a TFLite artifact first.'); return Number($('#edge-artifact').value); }
function selectedPackage() { if (!$('#edge-package').value) throw new Error('Prepare and select a firmware package first.'); return $('#edge-package').value; }
async function savePackage(id) {
  const response = await fetch(api.packageUrl(id)); if (!response.ok) throw new Error('Firmware package download failed.');
  download(`EdgeLens-${id}.zip`, await response.blob(), 'application/zip');
}
$('#upload-model').addEventListener('click', uploadModel);
$('#run-model').addEventListener('change', modelChanged);
$('#run-strategy').addEventListener('change', modelChanged); $('#calibration-select').addEventListener('change', updateReadiness);
$('#validation-select').addEventListener('change', updateReadiness);
$('#prepare-edge').addEventListener('click', event => edgeAction(event.currentTarget, async () => {
  const value = await api.preparePackage(state.run.id, {artifact_index: selectedArtifact(), arena_kib: Number($('#arena-kib').value)});
  await refreshEdgeRecords(); $('#edge-package').value = value.id; await savePackage(value.id);
  return 'Firmware package saved. Build it with ESP-IDF and flash your confirmed chip, then capture a report here. No firmware was flashed by EdgeLens.';
}));
$('#download-edge').addEventListener('click', event => edgeAction(event.currentTarget, async () => { await savePackage(selectedPackage()); return 'Firmware package ready to save.'; }));
$('#scan-ports').addEventListener('click', event => edgeAction(event.currentTarget, async () => {
  const ports = await api.ports(); $('#edge-port').innerHTML = '<option value="">Select a USB serial port</option>' + ports.map(x => `<option value="${escapeHtml(x.port)}">${escapeHtml(x.port)} · ${escapeHtml(x.description)}</option>`).join('');
  return ports.length ? 'Ports listed. A COM port is not proof of the chip variant; confirm the board before flashing.' : 'No COM ports detected. Check the USB data cable, USB-UART driver, and board connection.';
}));
$('#capture-edge').addEventListener('click', event => edgeAction(event.currentTarget, async () => {
  await api.captureSerial(selectedPackage(), $('#edge-port').value); await reloadEdgeReport(); return 'Matching USB device report saved in SQLite and attached to this run.';
}));
$('#import-edge').addEventListener('click', event => edgeAction(event.currentTarget, async () => {
  const file = $('#edge-json').files[0]; if (!file || file.size > 200000) throw new Error('Choose a device JSON report up to 200 KB.');
  await api.importDeviceReport(selectedPackage(), JSON.parse(await file.text())); await reloadEdgeReport(); return 'Imported device report saved with its import provenance.';
}));
$('#submit-ei').addEventListener('click', event => edgeAction(event.currentTarget, async () => {
  if (!$('#ei-consent').checked) throw new Error('Confirm you want to send this model to Edge Impulse.');
  const job = await api.startProfile(state.run.id, {artifact_index: selectedArtifact(), project_id: Number($('#ei-project').value), device: $('#ei-device').value.trim(), api_key: $('#ei-key').value, consent_upload: true});
  await refreshEdgeRecords(); $('#ei-job').value = job.id; return 'Edge Impulse job submitted. Check the result after processing finishes.';
}));
$('#refresh-ei').addEventListener('click', event => edgeAction(event.currentTarget, async () => {
  if (!$('#ei-job').value) throw new Error('Select a submitted Edge Impulse job.');
  await api.refreshProfile($('#ei-job').value, $('#ei-key').value); await reloadEdgeReport(); return 'Edge Impulse analysis saved separately from hardware measurements.';
}));
