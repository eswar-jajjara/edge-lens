import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { readFile } from 'node:fs/promises';

const source = await readFile(new URL('../frontend/src/api.js', import.meta.url), 'utf8');
function client(fetch) {
  const context = { window: { EDGELENS_CONFIG: { apiBaseUrl: 'https://api.example/api/v1/' } }, fetch, AbortSignal };
  vm.runInNewContext(source, context);
  return context.window.EdgeLensAPI;
}
const success = body => ({ ok: true, json: async () => body });

test('provider target discovery keeps the transient key in the request body', async () => {
  let requestedUrl, options;
  const api = client(async (url, value) => { requestedUrl = url; options = value; return success([{mcu:'target',name:'Target'}]); });
  const targets = await api.profileTargets(7, 'test_credential');
  assert.equal(requestedUrl, 'https://api.example/api/v1/edge/impulse/targets');
  assert.equal(options.method, 'POST');
  assert.deepEqual(JSON.parse(options.body), {project_id:7,api_key:'test_credential'});
  assert.equal(targets[0].mcu, 'target');
  assert.ok(!requestedUrl.includes('test_credential'));
});

test('configured API base is used for capabilities, datasets and history', async () => {
  const urls = [], api = client(async url => { urls.push(url); return success([]); });
  await api.health(); await api.capabilities(); await api.datasets(); await api.runs();
  assert.deepEqual(urls, ['health', 'capabilities', 'datasets', 'runs'].map(path => `https://api.example/api/v1/${path}`));
});

test('unavailable runtime failures surface without creating demo success', async () => {
  const api = client(async () => ({ ok: false, status: 503, json: async () => ({ detail: { code: 'RUNTIME_UNAVAILABLE', message: 'Install ONNX dependencies' } }) }));
  await assert.rejects(api.createRun({}), error => error.status === 503 && error.code === 'RUNTIME_UNAVAILABLE' && error.message === 'Install ONNX dependencies');
});

test('createRun sends complete model, dataset, target and settings as JSON', async () => {
  let options;
  const payload = { model_id: 'mobilenet_v2', format: 'onnx', target: 'raspberry_pi', dataset_id: 'dataset-1', settings: { warmup_runs: 3, measured_runs: 10, threads: 1, atol: 0.0001, rtol: 0.001 } };
  const api = client(async (url, value) => { options = value; return success({ id: 'run-1', status: 'queued' }); });
  assert.equal((await api.createRun(payload)).status, 'queued');
  assert.equal(options.method, 'POST'); assert.equal(options.headers['Content-Type'], 'application/json');
  assert.deepEqual(JSON.parse(options.body), payload);
});

test('dataset upload transmits ZIP body directly with upload metadata', async () => {
  let options, requestedUrl;
  const file = new Uint8Array([80, 75, 3, 4]);
  const api = client(async (url, value) => { requestedUrl = url; options = value; return success({ id: 'dataset-1', image_count: 2 }); });
  await api.uploadDataset(file, 'my images');
  assert.equal(requestedUrl, 'https://api.example/api/v1/datasets'); assert.equal(options.body, file);
  assert.equal(options.method, 'POST'); assert.equal(options.headers['Content-Type'], 'application/zip'); assert.equal(options.headers['X-Dataset-Name'], 'my images');
});

test('run IDs are encoded in detail and export URLs', async () => {
  const urls = [], api = client(async url => { urls.push(url); return success({}); });
  await api.run('run/1?other'); await api.report('run/1?other');
  assert.equal(urls[0], 'https://api.example/api/v1/runs/run%2F1%3Fother');
  assert.equal(urls[1], 'https://api.example/api/v1/runs/run%2F1%3Fother/report');
  assert.equal(api.reportUrl('run-1', 'html'), 'https://api.example/api/v1/runs/run-1/report.html');
  assert.equal(api.reportUrl('run-1', 'csv'), 'https://api.example/api/v1/runs/run-1/report.csv');
  assert.equal(api.reportUrl('run-1', 'json'), 'https://api.example/api/v1/runs/run-1/report');
  assert.throws(() => api.reportUrl('run-1', '../unexpected'), /Unsupported report format/);
});

test('structured request validation failures identify the bad setting', async () => {
  const api = client(async () => ({ ok: false, status: 422, json: async () => ({ detail: [{ loc: ['body', 'settings', 'measured_runs'], msg: 'Input should be greater than 0' }] }) }));
  await assert.rejects(api.createRun({}), /settings.measured_runs: Input should be greater than 0/);
});

test('string API errors remain actionable', async () => {
  const api = client(async () => ({ ok: false, status: 400, json: async () => ({ detail: 'Missing labels.json' }) }));
  await assert.rejects(api.uploadDataset({}, 'images'), /Missing labels.json/);
});

test('non-JSON API failures produce a status-specific error', async () => {
  const api = client(async () => ({ ok: false, status: 502, json: async () => { throw new SyntaxError(); } }));
  await assert.rejects(api.health(), /API request failed \(502\)/);
});

test('an HTML fallback page cannot be treated as a successful API response', async () => {
  const api = client(async () => ({ ok: true, status: 200, json: async () => { throw new SyntaxError(); } }));
  await assert.rejects(api.capabilities(), /unreadable response/);
});

test('developer model upload preserves explicit settings and binary bytes', async () => {
  let seen;
  const api = client(async (url, options) => { seen = {url,options}; return success({id:'model_one'}); });
  const spec = {name:'Developer classifier',format:'pt2',input_shape:[1,3,32,32],class_count:2,trusted_source:true};
  const binary = new Uint8Array([1,2,3]);
  await api.uploadModel(binary,spec);
  assert.equal(seen.url,'https://api.example/api/v1/models');
  assert.equal(seen.options.body,binary);
  assert.deepEqual(JSON.parse(decodeURIComponent(seen.options.headers['X-Model-Spec'])),spec);
});

test('hardware capture is bound to an encoded prepared package and explicit port', async () => {
  let seen;
  const api = client(async (url,options) => { seen = {url,options}; return success({source:'usb_serial_capture'}); });
  await api.captureSerial('pkg/test','COM7');
  assert.equal(seen.url,'https://api.example/api/v1/edge/packages/pkg%2Ftest/capture');
  assert.equal(seen.options.method,'POST');
  assert.deepEqual(JSON.parse(seen.options.body),{port:'COM7'});
});

test('provider key is sent in a POST body and never embedded in a URL', async () => {
  let seen;
  const api = client(async (url,options) => { seen = {url,options}; return success({}); });
  await api.refreshProfile('edge_job','test_fixture_secret');
  assert.equal(seen.url,'https://api.example/api/v1/edge/impulse/edge_job/refresh');
  assert.equal(seen.options.method,'POST');
  assert.deepEqual(JSON.parse(seen.options.body),{api_key:'test_fixture_secret'});
  assert.ok(!seen.url.includes('test_fixture_secret'));
});

