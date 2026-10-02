'use strict';
(function () {
  const base = () => (window.EDGELENS_CONFIG?.apiBaseUrl || '/api/v1').replace(/\/$/, '');
  const runPath = id => `/runs/${encodeURIComponent(id)}`;
  async function request(path, options = {}) {
    const response = await fetch(`${base()}${path}`, { ...options, headers: { Accept: 'application/json', ...options.headers }, signal: options.signal || AbortSignal.timeout(15000) });
    const body = await response.json().catch(() => null);
    if (!response.ok) {
      const detail = body?.detail;
      const message = typeof detail === 'string' ? detail : detail?.message;
      const validation = Array.isArray(detail) ? detail.map(item => `${(item.loc || []).slice(1).join('.')}: ${item.msg}`).join('; ') : null;
      const error = new Error(message || validation || `API request failed (${response.status})`);
      error.status = response.status;
      error.code = detail?.code;
      throw error;
    }
    if (response.status === 204) return null;
    if (body === null) throw new Error('The server returned an unreadable response. Check the API address.');
    return body;
  }
  window.EdgeLensAPI = Object.freeze({
    health: () => request('/health'), capabilities: () => request('/capabilities'), datasets: () => request('/datasets'),
    uploadDataset: (file, name) => request('/datasets', { method: 'POST', headers: { 'Content-Type': 'application/zip', 'X-Dataset-Name': name.replace(/[^\x20-\x7e]/g, '_') }, body: file, signal: AbortSignal.timeout(120000) }),
    runs: () => request('/runs'), run: id => request(runPath(id)), report: id => request(`${runPath(id)}/report`),
    deleteRun: id => request(runPath(id), {method:'DELETE'}),
    importWorkerReport: file => request('/worker-reports', {method:'POST', headers:{'Content-Type':'application/zip'}, body:file, signal:AbortSignal.timeout(120000)}),
    diagnosticSelfTest: () => request('/diagnostics/self-test', {method:'POST', signal:AbortSignal.timeout(120000)}),
    artifactUrl: (id, index) => { if (!Number.isInteger(index) || index < 0) throw new Error('Invalid artifact index'); return `${base()}${runPath(id)}/artifacts/${index}`; },
    reportUrl: (id, format) => { if (!['html', 'csv', 'json'].includes(format)) throw new Error('Unsupported report format'); return `${base()}${runPath(id)}/report${format === 'json' ? '' : `.${format}`}`; },
    models: () => request('/models'),
    uploadModel: (file, spec) => request('/models', {method:'POST', headers:{'Content-Type':'application/octet-stream','X-Model-Spec':encodeURIComponent(JSON.stringify(spec))},body:file,signal:AbortSignal.timeout(120000)}),
    ports: () => request('/edge/ports'), edgeRecords: id => request(`${runPath(id)}/edge`),
    preparePackage: (id,payload) => request(`${runPath(id)}/edge/packages`, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload),signal:AbortSignal.timeout(60000)}),
    packageUrl: id => `${base()}/edge/packages/${encodeURIComponent(id)}/download`,
    captureSerial: (id,port) => request(`/edge/packages/${encodeURIComponent(id)}/capture`, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({port}),signal:AbortSignal.timeout(20000)}),
    importDeviceReport: (id,payload) => request(`/edge/packages/${encodeURIComponent(id)}/import`, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}),
    impulseConnections: () => request('/edge/impulse/connections'),
    connectImpulse: api_key => request('/edge/impulse/connections', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({api_key}),signal:AbortSignal.timeout(60000)}),
    disconnectImpulse: id => request(`/edge/impulse/connections/${encodeURIComponent(id)}`, {method:'DELETE'}),
    profileTargets: (project_id,credential) => request('/edge/impulse/targets', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({project_id,...(typeof credential === 'string' ? {api_key:credential} : credential)}),signal:AbortSignal.timeout(60000)}),
    startProfile: (id,payload) => request(`${runPath(id)}/edge/impulse`, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload),signal:AbortSignal.timeout(60000)}),
    refreshProfile: (id,credential) => request(`/edge/impulse/${encodeURIComponent(id)}/refresh`, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(typeof credential === 'string' ? {api_key:credential} : credential),signal:AbortSignal.timeout(60000)}),
    createRun: payload => request('/runs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }),
  });
})();
