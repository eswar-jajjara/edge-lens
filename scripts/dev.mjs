import http from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(fileURLToPath(new URL('../frontend/', import.meta.url)));
const port = Number(process.env.PORT || 5173);
const types = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.svg': 'image/svg+xml', '.json': 'application/json' };
const server = http.createServer(async (req, res) => {
  if (req.url === '/config.js' && process.env.DESKTOP_PREVIEW === '1') {
    res.writeHead(200, { 'Content-Type': 'application/javascript', 'Cache-Control': 'no-store' });
    return res.end("window.EDGELENS_CONFIG = Object.freeze({apiBaseUrl:'/api/v1',desktop:true});");
  }
  if (req.url.startsWith('/api/')) {
    const upstream = http.request({ hostname: '127.0.0.1', port: Number(process.env.API_PORT || 8000), path: req.url, method: req.method, headers: req.headers }, response => {
      res.writeHead(response.statusCode, response.headers); response.pipe(res);
    });
    upstream.on('error', () => { if (!res.headersSent) res.writeHead(502, { 'Content-Type': 'application/json' }); res.end(JSON.stringify({ detail: 'The local Python engine is not running on port8000.' })); });
    req.pipe(upstream); return;
  }
  if (!['GET', 'HEAD'].includes(req.method)) { res.writeHead(405, { Allow: 'GET, HEAD' }); return res.end(); }
  try {
    const urlPath = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
    const relative = urlPath === '/' ? 'index.html' : urlPath.slice(1);
    const filename = path.resolve(root, relative === 'config.js' ? 'public/config.js' : relative);
    if (!filename.startsWith(root + path.sep)) { res.writeHead(403); return res.end('Forbidden'); }
    if (!(await stat(filename)).isFile()) throw new Error('Not a file');
    const content = await readFile(filename);
    res.writeHead(200, { 'Content-Type': types[path.extname(filename)] || 'application/octet-stream', 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff' });
    res.end(req.method === 'HEAD' ? undefined : content);
  } catch { res.writeHead(404, { 'Content-Type': 'text/plain' }); res.end('Not found'); }
});
server.listen(port, '127.0.0.1', () => console.log(`EdgeLens development preview: http://127.0.0.1:${port}`));
