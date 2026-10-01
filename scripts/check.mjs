import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
const root = fileURLToPath(new URL('../', import.meta.url));
const files = ['frontend/public/config.js', 'frontend/src/demo-data.js', 'frontend/src/api.js', 'frontend/src/app.js'];
for (const name of files) new vm.Script(await readFile(path.join(root, name), 'utf8'), { filename: name });
const html = await readFile(path.join(root, 'frontend/index.html'), 'utf8');
for (const [, resource] of html.matchAll(/(?:src|href)="([^"#]+)"/g)) {
  if (/^(data:|https?:)/.test(resource)) continue;
  await readFile(path.join(root, 'frontend', resource === 'config.js' ? 'public/config.js' : resource));
}
console.log('Frontend JavaScript syntax and local asset references passed.');
