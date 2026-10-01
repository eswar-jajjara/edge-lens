import { cp, mkdir, readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = fileURLToPath(new URL('../', import.meta.url));
const output = path.join(root, 'dist');
await mkdir(output, { recursive: true });
// Explicit files only. Does not recursively delete directories or user files.
await cp(path.join(root, 'frontend/index.html'), path.join(output, 'index.html'));
await cp(path.join(root, 'frontend/src'), path.join(output, 'src'), { recursive: true });
await cp(path.join(root, 'frontend/public'), output, { recursive: true });
await writeFile(path.join(output, 'build-info.json'), JSON.stringify({
  version: JSON.parse(await readFile(path.join(root, 'package.json'), 'utf8')).version,
  mode: 'desktop-and-server-interface',
}, null, 2));
console.log(`Frontend built: ${output}`);
