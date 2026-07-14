import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const root = new URL('..', import.meta.url);

test('loads a read-only Monaco enhancer while retaining the Rouge fallback', async () => {
  const [layout, script] = await Promise.all([
    readFile(new URL('_layouts/default.html', root), 'utf8'),
    readFile(new URL('assets/js/monaco-readonly.js', root), 'utf8').catch(() => ''),
  ]);

  assert.match(layout, /assets\/js\/monaco-readonly\.js[^>]*defer/);
  assert.match(script, /matchMedia\('\(min-width: 768px\)'\)/);
  assert.match(script, /querySelectorAll\('\.highlight'\)/);
  assert.match(script, /readOnly:\s*true/);
  assert.match(script, /fallback\.hidden\s*=\s*true/);
  assert.match(
    script,
    /fallback\.querySelector\('\.rouge-code pre'\) \|\| fallback\.querySelector\('pre'\)/,
  );
});
