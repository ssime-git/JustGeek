import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const post = await readFile(new URL('../_posts/2026-06-27-apprendre-a-coder-en-2026.md', import.meta.url), 'utf8');

test('async demo reaches the intended undefined result and formats its question', () => {
  assert.match(post, /const response = await fetch\(`\/api\/users\/\$\{id\}`\);/);
  assert.match(post, /const data = response\.json\(\);/);
  const asyncQuestion = post.match(/<div class="js-cell-question"[^>]*>([\s\S]*?)<div class="codecoach-callout">/)?.[1] || '';
  assert.doesNotMatch(asyncQuestion, /\*\*/);
  assert.match(asyncQuestion, /Question : <strong>pourquoi <code>data\.name<\/code> est-il <code>undefined<\/code>/);
});
