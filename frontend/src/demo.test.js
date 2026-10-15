import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const demo = JSON.parse(readFileSync(new URL('../../data/demo/toy.json', import.meta.url), 'utf8'));

test('shared demo contains the complete AI article in sentence order', () => {
  assert.equal(demo.doc_id, 'test_nmf_01');
  assert.equal(demo.sentences.length, 12);
  assert.equal(demo.text, demo.sentences.join(' '));
  assert.equal(demo.source, 'synthetic');
});

test('demo reference preserves exactly the four user-provided sentences', () => {
  assert.deepEqual(demo.ground_truth_indices, [0, 4, 7, 9]);
  assert.equal(
    demo.extractive_summary,
    demo.ground_truth_indices.map((index) => demo.sentences[index]).join(' '),
  );
  assert.deepEqual(
    demo.ground_truth_labels,
    demo.sentences.map((_, index) => Number(demo.ground_truth_indices.includes(index))),
  );
});
