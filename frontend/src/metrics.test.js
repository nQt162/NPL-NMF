import assert from 'node:assert/strict';
import test from 'node:test';
import { parseMetricsCsv, summarizeMetrics } from './metrics.js';

test('reads quoted CSV and averages each method', () => {
  const csv = '\uFEFFid,method,rouge1_f1,rouge2_f1,rougeL_f1,runtime_seconds,sentence_count,fallback_reason\r\n'
    + '"bài, 1",Lead-N,0.5,0.2,0.4,0.01,3,""\r\n'
    + '"bài, 1",NMF,0.7,0.3,0.6,0.2,3,""\r\n'
    + 'bài-2,NMF,0.9,0.5,0.8,0.4,3,"ít từ, fallback"\r\n';
  const rows = parseMetricsCsv(csv);
  assert.equal(rows.length, 3);
  assert.equal(rows[0].id, 'bài, 1');
  const summary = summarizeMetrics(rows);
  assert.equal(summary[0].method, 'Lead-N');
  assert.equal(summary[1].rouge1_f1, 0.8);
  assert.equal(summary[1].fallbackCount, 1);
});

test('handles empty results and rejects malformed rows', () => {
  assert.deepEqual(parseMetricsCsv(''), []);
  assert.throws(() => parseMetricsCsv('id,method,rouge1_f1,rouge2_f1,rougeL_f1,runtime_seconds,sentence_count\na,NMF,no,0,0,0,1'));
});
