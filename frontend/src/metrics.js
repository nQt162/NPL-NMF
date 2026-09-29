const METRIC_FIELDS = ['rouge1_f1', 'rouge2_f1', 'rougeL_f1', 'runtime_seconds', 'sentence_count'];

export function parseMetricsCsv(text) {
  if (!text.trim()) return [];
  const rows = [];
  let row = [];
  let cell = '';
  let quoted = false;
  const source = text.replace(/^\uFEFF/, '');

  for (let index = 0; index < source.length; index += 1) {
    const character = source[index];
    if (character === '"') {
      if (quoted && source[index + 1] === '"') {
        cell += '"';
        index += 1;
      } else {
        quoted = !quoted;
      }
    } else if (character === ',' && !quoted) {
      row.push(cell);
      cell = '';
    } else if ((character === '\n' || character === '\r') && !quoted) {
      if (character === '\r' && source[index + 1] === '\n') index += 1;
      row.push(cell);
      if (row.some((value) => value !== '')) rows.push(row);
      row = [];
      cell = '';
    } else {
      cell += character;
    }
  }
  if (quoted) throw new Error('metrics.csv có dấu ngoặc kép chưa đóng.');
  row.push(cell);
  if (row.some((value) => value !== '')) rows.push(row);
  if (rows.length < 2) return [];

  const headers = rows.shift().map((value) => value.trim());
  const required = ['id', 'method', ...METRIC_FIELDS];
  if (required.some((field) => !headers.includes(field))) {
    throw new Error('metrics.csv thiếu cột bắt buộc: id, method hoặc chỉ số đánh giá.');
  }
  return rows.map((values, index) => {
    const record = Object.fromEntries(headers.map((header, column) => [header, values[column] ?? '']));
    if (values.length !== headers.length) throw new Error(`Dòng ${index + 2} trong metrics.csv sai số cột.`);
    for (const field of METRIC_FIELDS) {
      record[field] = Number(record[field]);
      if (!Number.isFinite(record[field])) throw new Error(`Dòng ${index + 2}: ${field} không phải số.`);
    }
    return record;
  });
}

export function summarizeMetrics(records) {
  const groups = new Map();
  for (const record of records) {
    if (!groups.has(record.method)) groups.set(record.method, []);
    groups.get(record.method).push(record);
  }
  return [...groups.entries()].map(([method, items]) => ({
    method,
    count: items.length,
    fallbackCount: items.filter((item) => item.fallback_reason).length,
    ...Object.fromEntries(METRIC_FIELDS.map((field) => [
      field,
      items.reduce((total, item) => total + item[field], 0) / items.length,
    ])),
  })).sort((a, b) => (a.method === 'Lead-N' ? -1 : b.method === 'Lead-N' ? 1 : a.method.localeCompare(b.method)));
}
