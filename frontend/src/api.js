import { parseMetricsCsv } from './metrics.js';

const configuredBase = import.meta.env.VITE_API_URL?.trim().replace(/\/$/, '') || '';

function errorMessage(detail, status) {
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail.map((item) => {
      const field = Array.isArray(item.loc) ? item.loc.slice(1).join('.') : '';
      return `${field ? `${field}: ` : ''}${item.msg || 'Dữ liệu không hợp lệ'}`;
    }).join('; ');
  }
  return `Máy chủ trả về lỗi ${status}.`;
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${configuredBase}${path}`, options);
  } catch {
    throw new Error('Không kết nối được backend. Hãy chạy FastAPI và kiểm tra địa chỉ API.');
  }
  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new Error(`Phản hồi từ backend không phải JSON (HTTP ${response.status}).`);
  }
  if (!response.ok) throw new Error(errorMessage(payload.detail, response.status));
  return payload;
}

function post(path, body) {
  return request(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

export const health = () => request('/health');
export const simulate = (body) => post('/api/simulate', body);
export const summarize = (body) => post('/api/summarize', body);
export const evaluate = (body) => post('/api/evaluate', body);

export async function loadBatchMetrics() {
  const response = await fetch(`${import.meta.env.BASE_URL}experiment/metrics.csv`, { cache: 'no-store' });
  if (!response.ok) throw new Error('Không đọc được results/metrics.csv. Hãy chạy lại thí nghiệm hoặc build frontend.');
  return parseMetricsCsv(await response.text());
}
