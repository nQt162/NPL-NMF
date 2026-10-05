import { useState } from 'react';
import toy from '../../../data/demo/toy.json';
import { summarize } from '../api.js';

function number(value) {
  return Number(value ?? 0).toFixed(3);
}

function methodLabel(result) {
  if (!result) return 'KL + L1';
  return result.loss_name === 'kullback-leibler' ? 'KL divergence + L1' : result.loss_name;
}

export default function Summarizer() {
  const [text, setText] = useState(toy.text);
  const [k, setK] = useState('');
  const [sentenceCount, setSentenceCount] = useState(3);
  const [alpha, setAlpha] = useState(1);
  const [beta, setBeta] = useState(1);
  const [gamma, setGamma] = useState(0.5);
  const [positionWeight, setPositionWeight] = useState(0.15);
  const [lengthWeight, setLengthWeight] = useState(0.1);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [copied, setCopied] = useState(false);

  function updateField(setter, value) {
    setter(value);
    setResult(null);
    setCopied(false);
    setError('');
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setError('');
    setResult(null);
    setCopied(false);
    if (!text.trim()) { setError('Hãy nhập văn bản cần tóm tắt.'); return; }
    setLoading(true);
    try {
      const payload = {
        text,
        summary_sentences: Number(sentenceCount),
        alpha: Number(alpha),
        beta: Number(beta),
        gamma: Number(gamma),
        position_weight: Number(positionWeight),
        length_weight: Number(lengthWeight),
      };
      if (k !== '') payload.k = Number(k);
      const data = await summarize(payload);
      setResult(data);
    } catch (cause) {
      setError(cause.message);
    } finally {
      setLoading(false);
    }
  }

  async function copySummary() {
    try {
      await navigator.clipboard.writeText(result.summary);
      setCopied(true);
    } catch {
      setError('Không sao chép được trong trình duyệt này.');
    }
  }

  return (
    <section aria-labelledby="summarizer-heading">
      <div className="section-heading">
        <div>
          <div className="section-kicker">02 / TÓM TẮT TRÍCH XUẤT</div>
          <h2 id="summarizer-heading">Giữ lại điều quan trọng</h2>
          <p>Fit TF-IDF cục bộ, chọn k bằng masked KL imputation, rồi chọn câu theo salience, coverage và redundancy.</p>
        </div>
        <span className="section-index">02</span>
      </div>

      <div className="workspace-grid">
        <form className="panel form-panel" onSubmit={handleSubmit}>
          <div className="panel-top"><span className="panel-label">VĂN BẢN NGUỒN</span><span className="panel-meta">LOCAL KL-NMF</span></div>
          <label className="field-label" htmlFor="summarize-text">Nội dung cần tóm tắt</label>
          <textarea id="summarize-text" value={text} onChange={(event) => updateField(setText, event.target.value)} rows={12} placeholder="Dán văn bản tiếng Việt vào đây..." />
          <div className="field-hint"><span>{text.length.toLocaleString('vi-VN')} ký tự</span><button type="button" className="text-button" onClick={() => { setText(toy.text); setResult(null); setError(''); }}>Nạp ví dụ</button></div>
          <div className="form-row two-cols">
            <label className="field-group">Số chủ đề k<input type="number" min="1" max="10" value={k} onChange={(event) => updateField(setK, event.target.value)} placeholder="Tự chọn" /></label>
            <label className="field-group">Số câu tóm tắt<input type="number" min="1" value={sentenceCount} onChange={(event) => updateField(setSentenceCount, event.target.value)} required /></label>
          </div>
          <details className="advanced-settings">
            <summary>Trọng số chọn câu <span>↘</span></summary>
            <div className="form-row three-cols">
              <label className="field-group">Salience α<input type="number" min="0" step="0.1" value={alpha} onChange={(event) => updateField(setAlpha, event.target.value)} required /></label>
              <label className="field-group">Coverage β<input type="number" min="0" step="0.1" value={beta} onChange={(event) => updateField(setBeta, event.target.value)} required /></label>
              <label className="field-group">Redundancy γ<input type="number" min="0" step="0.1" value={gamma} onChange={(event) => updateField(setGamma, event.target.value)} required /></label>
              <label className="field-group">Vị trí<input type="number" min="0" step="0.05" value={positionWeight} onChange={(event) => updateField(setPositionWeight, event.target.value)} required /></label>
              <label className="field-group">Độ dài<input type="number" min="0" step="0.05" value={lengthWeight} onChange={(event) => updateField(setLengthWeight, event.target.value)} required /></label>
            </div>
          </details>
          {error && <div className="alert alert-error" role="alert">{error}</div>}
          <button className="primary-button" type="submit" disabled={loading}>{loading ? 'Đang phân tích...' : 'Tạo bản tóm tắt'}<span aria-hidden="true">→</span></button>
        </form>

        <div className="panel insight-panel summarize-insight">
          <div className="panel-top"><span className="panel-label">QUY TRÌNH</span><span className="panel-meta">KL + L1</span></div>
          <div className="process-list">
            <div><span>01</span><p><strong>Tách câu gốc</strong><small>Mỗi câu là một document, vẫn giữ nguyên câu để trích xuất.</small></p></div>
            <div><span>02</span><p><strong>Chọn k</strong><small>Mask một phần TF-IDF dương, fit NMF và chọn k impute tốt nhất.</small></p></div>
            <div><span>03</span><p><strong>NMF KL + L1</strong><small>MU solver, alpha W/H = 0.1, l1 ratio = 1.0.</small></p></div>
            <div><span>04</span><p><strong>Chọn câu greedy</strong><small>Cân bằng salience, coverage, chống lặp và thứ tự gốc.</small></p></div>
          </div>
          <div className="insight-foot">KHÔNG VIẾT LẠI · KHÔNG DÙNG MODEL GLOBAL</div>
        </div>
      </div>

      {!result && !loading && <div className="empty-state"><span className="empty-symbol">≋</span><strong>Bản tóm tắt sẽ xuất hiện ở đây</strong><p>Nhập văn bản và chọn số câu để xem NMF chọn những câu nào.</p></div>}
      {loading && <div className="empty-state" role="status"><span className="loading-spinner" /><strong>Đang xác định các chủ đề...</strong></div>}
      {result && (
        <div className="result-stack">
          <div className="result-heading">
            <div><div className="section-kicker">KẾT QUẢ TÓM TẮT</div><h3>{result.selected_indices.length} câu được giữ lại</h3></div>
            <div className="result-stats">
              <span>{result.sentence_analysis.length} câu nguồn</span>
              <span>k = {result.k}</span>
              <span>{methodLabel(result)}</span>
              <span>{result.solver?.toUpperCase()} solver</span>
            </div>
          </div>
          {result.fallback_reason && <div className="alert alert-warning" role="status">Dùng Lead-N: {result.fallback_reason}</div>}
          {result.k_candidates?.length > 0 && <div className="panel k-panel"><div className="panel-top"><span className="panel-label">ỨNG VIÊN k</span><span className="panel-meta">{result.k_selection_method}</span></div><div className="k-candidates">{result.k_candidates.map((candidate) => <span className={candidate.k === result.k ? 'is-selected' : ''} key={candidate.k}>k={candidate.k}<strong>{Number(candidate.score).toFixed(5)}</strong></span>)}</div></div>}
          <div className="panel summary-card">
            <div className="panel-top"><span className="panel-label">BẢN TÓM TẮT</span><button type="button" className="text-button" onClick={copySummary}>{copied ? 'Đã sao chép' : 'Sao chép'}</button></div>
            <blockquote>{result.summary}</blockquote>
            <div className="summary-caption">Các câu được giữ nguyên từ văn bản nguồn, theo thứ tự xuất hiện.</div>
          </div>
          {result.topics.length > 0 && <div className="topic-section"><div className="subheading"><span className="panel-label">TỪ KHÓA THEO CHỦ ĐỀ</span><small>Top 5 từ từ ma trận H cục bộ</small></div><div className="topic-grid">{result.topics.map((topic, index) => <div className="panel topic-card" key={index}><div className="topic-label"><span>CHỦ ĐỀ</span><strong>{String(index + 1).padStart(2, '0')}</strong></div><div className="term-list">{topic.top_terms.map((term) => <span key={term}>{term.replaceAll('_', ' ')}</span>)}</div></div>)}</div></div>}
          <div className="sentence-section">
            <div className="subheading"><span className="panel-label">PHÂN TÍCH TỪNG CÂU</span><small>Điểm lấy tại lúc xét/chọn trong greedy selection</small></div>
            <div className="sentence-list">{result.sentence_analysis.map((item) => <article className={`panel sentence-card ${item.selected ? 'selected' : ''}`} key={item.index}><div className="sentence-card-head"><span className="sentence-no">CÂU {String(item.index + 1).padStart(2, '0')}</span><span className={`sentence-status ${item.selected ? 'is-selected' : ''}`}>{item.selected ? 'ĐƯỢC CHỌN' : 'BỎ QUA'}</span></div><p>{item.text}</p><div className="score-list"><span>Chủ đề <strong>{item.dominant_topic === null ? '-' : item.dominant_topic + 1}</strong></span><span>Salience <strong>{number(item.relevance)}</strong></span><span>Coverage <strong>{number(item.coverage_gain)}</strong></span><span>Redundancy <strong>{number(item.redundancy)}</strong></span><span>Điểm <strong>{number(item.score)}</strong></span></div></article>)}</div>
          </div>
        </div>
      )}
    </section>
  );
}
