import { useState } from 'react';
import toy from '../../../data/demo/toy.json';
import { summarize } from '../api.js';

function number(value) {
  return Number(value).toFixed(3);
}

export default function Summarizer() {
  const [text, setText] = useState(toy.text);
  const [k, setK] = useState(2);
  const [sentenceCount, setSentenceCount] = useState(3);
  const [alpha, setAlpha] = useState(1);
  const [beta, setBeta] = useState(1);
  const [gamma, setGamma] = useState(0.5);
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
      const data = await summarize({
        text,
        k: Number(k),
        summary_sentences: Number(sentenceCount),
        alpha: Number(alpha),
        beta: Number(beta),
        gamma: Number(gamma),
      });
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
      <div className="section-heading"><div><div className="section-kicker">02 / TÓM TẮT TRÍCH XUẤT</div><h2 id="summarizer-heading">Giữ lại điều quan trọng</h2><p>Chọn các câu bao phủ nhiều chủ đề và hạn chế lặp nội dung.</p></div><span className="section-index">02</span></div>
      <div className="workspace-grid">
        <form className="panel form-panel" onSubmit={handleSubmit}>
          <div className="panel-top"><span className="panel-label">VĂN BẢN NGUỒN</span><span className="panel-meta">TỐI ĐA 20.000 KÝ TỰ</span></div>
          <label className="field-label" htmlFor="summarize-text">Nội dung cần tóm tắt</label>
          <textarea id="summarize-text" value={text} onChange={(event) => updateField(setText, event.target.value)} maxLength={20000} rows={12} placeholder="Dán văn bản tiếng Việt vào đây…" />
          <div className="field-hint"><span>{text.length.toLocaleString('vi-VN')} / 20.000 ký tự</span><button type="button" className="text-button" onClick={() => { setText(toy.text); setResult(null); setError(''); }}>Nạp ví dụ</button></div>
          <div className="form-row two-cols">
            <label className="field-group">Số chủ đề k<input type="number" min="1" max="10" value={k} onChange={(event) => updateField(setK, event.target.value)} required /></label>
            <label className="field-group">Số câu tóm tắt<input type="number" min="1" max="5" value={sentenceCount} onChange={(event) => updateField(setSentenceCount, event.target.value)} required /></label>
          </div>
          <details className="advanced-settings"><summary>Tinh chỉnh hệ số chấm điểm <span>↘</span></summary><div className="form-row three-cols"><label className="field-group">Liên quan α<input type="number" min="0" step="0.1" value={alpha} onChange={(event) => updateField(setAlpha, event.target.value)} required /></label><label className="field-group">Bao phủ β<input type="number" min="0" step="0.1" value={beta} onChange={(event) => updateField(setBeta, event.target.value)} required /></label><label className="field-group">Giảm lặp γ<input type="number" min="0" step="0.1" value={gamma} onChange={(event) => updateField(setGamma, event.target.value)} required /></label></div></details>
          {error && <div className="alert alert-error" role="alert">{error}</div>}
          <button className="primary-button" type="submit" disabled={loading}>{loading ? 'Đang phân tích…' : 'Tạo bản tóm tắt'}<span aria-hidden="true">↗</span></button>
        </form>

        <div className="panel insight-panel summarize-insight">
          <div className="panel-top"><span className="panel-label">QUY TRÌNH</span><span className="panel-meta">01 → 04</span></div>
          <div className="process-list">
            <div><span>01</span><p><strong>Tách câu gốc</strong><small>Giữ nguyên câu để trích xuất.</small></p></div>
            <div><span>02</span><p><strong>TF–IDF & NMF</strong><small>Khám phá các nhóm từ nổi bật.</small></p></div>
            <div><span>03</span><p><strong>Chấm điểm</strong><small>Cân bằng liên quan, bao phủ và độ trùng.</small></p></div>
            <div><span>04</span><p><strong>Sắp lại thứ tự</strong><small>Đưa câu đã chọn về đúng vị trí gốc.</small></p></div>
          </div>
          <div className="insight-foot">KHÔNG VIẾT LẠI · KHÔNG DÙNG LLM SINH CÂU</div>
        </div>
      </div>

      {!result && !loading && <div className="empty-state"><span className="empty-symbol">≋</span><strong>Bản tóm tắt sẽ xuất hiện ở đây</strong><p>Nhập văn bản và chọn số câu để xem NMF chọn những câu nào.</p></div>}
      {loading && <div className="empty-state" role="status"><span className="loading-spinner" /><strong>Đang xác định các chủ đề…</strong></div>}
      {result && (
        <div className="result-stack">
          <div className="result-heading"><div><div className="section-kicker">KẾT QUẢ TÓM TẮT</div><h3>{result.selected_indices.length} câu được giữ lại</h3></div><div className="result-stats"><span>{result.sentence_analysis.length} câu nguồn</span><span>{result.topics.length} chủ đề</span></div></div>
          {result.fallback_reason && <div className="alert alert-warning" role="status">Dùng Lead-N: {result.fallback_reason}</div>}
          <div className="panel summary-card"><div className="panel-top"><span className="panel-label">BẢN TÓM TẮT</span><button type="button" className="text-button" onClick={copySummary}>{copied ? 'Đã sao chép ✓' : 'Sao chép ↗'}</button></div><blockquote>{result.summary}</blockquote><div className="summary-caption">Các câu được giữ nguyên từ văn bản nguồn, theo thứ tự xuất hiện.</div></div>
          {result.topics.length > 0 && <div className="topic-section"><div className="subheading"><span className="panel-label">TỪ KHÓA THEO CHỦ ĐỀ</span><small>Top 5 từ từ ma trận H</small></div><div className="topic-grid">{result.topics.map((topic, index) => <div className="panel topic-card" key={index}><div className="topic-label"><span>CHỦ ĐỀ</span><strong>{String(index + 1).padStart(2, '0')}</strong></div><div className="term-list">{topic.top_terms.map((term) => <span key={term}>{term.replaceAll('_', ' ')}</span>)}</div></div>)}</div></div>}
          <div className="sentence-section"><div className="subheading"><span className="panel-label">PHÂN TÍCH TỪNG CÂU</span><small>Câu được chọn: điểm tại lượt chọn · câu khác: điểm ở lượt đầu</small></div><div className="sentence-list">{result.sentence_analysis.map((item) => <article className={`panel sentence-card ${item.selected ? 'selected' : ''}`} key={item.index}><div className="sentence-card-head"><span className="sentence-no">CÂU {String(item.index + 1).padStart(2, '0')}</span><span className={`sentence-status ${item.selected ? 'is-selected' : ''}`}>{item.selected ? 'ĐƯỢC CHỌN' : 'BỎ QUA'}</span></div><p>{item.text}</p><div className="score-list"><span>Chủ đề <strong>{item.dominant_topic === null ? '—' : item.dominant_topic + 1}</strong></span><span>Liên quan <strong>{number(item.relevance)}</strong></span><span>Bao phủ <strong>{number(item.coverage_gain)}</strong></span><span>Trùng lặp <strong>{number(item.redundancy)}</strong></span><span>Điểm <strong>{number(item.score)}</strong></span></div></article>)}</div></div>
        </div>
      )}
    </section>
  );
}
