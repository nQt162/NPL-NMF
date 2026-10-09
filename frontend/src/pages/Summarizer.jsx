import { useState } from 'react';
import toy from '../../../data/demo/toy.json';
import { summarize } from '../api.js';

const METHOD_LABELS = {
  local_kl_mmr: 'Local KL-NMF + L1 + MMR',
  nmfts_pairwise: 'NMFTS-inspired pairwise KL',
  snmf: 'SNMF đồ thị câu',
};

const METHOD_NOTES = {
  local_kl_mmr: 'Fit TF-IDF và KL-NMF trực tiếp trên văn bản; L1 làm topic thưa, MMR giảm câu trùng lặp.',
  nmfts_pairwise: 'Frobenius NMF cộng phạt symmetric-KL giữa câu tương tự; chọn câu bằng MMR sau phân rã.',
  snmf: 'Phân rã đối xứng đồ thị cosine câu, gom cụm và lần lượt chọn câu đại diện.',
};

function number(value, digits = 3) {
  return Number(value ?? 0).toFixed(digits);
}

export default function Summarizer() {
  const [text, setText] = useState(toy.text);
  const [k, setK] = useState('');
  const [sentenceCount, setSentenceCount] = useState(3);
  const [method, setMethod] = useState('local_kl_mmr');
  const [mmrLambda, setMmrLambda] = useState(0.7);
  const [pairwiseLambda, setPairwiseLambda] = useState(0.1);
  const [alphaW, setAlphaW] = useState(0.1);
  const [alphaH, setAlphaH] = useState(0.1);
  const [l1Ratio, setL1Ratio] = useState(1);
  const [query, setQuery] = useState('');
  const [queryWeight, setQueryWeight] = useState(0.5);
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
    if (!text.trim()) {
      setError('Hãy nhập văn bản cần tóm tắt.');
      return;
    }
    setLoading(true);
    try {
      const payload = {
        text,
        summary_sentences: Number(sentenceCount),
        method,
        mmr_lambda: Number(mmrLambda),
        pairwise_lambda: Number(pairwiseLambda),
        alpha_w: Number(alphaW),
        alpha_h: Number(alphaH),
        l1_ratio: Number(l1Ratio),
      };
      if (k !== '') payload.k = Number(k);
      if (query.trim() && method === 'local_kl_mmr') {
        payload.query = query.trim();
        payload.query_weight = Number(queryWeight);
      }
      setResult(await summarize(payload));
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

  const localMethod = method === 'local_kl_mmr';
  const usesMmr = method !== 'snmf';

  return (
    <section aria-labelledby="summarizer-heading">
      <div className="section-heading">
        <div>
          <div className="section-kicker">02 / TÓM TẮT TRÍCH XUẤT</div>
          <h2 id="summarizer-heading">Giữ lại điều quan trọng</h2>
          <p>Chọn biến thể NMF, xem objective tương ứng và giữ nguyên câu trích theo thứ tự gốc.</p>
        </div>
        <span className="section-index">02</span>
      </div>

      <div className="workspace-grid">
        <form className="panel form-panel" onSubmit={handleSubmit}>
          <div className="panel-top"><span className="panel-label">VĂN BẢN NGUỒN</span><span className="panel-meta">LOCAL / PAPER VARIANTS</span></div>
          <label className="field-label" htmlFor="summarize-text">Nội dung cần tóm tắt</label>
          <textarea id="summarize-text" value={text} onChange={(event) => updateField(setText, event.target.value)} rows={12} placeholder="Dán văn bản tiếng Việt vào đây..." />
          <div className="field-hint"><span>{text.length.toLocaleString('vi-VN')} ký tự</span><button type="button" className="text-button" onClick={() => { setText(toy.text); setResult(null); setError(''); }}>Nạp ví dụ</button></div>

          <label className="field-group field-block" htmlFor="summary-method">Phương pháp
            <select id="summary-method" value={method} onChange={(event) => updateField(setMethod, event.target.value)}>
              <option value="local_kl_mmr">Local KL-NMF + L1 + MMR</option>
              <option value="nmfts_pairwise">NMFTS-inspired pairwise KL</option>
              <option value="snmf">SNMF đồ thị câu</option>
            </select>
          </label>
          <p className="method-description">{METHOD_NOTES[method]}</p>

          <div className="form-row two-cols">
            <label className="field-group">Số chủ đề k<input type="number" min="1" max="10" value={k} onChange={(event) => updateField(setK, event.target.value)} placeholder="Tự chọn" /></label>
            <label className="field-group">Số câu tóm tắt<input type="number" min="1" value={sentenceCount} onChange={(event) => updateField(setSentenceCount, event.target.value)} required /></label>
          </div>

          {localMethod && <div className="query-box">
            <label className="field-label" htmlFor="summary-query">Truy vấn (không bắt buộc)</label>
            <input id="summary-query" type="text" value={query} maxLength={2000} onChange={(event) => updateField(setQuery, event.target.value)} placeholder="Ví dụ: tác động của chính sách..." />
            {query.trim() && <label className="field-group query-weight">Trọng số truy vấn<input type="number" min="0" max="1" step="0.05" value={queryWeight} onChange={(event) => updateField(setQueryWeight, event.target.value)} /></label>}
          </div>}

          <details className="advanced-settings">
            <summary>Tham số thuật toán <span>+</span></summary>
            <div className="form-row two-cols advanced-grid">
              {usesMmr && <label className="field-group">MMR λ<input type="number" min="0" max="1" step="0.05" value={mmrLambda} onChange={(event) => updateField(setMmrLambda, event.target.value)} /></label>}
              {method === 'nmfts_pairwise' && <label className="field-group">Pairwise λ<input type="number" min="0" max="10" step="0.05" value={pairwiseLambda} onChange={(event) => updateField(setPairwiseLambda, event.target.value)} /></label>}
              {localMethod && <>
                <label className="field-group">α W<input type="number" min="0" max="1" step="0.01" value={alphaW} onChange={(event) => updateField(setAlphaW, event.target.value)} /></label>
                <label className="field-group">α H<input type="number" min="0" max="1" step="0.01" value={alphaH} onChange={(event) => updateField(setAlphaH, event.target.value)} /></label>
                <label className="field-group">L1 ratio<input type="number" min="0" max="1" step="0.05" value={l1Ratio} onChange={(event) => updateField(setL1Ratio, event.target.value)} /></label>
              </>}
            </div>
          </details>
          {error && <div className="alert alert-error" role="alert">{error}</div>}
          <button className="primary-button" type="submit" disabled={loading}>{loading ? 'Đang phân tích...' : 'Tạo bản tóm tắt'}<span aria-hidden="true">→</span></button>
        </form>

        <div className="panel insight-panel summarize-insight">
          <div className="panel-top"><span className="panel-label">PHƯƠNG PHÁP ĐANG CHỌN</span><span className="panel-meta">{METHOD_LABELS[method]}</span></div>
          <p className="method-description method-description-panel">{METHOD_NOTES[method]}</p>
          <div className="process-list">
            <div><span>01</span><p><strong>TF-IDF cục bộ</strong><small>Mỗi câu là một document; vocabulary fit từ văn bản hiện tại.</small></p></div>
            <div><span>02</span><p><strong>Chọn k</strong><small>Tự chọn bằng masked KL imputation; có thể nhập k cố định.</small></p></div>
            <div><span>03</span><p><strong>Phân rã riêng objective</strong><small>KL+L1, NMFTS pairwise hoặc SNMF graph không dùng chung loss.</small></p></div>
            <div><span>04</span><p><strong>Trích xuất câu</strong><small>MMR giảm lặp; SNMF chọn đại diện theo cụm. Câu giữ thứ tự gốc.</small></p></div>
          </div>
          <div className="insight-foot">TRÍCH XUẤT CÂU GỐC · KHÔNG SINH VĂN BẢN</div>
        </div>
      </div>

      {!result && !loading && <div className="empty-state"><span className="empty-symbol">↗</span><strong>Bản tóm tắt sẽ xuất hiện ở đây</strong><p>Chọn một phương pháp để xem câu, chủ đề và loss tương ứng.</p></div>}
      {loading && <div className="empty-state" role="status"><span className="loading-spinner" /><strong>Đang phân tích văn bản...</strong></div>}
      {result && (
        <div className="result-stack">
          <div className="result-heading">
            <div><div className="section-kicker">KẾT QUẢ TÓM TẮT</div><h3>{result.selected_indices.length} câu được giữ lại</h3></div>
            <div className="result-stats">
              <span>{result.sentence_analysis.length} câu nguồn</span>
              <span>k = {result.k}</span>
              <span>{METHOD_LABELS[result.method] || result.method}</span>
              {usesMmr && <span>MMR λ = {number(result.mmr_lambda, 2)}</span>}
              <span>{result.solver}</span>
            </div>
          </div>
          {result.fallback_reason && <div className="alert alert-warning" role="status">Dùng Lead-N: {result.fallback_reason}</div>}
          {result.query_feedback_reason && query.trim() && <div className="alert alert-warning" role="status">Query PRF chưa áp dụng: {result.query_feedback_reason}.</div>}
          {result.loss_value !== null && result.loss_value !== undefined && <div className="panel loss-summary">
            <div><span className="panel-label">OBJECTIVE / {result.loss_name}</span><strong>{number(result.loss_value, 6)}</strong></div>
            {result.data_loss !== null && result.data_loss !== undefined && <div><span>Thành phần tái tạo</span><strong>{number(result.data_loss, 6)}</strong></div>}
            {result.regularization_loss !== null && result.regularization_loss !== undefined && <div><span>{result.method === 'nmfts_pairwise' ? 'Pairwise penalty' : 'L1 penalty'}</span><strong>{number(result.regularization_loss, 6)}</strong></div>}
          </div>}
          {result.k_candidates?.length > 0 && <div className="panel k-panel"><div className="panel-top"><span className="panel-label">ỨNG VIÊN k</span><span className="panel-meta">{result.k_selection_method}</span></div><div className="k-candidates">{result.k_candidates.map((candidate) => <span className={candidate.k === result.k ? 'is-selected' : ''} key={candidate.k}>k={candidate.k}<strong>{Number(candidate.score).toFixed(5)}</strong></span>)}</div><p className="field-hint k-note">Khi tự chọn, masked KL imputation được dùng chung để có cùng chính sách k giữa các biến thể; đây không phải tối ưu riêng objective NMFTS/SNMF.</p></div>}
          <div className="panel summary-card">
            <div className="panel-top"><span className="panel-label">BẢN TÓM TẮT</span><button type="button" className="text-button" onClick={copySummary}>{copied ? 'Đã sao chép' : 'Sao chép'}</button></div>
            <blockquote>{result.summary}</blockquote>
            <div className="summary-caption">Các câu được giữ nguyên từ văn bản nguồn, theo thứ tự xuất hiện.{result.query_feedback_applied && ` Câu pseudo-relevant: ${result.pseudo_relevant_indices.map((index) => index + 1).join(', ')}.`}</div>
          </div>
          {result.topics.length > 0 && <div className="topic-section"><div className="subheading"><span className="panel-label">TỪ KHÓA THEO CHỦ ĐỀ</span><small>Đại diện từ ma trận factor hoặc cụm câu</small></div><div className="topic-grid">{result.topics.map((topic, index) => <div className="panel topic-card" key={index}><div className="topic-label"><span>CHỦ ĐỀ</span><strong>{String(index + 1).padStart(2, '0')}</strong></div><div className="term-list">{topic.top_terms.map((term) => <span key={term}>{term.replaceAll('_', ' ')}</span>)}</div></div>)}</div></div>}
          <div className="sentence-section">
            <div className="subheading"><span className="panel-label">PHÂN TÍCH TỪNG CÂU</span><small>{result.selection_method}</small></div>
            <div className="sentence-list">{result.sentence_analysis.map((item) => <article className={`panel sentence-card ${item.selected ? 'selected' : ''}`} key={item.index}><div className="sentence-card-head"><span className="sentence-no">CÂU {String(item.index + 1).padStart(2, '0')}</span><span className={`sentence-status ${item.selected ? 'is-selected' : ''}`}>{item.selected ? 'ĐƯỢC CHỌN' : 'BỎ QUA'}</span></div><p>{item.text}</p><div className="score-list"><span>Chủ đề <strong>{item.dominant_topic === null ? '-' : item.dominant_topic + 1}</strong></span><span>Relevance <strong>{number(item.relevance)}</strong></span><span>Redundancy <strong>{number(item.redundancy)}</strong></span><span>Điểm chọn <strong>{number(item.score)}</strong></span>{result.query_feedback_applied && <span>Query <strong>{number(item.query_relevance)}</strong></span>}</div></article>)}</div>
          </div>
        </div>
      )}
    </section>
  );
}
