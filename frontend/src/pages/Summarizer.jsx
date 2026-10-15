import { useState } from 'react';
import toy from '../../../data/demo/toy.json';
import { summarize } from '../api.js';

const METHOD_NOTES = {
  local_kl_mmr: 'Fit TF-IDF và KL-NMF cục bộ; dùng cùng cấu hình regularization trên W/H và MMR để chọn câu.',
  local_frobenius_mmr: 'Fit TF-IDF và Frobenius-NMF cục bộ; dùng cùng regularization, k và MMR để đối chiếu riêng hàm mất mát.',
};

function number(value, digits = 3) {
  return Number(value ?? 0).toFixed(digits);
}

function candidateScore(value) {
  return value == null ? 'heuristic' : Number(value).toFixed(5);
}

function penaltyLabel(ratio) {
  const value = Number(ratio);
  if (value === 1) return 'L1';
  if (value === 0) return 'L2';
  return 'L1/L2';
}

function methodLabel(method, ratio) {
  const family = method === 'local_frobenius_mmr' ? 'Frobenius-NMF' : 'KL-NMF';
  return `Local ${family} + ${penaltyLabel(ratio)} + MMR`;
}

function kMethodLabel(method) {
  return ({
    positive_entry_masked_kl_cv: 'Positive-entry masked KL CV',
    heuristic_low_observations: 'Heuristic: ít quan sát dương',
    single_feasible_k: 'Chỉ một k khả thi',
    requested: 'k do người dùng nhập',
    fallback: 'Fallback',
  })[method] || method;
}

export default function Summarizer() {
  const [text, setText] = useState(toy.text);
  const [k, setK] = useState('');
  const [sentenceCount, setSentenceCount] = useState(toy.ground_truth_indices.length);
  const [method, setMethod] = useState('local_kl_mmr');
  const [mmrLambda, setMmrLambda] = useState(0.7);
  const [alphaW, setAlphaW] = useState(0.001);
  const [alphaH, setAlphaH] = useState(0.001);
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
        alpha_w: Number(alphaW),
        alpha_h: Number(alphaH),
        l1_ratio: Number(l1Ratio),
      };
      if (k !== '') payload.k = Number(k);
      if (query.trim()) {
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
          <div className="panel-top"><span className="panel-label">VĂN BẢN NGUỒN</span><span className="panel-meta">LOCAL NMF / TWO OBJECTIVES</span></div>
          <label className="field-label" htmlFor="summarize-text">Nội dung cần tóm tắt</label>
          <textarea id="summarize-text" value={text} onChange={(event) => updateField(setText, event.target.value)} rows={12} placeholder="Dán văn bản tiếng Việt vào đây..." />
          <div className="field-hint"><span>{text.length.toLocaleString('vi-VN')} ký tự</span><button type="button" className="text-button" onClick={() => { setText(toy.text); setSentenceCount(toy.ground_truth_indices.length); setResult(null); setCopied(false); setError(''); }}>Nạp ví dụ</button></div>

          <label className="field-group field-block" htmlFor="summary-method">Phương pháp
            <select id="summary-method" value={method} onChange={(event) => updateField(setMethod, event.target.value)}>
              <option value="local_kl_mmr">Local KL-NMF + shared penalty + MMR</option>
              <option value="local_frobenius_mmr">Local Frobenius-NMF + shared penalty + MMR</option>
            </select>
          </label>
          <p className="method-description">{METHOD_NOTES[method]}</p>

          <div className="form-row two-cols">
            <label className="field-group">Số chủ đề k<input type="number" min="1" max="10" value={k} onChange={(event) => updateField(setK, event.target.value)} placeholder="Tự chọn" /></label>
            <label className="field-group">Số câu tóm tắt<input type="number" min="1" value={sentenceCount} onChange={(event) => updateField(setSentenceCount, event.target.value)} required /></label>
          </div>

          <div className="query-box">
            <label className="field-label" htmlFor="summary-query">Truy vấn (không bắt buộc)</label>
            <input id="summary-query" type="text" value={query} maxLength={2000} onChange={(event) => updateField(setQuery, event.target.value)} placeholder="Ví dụ: tác động của chính sách..." />
            {query.trim() && <label className="field-group query-weight">Trọng số truy vấn<input type="number" min="0" max="1" step="0.05" value={queryWeight} onChange={(event) => updateField(setQueryWeight, event.target.value)} /></label>}
          </div>

          <details className="advanced-settings">
            <summary>Tham số thuật toán <span>+</span></summary>
            <div className="form-row two-cols advanced-grid">
              <label className="field-group">MMR λ<input type="number" min="0" max="1" step="0.05" value={mmrLambda} onChange={(event) => updateField(setMmrLambda, event.target.value)} /></label>
              <label className="field-group">α W (cả hai loss)<input type="number" min="0" max="1" step="0.0005" value={alphaW} onChange={(event) => updateField(setAlphaW, event.target.value)} /></label>
              <label className="field-group">α H (cả hai loss)<input type="number" min="0" max="1" step="0.0005" value={alphaH} onChange={(event) => updateField(setAlphaH, event.target.value)} /></label>
              <label className="field-group">L1 ratio (cả hai)<input type="number" min="0" max="1" step="0.05" value={l1Ratio} onChange={(event) => updateField(setL1Ratio, event.target.value)} /></label>
            </div>
            <p className="field-hint">L1 ratio 1 = L1; 0 = L2; giá trị giữa trộn hai penalty. Alpha được scale theo kích thước ma trận; nếu penalty làm factor về 0, hệ thống báo fallback.</p>
          </details>
          {error && <div className="alert alert-error" role="alert">{error}</div>}
          <button className="primary-button" type="submit" disabled={loading}>{loading ? 'Đang phân tích...' : 'Tạo bản tóm tắt'}<span aria-hidden="true">→</span></button>
        </form>

        <div className="panel insight-panel summarize-insight">
          <div className="panel-top"><span className="panel-label">PHƯƠNG PHÁP ĐANG CHỌN</span><span className="panel-meta">{methodLabel(method, l1Ratio)}</span></div>
          <p className="method-description method-description-panel">{METHOD_NOTES[method]}</p>
          <div className="process-list">
            <div><span>01</span><p><strong>TF-IDF cục bộ</strong><small>Mỗi câu là một document; vocabulary fit từ văn bản hiện tại.</small></p></div>
            <div><span>02</span><p><strong>Chọn k</strong><small>Masked KL CV trên các phần tử TF-IDF dương; heuristic được đánh dấu khi dữ liệu quá ít.</small></p></div>
            <div><span>03</span><p><strong>Đối chiếu objective</strong><small>Hai loss dùng cùng alpha/L1; khác biệt chính là KL divergence so với Frobenius.</small></p></div>
            <div><span>04</span><p><strong>Trích xuất câu bằng MMR</strong><small>Cả hai cách dùng cùng ngân sách câu, MMR và giữ thứ tự gốc.</small></p></div>
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
              <span>{methodLabel(result.method, result.l1_ratio)}</span>
              <span>MMR λ = {number(result.mmr_lambda, 2)}</span>
              <span>{result.solver}</span>
            </div>
          </div>
          {result.fallback_reason && <div className="alert alert-warning" role="status">Dùng Lead-N: {result.fallback_reason}</div>}
          {result.query_feedback_reason && query.trim() && <div className="alert alert-warning" role="status">Query PRF chưa áp dụng: {result.query_feedback_reason}.</div>}
          {result.loss_value !== null && result.loss_value !== undefined && <div className="panel loss-summary">
            <div><span className="panel-label">OBJECTIVE / {result.loss_name}</span><strong>{number(result.loss_value, 6)}</strong></div>
            {result.data_loss !== null && result.data_loss !== undefined && <div><span>Thành phần tái tạo</span><strong>{number(result.data_loss, 6)}</strong></div>}
            {result.regularization_loss !== null && result.regularization_loss !== undefined && <div><span>L1 penalty</span><strong>{number(result.regularization_loss, 6)}</strong></div>}
          </div>}
          {result.k_candidates?.length > 0 && <div className="panel k-panel"><div className="panel-top"><span className="panel-label">ỨNG VIÊN k</span><span className="panel-meta">{kMethodLabel(result.k_selection_method)}</span></div><div className="k-candidates">{result.k_candidates.map((candidate) => <span className={candidate.k === result.k ? 'is-selected' : ''} key={candidate.k}>k={candidate.k}<strong>{candidateScore(candidate.score)}</strong></span>)}</div><p className="field-hint k-note">Điểm thấp hơn tốt hơn theo positive-entry masked KL CV (k = 1…10). Heuristic không có điểm CV. Đây không phải ROUGE hay điểm chất lượng tóm tắt và không tối ưu riêng cho từng loss.</p></div>}
          <div className="panel summary-card">
            <div className="panel-top"><span className="panel-label">BẢN TÓM TẮT</span><button type="button" className="text-button" onClick={copySummary}>{copied ? 'Đã sao chép' : 'Sao chép'}</button></div>
            <blockquote>{result.summary}</blockquote>
            <div className="summary-caption">Các câu được giữ nguyên từ văn bản nguồn, theo thứ tự xuất hiện.{result.query_feedback_applied && ` Câu pseudo-relevant: ${result.pseudo_relevant_indices.map((index) => index + 1).join(', ')}.`}</div>
          </div>
          {result.topics.length > 0 && <div className="topic-section"><div className="subheading"><span className="panel-label">TỪ KHÓA ĐẶC TRƯNG THEO CHỦ ĐỀ</span><small>Xếp hạng từ H đã chuẩn hóa và độ độc quyền giữa các chủ đề</small></div><div className="topic-grid">{result.topics.map((topic, index) => <div className="panel topic-card" key={index}><div className="topic-label"><span>CHỦ ĐỀ</span><strong>{String(index + 1).padStart(2, '0')}</strong></div><div className="term-list">{topic.top_terms.map((term) => <span key={term}>{term.replaceAll('_', ' ')}</span>)}</div><div className="topic-examples"><span className="panel-label">CÂU ĐẠI DIỆN</span>{topic.representative_sentences?.map((sentence, sentenceIndex) => <p key={`${index}-${sentenceIndex}`}>{sentence}</p>)}</div></div>)}</div></div>}
          <div className="sentence-section">
            <div className="subheading"><span className="panel-label">PHÂN TÍCH TỪNG CÂU</span><small>{result.selection_method}</small></div>
            <div className="sentence-list">{result.sentence_analysis.map((item) => <article className={`panel sentence-card ${item.selected ? 'selected' : ''}`} key={item.index}><div className="sentence-card-head"><span className="sentence-no">CÂU {String(item.index + 1).padStart(2, '0')}</span><span className={`sentence-status ${item.selected ? 'is-selected' : ''}`}>{item.selected ? 'ĐƯỢC CHỌN' : 'BỎ QUA'}</span></div><p>{item.text}</p><div className="score-list"><span>Chủ đề <strong>{item.dominant_topic === null ? '-' : item.dominant_topic + 1}</strong></span><span>Relevance <strong>{number(item.relevance)}</strong></span><span>Redundancy <strong>{number(item.redundancy)}</strong></span><span>Điểm chọn (vòng {item.score_round ?? '-'}) <strong>{number(item.score)}</strong></span>{result.query_feedback_applied && <span>Query <strong>{number(item.query_relevance)}</strong></span>}</div></article>)}</div>
          </div>
          {result.selection_trace?.length > 0 && <details className="panel selection-trace"><summary>Xem so sánh điểm MMR theo từng vòng</summary><p className="field-hint">Chỉ so sánh các ứng viên trong cùng một vòng: redundancy thay đổi sau mỗi câu được chọn.</p>{result.selection_trace.map((round) => <div className="trace-round" key={round.round}><h4>Vòng {round.round}: chọn câu {round.selected_index + 1}</h4><div className="table-scroll"><table className="data-table"><thead><tr><th>Câu</th><th>Relevance</th><th>Redundancy</th><th>MMR</th></tr></thead><tbody>{round.candidates.map((candidate) => <tr key={candidate.index} className={candidate.index === round.selected_index ? 'ablation-baseline-row' : ''}><th scope="row">Câu {candidate.index + 1}</th><td>{number(candidate.relevance)}</td><td>{number(candidate.redundancy)}</td><td>{number(candidate.score)}</td></tr>)}</tbody></table></div></div>)}</details>}
        </div>
      )}
    </section>
  );
}
