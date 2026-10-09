import { useState } from 'react';
import toy from '../../../data/demo/toy.json';
import { compareGlobalLocal } from '../api.js';

function number(value, digits = 3) {
  return Number(value ?? 0).toFixed(digits);
}

function percent(value) {
  return `${Math.round(Number(value ?? 0) * 100)}%`;
}

function MethodPanel({ data, sentences }) {
  const meta = data.metadata || {};
  const selected = data.selected_indices || [];
  const isLocal = data.method_id === 'local_kl_nmf_mmr';
  const lossLabel = isLocal ? `${meta.solver?.toUpperCase?.() || 'MU'} / KL + L1` : `${String(meta.solver || 'CD').toUpperCase()} / ${meta.loss_name || 'frobenius'}`;
  const vocabSize = meta.vocabulary_size ?? 0;

  return (
    <article className={`panel compare-method-card ${isLocal ? 'is-local' : 'is-global'}`}>
      <div className="panel-top">
        <span className="panel-label">{data.label}</span>
        <span className="panel-meta">{data.scope}</span>
      </div>
      <blockquote>{data.summary || 'Không tạo được bản tóm tắt.'}</blockquote>
      <div className="compare-meta-grid">
        <span>Loss/Solver <strong>{lossLabel}</strong></span>
        <span>Vocabulary <strong>{Number(vocabSize).toLocaleString('vi-VN')}</strong></span>
        <span>{isLocal ? 'k đã dùng' : 'Số topic'} <strong>{isLocal ? meta.k : meta.n_components}</strong></span>
        <span>{isLocal ? 'MMR λ' : 'OOV'} <strong>{isLocal ? number(meta.mmr_lambda, 2) : percent(meta.oov_rate)}</strong></span>
      </div>
      {meta.oov_examples?.length > 0 && (
        <div className="compare-note">Từ ngoài vocabulary global: {meta.oov_examples.slice(0, 8).join(', ')}</div>
      )}
      <div className="subheading">
        <span className="panel-label">CÂU ĐƯỢC CHỌN</span>
        <small>{selected.map((index) => index + 1).join(', ') || 'Không có'}</small>
      </div>
      <ol className="compare-selection-list">
        {selected.map((index) => <li key={index}>{sentences[index]}</li>)}
      </ol>
      {data.topics?.length > 0 && (
        <>
          <div className="subheading">
            <span className="panel-label">TỪ KHÓA TOPIC</span>
            <small>{data.topics.length} topic đại diện</small>
          </div>
          <div className="topic-grid compact-topic-grid">
            {data.topics.map((topic) => (
              <div className="topic-mini" key={topic.topic_index}>
                <strong>Topic {Number(topic.topic_index) + 1}</strong>
                <div className="term-list">{topic.top_terms.map((term) => <span key={term}>{term.replaceAll('_', ' ')}</span>)}</div>
              </div>
            ))}
          </div>
        </>
      )}
    </article>
  );
}

export default function Comparison() {
  const [text, setText] = useState(toy.text);
  const [k, setK] = useState('');
  const [sentenceCount, setSentenceCount] = useState(3);
  const [mmrLambda, setMmrLambda] = useState(0.7);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  function updateField(setter, value) {
    setter(value);
    setResult(null);
    setError('');
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setError('');
    setResult(null);
    if (!text.trim()) {
      setError('Hãy nhập văn bản để so sánh.');
      return;
    }
    setLoading(true);
    try {
      const payload = {
        text,
        summary_sentences: Number(sentenceCount),
        mmr_lambda: Number(mmrLambda),
      };
      if (k !== '') payload.k = Number(k);
      setResult(await compareGlobalLocal(payload));
    } catch (cause) {
      setError(cause.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section aria-labelledby="comparison-heading">
      <div className="section-heading">
        <div>
          <div className="section-kicker">04 / GLOBAL VS LOCAL</div>
          <h2 id="comparison-heading">So sánh hai luồng NMF</h2>
          <p>Đặt mô hình global train sẵn trên 3260 bài báo cạnh Local KL-NMF mới để thấy khác biệt về vocabulary, loss, topic và câu được chọn.</p>
        </div>
        <span className="section-index">04</span>
      </div>

      <div className="workspace-grid">
        <form className="panel form-panel" onSubmit={handleSubmit}>
          <div className="panel-top"><span className="panel-label">VĂN BẢN KIỂM TRA</span><span className="panel-meta">SO SÁNH TRỰC TIẾP</span></div>
          <label className="field-label" htmlFor="compare-text">Nội dung cần so sánh</label>
          <textarea id="compare-text" rows={12} value={text} onChange={(event) => updateField(setText, event.target.value)} placeholder="Dán văn bản tiếng Việt vào đây..." />
          <div className="field-hint"><span>{text.length.toLocaleString('vi-VN')} ký tự</span><button type="button" className="text-button" onClick={() => { setText(toy.text); setResult(null); setError(''); }}>Nạp ví dụ</button></div>
          <div className="form-row three-cols">
            <label className="field-group">Số câu tóm tắt<input type="number" min="1" value={sentenceCount} onChange={(event) => updateField(setSentenceCount, event.target.value)} required /></label>
            <label className="field-group">k local<input type="number" min="1" max="10" value={k} onChange={(event) => updateField(setK, event.target.value)} placeholder="Tự chọn" /></label>
            <label className="field-group">MMR λ<input type="number" min="0" max="1" step="0.05" value={mmrLambda} onChange={(event) => updateField(setMmrLambda, event.target.value)} required /></label>
          </div>
          {error && <div className="alert alert-error" role="alert">{error}</div>}
          <button className="primary-button" type="submit" disabled={loading}>{loading ? 'Đang so sánh...' : 'Chạy so sánh'}<span aria-hidden="true">→</span></button>
        </form>

        <aside className="panel insight-panel">
          <div className="panel-top"><span className="panel-label">CÁCH ĐỌC</span><span className="panel-meta">OLD / NEW</span></div>
          <p><strong>Global NMF cũ</strong> dùng vocabulary và ma trận H đã học từ 3260 bài báo. Khi văn bản ngắn khác miền dữ liệu, nhiều từ có thể ngoài vocabulary.</p>
          <div className="insight-rule" />
          <p><strong>Local KL-NMF mới</strong> fit lại TF-IDF và NMF trên chính các câu đầu vào, dùng KL + L1 và MMR để giảm trùng lặp câu.</p>
          <div className="insight-foot">GLOBAL FROBENIUS/CD · LOCAL KL/MU/L1/MMR</div>
        </aside>
      </div>

      {!result && !loading && <div className="empty-state"><span className="empty-symbol">⇄</span><strong>Chưa có kết quả so sánh</strong><p>Chạy cùng một văn bản để xem Global NMF cũ và Local KL-NMF mới chọn câu khác nhau ra sao.</p></div>}
      {loading && <div className="empty-state" role="status"><span className="loading-spinner" /><strong>Đang chạy hai luồng NMF...</strong></div>}
      {result && (
        <div className="result-stack">
          <div className="panel comparison-overview">
            <div>
              <div className="section-kicker">TÓM TẮT KHÁC BIỆT</div>
              <h3>{result.differences.summary_changed ? 'Hai luồng cho kết quả khác nhau' : 'Hai luồng chọn cùng nội dung chính'}</h3>
              <p>{result.differences.vocabulary_change}</p>
            </div>
            <div className="comparison-score">
              <span>{percent(result.differences.selection_jaccard)}</span>
              <small>trùng câu được chọn</small>
            </div>
          </div>

          <div className="compare-diff-grid">
            <div className="panel compare-diff-card">
              <span className="panel-label">CHỈ LOCAL CHỌN</span>
              <strong>{result.differences.local_only.map((index) => index + 1).join(', ') || 'Không có'}</strong>
              <p>{result.differences.selection_change}</p>
            </div>
            <div className="panel compare-diff-card">
              <span className="panel-label">CHỈ GLOBAL CHỌN</span>
              <strong>{result.differences.global_only.map((index) => index + 1).join(', ') || 'Không có'}</strong>
              <p>{result.differences.objective_change}</p>
            </div>
          </div>

          <div className="compare-method-grid">
            <MethodPanel data={result.global_nmf} sentences={result.sentences} />
            <MethodPanel data={result.local} sentences={result.sentences} />
          </div>
        </div>
      )}
    </section>
  );
}
