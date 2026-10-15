import { useState } from 'react';
import toy from '../../../data/demo/toy.json';
import { evaluate } from '../api.js';

const ROUGE_METRICS = [
  ['rouge1_f1', 'ROUGE-1'],
  ['rouge2_f1', 'ROUGE-2'],
  ['rougeL_f1', 'ROUGE-L'],
];

function formatScore(value) {
  return Number(value).toFixed(4);
}

function penaltyLabel(ratio) {
  const value = Number(ratio);
  if (value === 1) return 'L1';
  if (value === 0) return 'L2';
  return 'L1/L2';
}

export default function Evaluation() {
  const [text, setText] = useState(toy.text);
  const [reference, setReference] = useState(toy.extractive_summary);
  const [sentenceCount, setSentenceCount] = useState(toy.ground_truth_indices.length);
  const [k, setK] = useState('');
  const [mmrLambda, setMmrLambda] = useState(0.7);
  const [alphaW, setAlphaW] = useState(0.001);
  const [alphaH, setAlphaH] = useState(0.001);
  const [l1Ratio, setL1Ratio] = useState(1.0);
  const [seed, setSeed] = useState(42);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  function updateField(setter, value) {
    setter(value);
    setResult(null);
    setError('');
  }

  function loadExample() {
    setText(toy.text);
    setReference(toy.extractive_summary);
    setSentenceCount(toy.ground_truth_indices.length);
    setResult(null);
    setError('');
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setResult(null);
    setError('');
    if (!text.trim() || !reference.trim()) {
      setError('Hãy nhập văn bản đầu vào và bản tóm tắt tham chiếu.');
      return;
    }

    const payload = {
      text,
      reference_summary: reference,
      summary_sentences: Number(sentenceCount),
      mmr_lambda: Number(mmrLambda),
      alpha_w: Number(alphaW),
      alpha_h: Number(alphaH),
      l1_ratio: Number(l1Ratio),
      seed: Number(seed),
    };
    if (k !== '') payload.k = Number(k);

    setLoading(true);
    try {
      setResult(await evaluate(payload));
    } catch (cause) {
      setError(cause.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section aria-labelledby="evaluation-heading">
      <div className="section-heading">
        <div>
          <div className="section-kicker">03 / ĐÁNH GIÁ</div>
          <h2 id="evaluation-heading">So sánh hai objective bằng ROUGE</h2>
          <p>Cùng văn bản, k, alpha/L1, ngân sách câu, seed và MMR; chỉ đổi hàm mất mát.</p>
        </div>
        <span className="section-index">03</span>
      </div>

      <div className="workspace-grid evaluation-demo">
        <form className="panel form-panel" onSubmit={handleSubmit}>
          <div className="panel-top">
            <span className="panel-label">THIẾT LẬP SO SÁNH</span>
            <span className="panel-meta">KL + L1 + MMR / FROBENIUS + L1 + MMR</span>
          </div>
          <label className="field-label" htmlFor="evaluate-text">Văn bản đầu vào</label>
          <textarea
            id="evaluate-text"
            rows={9}
            value={text}
            onChange={(event) => updateField(setText, event.target.value)}
            placeholder="Nhập văn bản cần tóm tắt..."
            required
          />
          <div className="field-hint"><span>{text.length.toLocaleString('vi-VN')} ký tự</span><button type="button" className="text-button" onClick={loadExample}>Nạp ví dụ</button></div>
          <label className="field-label field-label-spaced" htmlFor="evaluate-reference">
            Bản tóm tắt tham chiếu
          </label>
          <textarea
            id="evaluate-reference"
            rows={4}
            value={reference}
            onChange={(event) => updateField(setReference, event.target.value)}
            placeholder="Nhập bản tham chiếu do người đánh giá biên soạn..."
            required
          />

          <div className="form-row two-cols field-label-spaced">
            <label className="field-group">
              Số chủ đề k
              <input
                type="number"
                min="1"
                value={k}
                onChange={(event) => updateField(setK, event.target.value)}
                placeholder="Tự chọn"
              />
            </label>
            <label className="field-group">
              Số câu tóm tắt
              <input
                type="number"
                min="1"
                max="20"
                value={sentenceCount}
                onChange={(event) => updateField(setSentenceCount, event.target.value)}
                required
              />
            </label>
          </div>

          <details className="advanced-settings">
            <summary>Tham số thuật toán <span>+</span></summary>
            <div className="form-row two-cols advanced-grid">
              <label className="field-group">
                MMR λ (cả hai)
                <input type="number" min="0" max="1" step="0.05" value={mmrLambda}
                  onChange={(event) => updateField(setMmrLambda, event.target.value)} />
              </label>
              <label className="field-group">
                Seed (cả hai)
                <input type="number" min="0" value={seed}
                  onChange={(event) => updateField(setSeed, event.target.value)} />
              </label>
              <label className="field-group">
                α W (cả hai loss)
                <input type="number" min="0" max="1" step="0.0005" value={alphaW}
                  onChange={(event) => updateField(setAlphaW, event.target.value)} />
              </label>
              <label className="field-group">
                α H (cả hai loss)
                <input type="number" min="0" max="1" step="0.0005" value={alphaH}
                  onChange={(event) => updateField(setAlphaH, event.target.value)} />
              </label>
              <label className="field-group">
                L1 ratio (cả hai)
                <input type="number" min="0" max="1" step="0.05" value={l1Ratio}
                  onChange={(event) => updateField(setL1Ratio, event.target.value)} />
              </label>
            </div>
            <p className="field-hint">L1 ratio 1 = L1; 0 = L2; giá trị giữa trộn hai penalty. Alpha được nhân theo kích thước ma trận, nên cần chọn trên validation; nếu penalty làm factor về 0, kết quả sẽ báo fallback.</p>
          </details>

          {error && <div className="alert alert-error" role="alert">{error}</div>}
          <button className="primary-button" type="submit" disabled={loading}>
            {loading ? 'Đang chạy hai phương pháp...' : 'Tạo và chấm hai bản tóm tắt'}
            <span aria-hidden="true">→</span>
          </button>
        </form>

        <div className="panel demo-result-panel" aria-live="polite">
          <div className="panel-top">
            <span className="panel-label">ROUGE F1</span>
            <span className="panel-meta">0–1 · k chung</span>
          </div>
          {result ? (
            <>
              <p className="evaluation-k">k = {result.k} · {result.k_selection_method}</p>
              <div className="evaluation-candidate">
                <h3>KL-NMF + {penaltyLabel(l1Ratio)} + MMR</h3>
                <div className="demo-scores">
                  {ROUGE_METRICS.map(([field, label]) => (
                    <div key={field}><span>{label}</span><strong>{formatScore(result.kl_l1_mmr[field])}</strong></div>
                  ))}
                </div>
                <p className="evaluation-summary">{result.kl_l1_mmr.summary}</p>
              </div>
              <div className="evaluation-candidate">
                <h3>Frobenius-NMF + {penaltyLabel(l1Ratio)} + MMR</h3>
                <div className="demo-scores">
                  {ROUGE_METRICS.map(([field, label]) => (
                    <div key={field}><span>{label}</span><strong>{formatScore(result.frobenius[field])}</strong></div>
                  ))}
                </div>
                <p className="evaluation-summary">{result.frobenius.summary}</p>
              </div>
            </>
          ) : (
            <div className="demo-empty">
              <span aria-hidden="true">R</span>
              <strong>Chưa có kết quả</strong>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
