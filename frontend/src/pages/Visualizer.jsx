import { useState } from 'react';
import toy from '../../../data/demo/toy.json';
import { simulate } from '../api.js';

function Matrix({ name, subtitle, values, rows, columns }) {
  if (!values?.length) return null;
  return (
    <div className="matrix-card">
      <div className="matrix-heading">
        <div><strong>{name}</strong><span>{subtitle}</span></div>
        <small>{values.length} x {columns.length}</small>
      </div>
      <div className="matrix-scroll">
        <table className="matrix-table">
          <thead>
            <tr>
              <th scope="col" />
              {columns.map((label, index) => <th scope="col" key={`${label}-${index}`} title={label}>{label}</th>)}
            </tr>
          </thead>
          <tbody>{values.map((line, rowIndex) => (
            <tr key={rowIndex}>
              <th scope="row">{rows[rowIndex]}</th>
              {line.map((value, columnIndex) => (
                <td key={columnIndex} title={Number(value).toFixed(6)}>{Number(value).toFixed(3)}</td>
              ))}
            </tr>
          ))}</tbody>
        </table>
      </div>
    </div>
  );
}

function lossLabel(lossName) {
  return lossName === 'kullback-leibler' ? 'KL + L1 loss' : 'Loss';
}

function format(value) {
  return Number(value ?? 0).toFixed(6);
}

export default function Visualizer() {
  const [text, setText] = useState(toy.text);
  const [k, setK] = useState('');
  const [iterations, setIterations] = useState(10);
  const [seed, setSeed] = useState(42);
  const [result, setResult] = useState(null);
  const [step, setStep] = useState(0);
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
    if (!text.trim()) { setError('Hãy nhập văn bản để mô phỏng.'); return; }
    setLoading(true);
    try {
      const payload = { text, iterations: Number(iterations), seed: Number(seed) };
      if (k !== '') payload.k = Number(k);
      const data = await simulate(payload);
      setResult(data);
      setStep(0);
    } catch (cause) {
      setError(cause.message);
    } finally {
      setLoading(false);
    }
  }

  const snapshots = result?.snapshots || [];
  const snapshot = snapshots[step];
  const lastStep = snapshots.length - 1;
  const sentenceLabels = result?.sentences?.map((_, index) => `C${index + 1}`) || [];
  const topicLabels = snapshot?.H?.map((_, index) => `T${index + 1}`) || [];
  const maxLoss = snapshots.length ? Math.max(...snapshots.map((item) => item.loss), 0.000001) : 1;

  return (
    <section aria-labelledby="visualizer-heading">
      <div className="section-heading">
        <div>
          <div className="section-kicker">01 / TRỰC QUAN HÓA</div>
          <h2 id="visualizer-heading">Bên trong mỗi vòng lặp</h2>
          <p>Theo dõi W, H, KL loss, L1 penalty và cách backend tự chọn k bằng masked imputation.</p>
        </div>
        <span className="section-index">01</span>
      </div>

      <div className="workspace-grid">
        <form className="panel form-panel" onSubmit={handleSubmit}>
          <div className="panel-top"><span className="panel-label">ĐẦU VÀO</span><span className="panel-meta">LOCAL KL-NMF</span></div>
          <label className="field-label" htmlFor="simulate-text">Văn bản tiếng Việt</label>
          <textarea id="simulate-text" value={text} onChange={(event) => updateField(setText, event.target.value)} maxLength={20000} rows={11} placeholder="Nhập văn bản..." />
          <div className="field-hint">
            <span>Mỗi câu là một document trong TF-IDF cục bộ.</span>
            <button type="button" className="text-button" onClick={() => { setText(toy.text); setResult(null); setError(''); }}>Nạp ví dụ</button>
          </div>
          <div className="form-row three-cols">
            <label className="field-group">Số chủ đề k<input type="number" min="1" max="10" value={k} onChange={(event) => updateField(setK, event.target.value)} placeholder="Tự chọn" /></label>
            <label className="field-group">Số vòng lặp<input type="number" min="1" max="20" value={iterations} onChange={(event) => updateField(setIterations, event.target.value)} required /></label>
            <label className="field-group">Seed<input type="number" value={seed} onChange={(event) => updateField(setSeed, event.target.value)} required /></label>
          </div>
          {error && <div className="alert alert-error" role="alert">{error}</div>}
          <button className="primary-button" type="submit" disabled={loading}>{loading ? 'Đang tính toán...' : 'Chạy mô phỏng'}<span aria-hidden="true">→</span></button>
        </form>

        <div className="panel insight-panel">
          <div className="panel-top"><span className="panel-label">CÁCH ĐỌC</span><span className="panel-meta">KL / MU / L1</span></div>
          <div className="equation-card"><span>V</span><b>≈</b><span>W</span><b>x</b><span>H</span></div>
          <p><strong>V</strong> là TF-IDF cục bộ của câu và từ. <strong>W</strong> là trọng số câu - chủ đề. <strong>H</strong> là trọng số chủ đề - từ vựng.</p>
          <div className="insight-rule" />
          <p>Loss tổng = KL divergence của dữ liệu + L1 penalty trên W và H. Nếu k để trống, backend che một phần ô TF-IDF dương rồi chọn k tái tạo tốt nhất.</p>
          <div className="insight-foot">SOLVER MU · ALPHA W/H 0.1 · L1 RATIO 1.0</div>
        </div>
      </div>

      {!result && !loading && <div className="empty-state"><span className="empty-symbol">▦</span><strong>Chưa có ma trận để hiển thị</strong><p>Nhấn “Chạy mô phỏng” để xem NMF KL + L1 trên văn bản bên trái.</p></div>}
      {loading && <div className="empty-state" role="status"><span className="loading-spinner" /><strong>Đang tạo các ma trận...</strong></div>}
      {result && snapshot && (
        <div className="result-stack">
          <div className="result-heading">
            <div><div className="section-kicker">KẾT QUẢ THỰC TẾ</div><h3>Diễn tiến phân rã</h3></div>
            <div className="result-stats">
              <span>{result.sentences.length} câu</span>
              <span>{result.terms.length} từ</span>
              <span>k = {result.k}</span>
              <span>{result.solver?.toUpperCase()} / KL</span>
            </div>
          </div>

          <div className="panel timeline-panel">
            <div className="timeline-top">
              <div><span className="panel-label">VÒNG LẶP</span><strong>{String(snapshot.iteration).padStart(2, '0')} <small>/ {String(lastStep).padStart(2, '0')}</small></strong></div>
              <div className="loss-number">{lossLabel(result.loss_name)} <strong>{format(snapshot.loss)}</strong></div>
            </div>
            <div className="loss-breakdown">
              <span>KL data <strong>{format(snapshot.data_loss)}</strong></span>
              <span>L1 penalty <strong>{format(snapshot.regularization_loss)}</strong></span>
              <span>Chọn k <strong>{result.k_selection_method}</strong></span>
            </div>
            <div className="timeline-control">
              <button type="button" onClick={() => setStep((value) => Math.max(0, value - 1))} disabled={step === 0} aria-label="Vòng trước">←</button>
              <input type="range" min="0" max={lastStep} value={step} onChange={(event) => setStep(Number(event.target.value))} aria-label="Chọn vòng lặp" />
              <button type="button" onClick={() => setStep((value) => Math.min(lastStep, value + 1))} disabled={step === lastStep} aria-label="Vòng sau">→</button>
            </div>
            <div className="loss-chart" aria-label="Loss theo từng vòng lặp">{snapshots.map((item, index) => <button type="button" key={item.iteration} className={`loss-bar ${index === step ? 'is-current' : ''}`} onClick={() => setStep(index)} style={{ height: `${Math.max(7, item.loss / maxLoss * 100)}%` }} title={`Vòng ${item.iteration}: ${item.loss.toFixed(6)}`} aria-label={`Xem vòng ${item.iteration}, loss ${item.loss.toFixed(6)}`} />)}</div>
          </div>

          {result.k_candidates?.length > 0 && <div className="panel k-panel"><div className="panel-top"><span className="panel-label">ỨNG VIÊN k</span><span className="panel-meta">Điểm càng thấp càng tốt</span></div><div className="k-candidates">{result.k_candidates.map((candidate) => <span className={candidate.k === result.k ? 'is-selected' : ''} key={candidate.k}>k={candidate.k}<strong>{Number(candidate.score).toFixed(5)}</strong></span>)}</div></div>}

          <div className="matrix-grid">
            <Matrix name="V" subtitle="TF-IDF câu x từ" values={result.V} rows={sentenceLabels} columns={result.terms} />
            <Matrix name="W" subtitle="Câu x chủ đề" values={snapshot.W} rows={sentenceLabels} columns={topicLabels} />
            <Matrix name="H" subtitle="Chủ đề x từ" values={snapshot.H} rows={topicLabels} columns={result.terms} />
            <Matrix name="WH" subtitle="Ma trận tái tạo" values={snapshot.WH} rows={sentenceLabels} columns={result.terms} />
          </div>
          <details className="panel sentence-legend"><summary>Xem câu tương ứng với C1, C2...</summary><ol>{result.sentences.map((sentence, index) => <li key={index}><strong>C{index + 1}</strong> {sentence}</li>)}</ol></details>
        </div>
      )}
    </section>
  );
}
