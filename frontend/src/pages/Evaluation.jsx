import { useEffect, useState } from 'react';
import toy from '../../../data/demo/toy.json';
import { evaluate, loadBatchMetrics } from '../api.js';
import { summarizeMetrics } from '../metrics.js';

function f1(value) { return Number(value).toFixed(3); }
function milliseconds(value) { return (Number(value) * 1000).toFixed(1); }

export default function Evaluation() {
  const [rows, setRows] = useState([]);
  const [batchLoading, setBatchLoading] = useState(true);
  const [batchError, setBatchError] = useState('');
  const [text, setText] = useState(toy.text);
  const [reference, setReference] = useState('');
  const [sentenceCount, setSentenceCount] = useState(3);
  const [demo, setDemo] = useState(null);
  const [demoLoading, setDemoLoading] = useState(false);
  const [demoError, setDemoError] = useState('');

  function updateDemoField(setter, value) {
    setter(value);
    setDemo(null);
    setDemoError('');
  }

  async function refreshBatch() {
    setBatchLoading(true);
    setBatchError('');
    try {
      setRows(await loadBatchMetrics());
    } catch (cause) {
      setRows([]);
      setBatchError(cause.message);
    } finally {
      setBatchLoading(false);
    }
  }

  useEffect(() => { refreshBatch(); }, []);

  async function handleDemo(event) {
    event.preventDefault();
    setDemo(null);
    setDemoError('');
    if (!text.trim() || !reference.trim()) {
      setDemoError('Cần cả văn bản và bản tóm tắt tham chiếu.');
      return;
    }
    setDemoLoading(true);
    try {
      setDemo(await evaluate({ text, reference_summary: reference, summary_sentences: Number(sentenceCount) }));
    } catch (cause) {
      setDemoError(cause.message);
    } finally {
      setDemoLoading(false);
    }
  }

  const aggregate = summarizeMetrics(rows);
  const articleCount = new Set(rows.map((row) => row.id)).size;
  const fallbackCount = rows.filter((row) => row.method === 'NMF' && row.fallback_reason).length;

  return (
    <section aria-labelledby="evaluation-heading">
      <div className="section-heading"><div><div className="section-kicker">03 / KIỂM CHỨNG</div><h2 id="evaluation-heading">Đo điều mô hình giữ lại</h2><p>So sánh NMF với Lead-N trên cùng văn bản, cùng số câu và cùng cách tính ROUGE.</p></div><span className="section-index">03</span></div>

      <div className="panel evaluation-overview"><div><span className="panel-label">THÍ NGHIỆM BATCH</span><h3>Lead-N <span>vs.</span> NMF</h3><p>Dữ liệu đọc trực tiếp từ <code>results/metrics.csv</code>. Chỉ hiển thị số liệu khi script thí nghiệm đã tạo file.</p></div><div className="evaluation-overview-side"><span>{articleCount}</span><small>BÀI ĐÃ ĐÁNH GIÁ</small><button type="button" className="secondary-button" onClick={refreshBatch} disabled={batchLoading}>{batchLoading ? 'Đang đọc…' : 'Làm mới dữ liệu ↻'}</button></div></div>

      {batchError && <div className="alert alert-error" role="alert">{batchError}</div>}
      {batchLoading && <div className="empty-state" role="status"><span className="loading-spinner" /><strong>Đang đọc kết quả thí nghiệm…</strong></div>}
      {!batchLoading && !batchError && rows.length === 0 && <div className="empty-state"><span className="empty-symbol">◫</span><strong>Chưa có kết quả batch</strong><p>Điền dữ liệu hợp lệ, chạy <code>scripts/run_experiment.py</code>, rồi nhấn “Làm mới dữ liệu”. Bản build tĩnh cần build lại sau khi có CSV mới.</p></div>}
      {!batchLoading && rows.length > 0 && (
        <div className="result-stack">
          <div className="batch-stats"><div className="panel stat-card"><span>SỐ BÀI HỢP LỆ</span><strong>{articleCount}</strong><small>Mỗi bài có một dòng cho mỗi phương pháp</small></div><div className="panel stat-card"><span>PHƯƠNG PHÁP</span><strong>{aggregate.length}</strong><small>{aggregate.map((item) => item.method).join(' · ')}</small></div><div className="panel stat-card"><span>NMF FALLBACK</span><strong>{fallbackCount}</strong><small>Bài chuyển sang Lead-N</small></div></div>
          <div className="panel table-panel"><div className="table-heading"><div><span className="panel-label">TRUNG BÌNH THEO PHƯƠNG PHÁP</span><h3>Chỉ số tổng hợp</h3></div><small>F1: 0–1 · thời gian: mili giây</small></div><div className="table-scroll"><table className="data-table"><thead><tr><th scope="col">Phương pháp</th><th scope="col">ROUGE-1</th><th scope="col">ROUGE-2</th><th scope="col">ROUGE-L</th><th scope="col">Thời gian</th><th scope="col">Số câu TB</th><th scope="col">Số bài</th></tr></thead><tbody>{aggregate.map((item) => <tr key={item.method}><th scope="row"><span className={`method-tag ${item.method === 'NMF' ? 'method-nmf' : ''}`}>{item.method}</span></th><td>{f1(item.rouge1_f1)}</td><td>{f1(item.rouge2_f1)}</td><td>{f1(item.rougeL_f1)}</td><td>{milliseconds(item.runtime_seconds)} ms</td><td>{item.sentence_count.toFixed(1)}</td><td>{item.count}</td></tr>)}</tbody></table></div></div>
          <details className="panel detail-table"><summary>Xem kết quả từng bài <span>({rows.length} dòng)</span></summary><div className="table-scroll"><table className="data-table"><thead><tr><th scope="col">ID bài</th><th scope="col">Phương pháp</th><th scope="col">R-1</th><th scope="col">R-2</th><th scope="col">R-L</th><th scope="col">Thời gian</th><th scope="col">Số câu</th><th scope="col">Ghi chú</th></tr></thead><tbody>{rows.map((item, index) => <tr key={`${item.id}-${item.method}-${index}`}><th scope="row">{item.id}</th><td>{item.method}</td><td>{f1(item.rouge1_f1)}</td><td>{f1(item.rouge2_f1)}</td><td>{f1(item.rougeL_f1)}</td><td>{milliseconds(item.runtime_seconds)} ms</td><td>{item.sentence_count}</td><td>{item.fallback_reason || '—'}</td></tr>)}</tbody></table></div></details>
        </div>
      )}

      <div className="demo-divider"><span>ĐÁNH GIÁ MỘT MẪU</span><i /></div>
      <div className="workspace-grid evaluation-demo">
        <form className="panel form-panel" onSubmit={handleDemo}>
          <div className="panel-top"><span className="panel-label">THỬ NGAY VỚI MỘT MẪU</span><span className="panel-meta">API / EVALUATE</span></div>
          <label className="field-label" htmlFor="evaluate-text">Văn bản nguồn</label>
          <textarea id="evaluate-text" rows={8} value={text} onChange={(event) => updateDemoField(setText, event.target.value)} placeholder="Nhập văn bản nguồn…" />
          <label className="field-label field-label-spaced" htmlFor="evaluate-reference">Bản tóm tắt tham chiếu</label>
          <textarea id="evaluate-reference" rows={4} value={reference} onChange={(event) => updateDemoField(setReference, event.target.value)} placeholder="Nhập bản tóm tắt tham chiếu của chính bạn…" />
          <label className="field-group compact-field">Số câu tóm tắt<input type="number" min="1" max="5" value={sentenceCount} onChange={(event) => updateDemoField(setSentenceCount, event.target.value)} required /></label>
          {demoError && <div className="alert alert-error" role="alert">{demoError}</div>}
          <button className="primary-button" type="submit" disabled={demoLoading}>{demoLoading ? 'Đang đánh giá…' : 'Chấm điểm mẫu này'}<span aria-hidden="true">↗</span></button>
        </form>
        <div className="panel demo-result-panel"><div className="panel-top"><span className="panel-label">KẾT QUẢ MỘT MẪU</span><span className="panel-meta">ROUGE F1</span></div>{demo ? <><div className="demo-scores"><div><span>ROUGE-1</span><strong>{f1(demo.rouge1_f1)}</strong></div><div><span>ROUGE-2</span><strong>{f1(demo.rouge2_f1)}</strong></div><div><span>ROUGE-L</span><strong>{f1(demo.rougeL_f1)}</strong></div></div>{demo.fallback_reason && <div className="alert alert-warning">Dùng Lead-N: {demo.fallback_reason}</div>}<div className="demo-summary"><span className="panel-label">BẢN TÓM TẮT NMF</span><p>{demo.summary}</p></div><p className="demo-note">Điểm của một mẫu chỉ để kiểm tra nhanh; bảng đối chứng phía trên lấy từ thí nghiệm batch.</p></> : <div className="demo-empty"><span>◎</span><strong>Chờ bản tham chiếu</strong><p>Nhập tóm tắt tham chiếu để xem ROUGE-1/2/L của một văn bản.</p></div>}</div>
      </div>
    </section>
  );
}
