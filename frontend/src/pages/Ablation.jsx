import { useEffect, useState } from 'react';
import { ablationArtifactUrl, getAblationJob, startAblation } from '../api.js';

const CONFIGS = [
  ['kl_no_l1', 'KL, không L1'],
  ['kl_l1_w_only', 'KL + L1 trên W'],
  ['kl_l1_h_only', 'KL + L1 trên H'],
  ['kl_l1_w_h', 'KL + L1 trên W và H'],
];
const ROUGE_FIELDS = [
  ['rouge1_f1', 'ROUGE-1'],
  ['rouge2_f1', 'ROUGE-2'],
  ['rougeL_f1', 'ROUGE-L'],
];

function score(value) {
  return value === null || value === undefined ? '-' : Number(value).toFixed(3);
}

function meanStd(stat) {
  if (!stat || stat.mean === null || stat.mean === undefined) return '-';
  return `${Number(stat.mean).toFixed(3)} ± ${Number(stat.std || 0).toFixed(3)}`;
}

function parseSeedList(value) {
  const parts = value.split(',').map((part) => part.trim()).filter(Boolean);
  if (!parts.length || parts.length > 20 || parts.some((part) => !/^\d+$/.test(part))) {
    throw new Error('Nhập 1–20 seed không âm, phân cách bằng dấu phẩy.');
  }
  const seeds = parts.map(Number);
  if (seeds.some((seed) => seed > 4_294_967_295) || new Set(seeds).size !== seeds.length) {
    throw new Error('Seed phải duy nhất và nằm trong khoảng 0–4294967295.');
  }
  return seeds;
}

export default function Ablation() {
  const [articleId, setArticleId] = useState('manual-article');
  const [article, setArticle] = useState('');
  const [referenceSummary, setReferenceSummary] = useState('');
  const [k, setK] = useState('');
  const [summarySentences, setSummarySentences] = useState(3);
  const [mmrLambda, setMmrLambda] = useState(0.7);
  const [alpha, setAlpha] = useState(0.001);
  const [seeds, setSeeds] = useState('42');
  const [runId, setRunId] = useState('');
  const [job, setJob] = useState(null);
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const isBusy = submitting || ['queued', 'running'].includes(job?.status);

  useEffect(() => {
    if (!runId || !job || !['queued', 'running'].includes(job.status)) return undefined;
    let cancelled = false;
    let timer;
    const poll = async () => {
      try {
        const next = await getAblationJob(runId);
        if (cancelled) return;
        setJob(next);
        if (['queued', 'running'].includes(next.status)) timer = window.setTimeout(poll, 1200);
      } catch (cause) {
        if (!cancelled) setError(cause.message);
      }
    };
    timer = window.setTimeout(poll, 500);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [runId, job?.status]);

  function updateField(setter) {
    return (event) => {
      setter(event.target.value);
      setError('');
      if (!['queued', 'running'].includes(job?.status)) {
        setJob(null);
        setRunId('');
      }
    };
  }

  async function onSubmit(event) {
    event.preventDefault();
    setError('');
    setJob(null);
    setRunId('');
    if (!article.trim() || !referenceSummary.trim()) {
      setError('Hãy nhập cả văn bản nguồn và reference summary.');
      return;
    }
    setSubmitting(true);
    try {
      const body = {
        article_id: articleId.trim() || 'manual-article',
        article: article.trim(),
        reference_summary: referenceSummary.trim(),
        k: k === '' ? null : Number(k),
        summary_sentences: Number(summarySentences),
        mmr_lambda: Number(mmrLambda),
        alpha: Number(alpha),
        seeds: parseSeedList(seeds),
      };
      const started = await startAblation(body);
      setRunId(started.run_id);
      setJob({ ...started, completed: 0 });
    } catch (cause) {
      setError(cause.message);
    } finally {
      setSubmitting(false);
    }
  }

  const report = job?.report;
  const progress = job?.total ? Math.min(100, (job.completed / job.total) * 100) : 0;

  return (
    <section aria-labelledby="ablation-heading">
      <div className="section-heading">
        <div>
          <div className="section-kicker">04 / KIỂM TRA THÀNH PHẦN</div>
          <h2 id="ablation-heading">Ablation điều chuẩn L1</h2>
          <p>Giữ KL-NMF, MMR, k và seed đồng nhất; thay đổi vị trí áp dụng L1 trên W/H.</p>
        </div>
        <span className="section-index">04</span>
      </div>

      <div className="ablation-configs" aria-label="Các cấu hình so sánh">
        {CONFIGS.map(([id, label], index) => (
          <div className={`ablation-config ${index === 0 ? 'is-baseline' : ''}`} key={id}>
            <span>{String(index + 1).padStart(2, '0')}</span>
            <strong>{label}</strong>
            <small>{id}</small>
          </div>
        ))}
      </div>

      <div className="ablation-layout">
        <form className="panel form-panel ablation-form" onSubmit={onSubmit}>
          <div className="panel-top"><span className="panel-label">VĂN BẢN VÀ THIẾT LẬP</span><span className="panel-meta">THỬ NGHIỆM MỘT BÀI</span></div>
          <div className="ablation-inputs">
            <label className="field-group">ID bài (tùy chọn)
              <input type="text" value={articleId} onChange={updateField(setArticleId)} maxLength="200" disabled={isBusy} />
            </label>
            <label className="field-group">Văn bản nguồn
              <textarea className="ablation-textarea" value={article} onChange={updateField(setArticle)} placeholder="Nhập đoạn văn cần đánh giá..." maxLength="100000" required disabled={isBusy} />
            </label>
            <label className="field-group">Reference summary
              <textarea className="ablation-textarea ablation-reference" value={referenceSummary} onChange={updateField(setReferenceSummary)} placeholder="Nhập bản tóm tắt tham chiếu để tính ROUGE F1..." maxLength="20000" required disabled={isBusy} />
            </label>
          </div>
          <p className="field-hint ablation-input-hint">Một lần chạy thủ công được ghi nhận là <code>pilot</code>. Reference summary là căn cứ tính ROUGE F1; hệ thống không tự tạo nhãn tham chiếu.</p>

          <div className="form-row two-cols ablation-fields">
            <label className="field-group">Số câu tóm tắt
              <input type="number" min="1" max="20" value={summarySentences} onChange={updateField(setSummarySentences)} required disabled={isBusy} />
            </label>
            <label className="field-group">k dùng chung
              <input type="number" min="1" value={k} onChange={updateField(setK)} placeholder="Tự chọn một lần" disabled={isBusy} />
            </label>
            <label className="field-group">MMR λ
              <input type="number" min="0" max="1" step="0.05" value={mmrLambda} onChange={updateField(setMmrLambda)} required disabled={isBusy} />
            </label>
            <label className="field-group">α L1
              <input type="number" min="0" max="1" step="0.0005" value={alpha} onChange={updateField(setAlpha)} required disabled={isBusy} />
            </label>
            <label className="field-group ablation-seeds">Seeds, phân cách bằng dấu phẩy
              <input type="text" value={seeds} onChange={updateField(setSeeds)} placeholder="42, 43, 44" required disabled={isBusy} />
            </label>
          </div>
          {error && <div className="alert alert-error" role="alert">{error}</div>}
          <button className="primary-button" type="submit" disabled={isBusy}>
            {submitting ? 'Đang gửi dữ liệu…' : ['queued', 'running'].includes(job?.status) ? 'Đang chạy ablation…' : 'Chạy bốn cấu hình'}
            <span aria-hidden="true">→</span>
          </button>
        </form>

        <aside className="panel insight-panel ablation-method">
          <div className="panel-top"><span className="panel-label">KIỂM SOÁT CÔNG BẰNG</span><span className="panel-meta">LOCAL KL-NMF + MMR</span></div>
          <div className="ablation-equation">D<sub>KL</sub>(V ‖ WH) + L1(W,H)</div>
          <ul>
            <li>Cùng văn bản, seed, TF-IDF, ngân sách câu và MMR λ.</li>
            <li>Chọn k theo nhánh không L1 một lần; dùng cùng k cho cả bốn nhánh.</li>
            <li>Độ thưa tính từ W/H vừa fit, với ngưỡng tuyệt đối 10⁻⁸.</li>
            <li>Không so sánh trực tiếp objective giữa cấu hình có penalty khác nhau.</li>
          </ul>
          <div className="insight-foot">L1 W ONLY · L1 H ONLY · L1 W + H</div>
        </aside>
      </div>

      {job && <div className="panel ablation-job" aria-live="polite">
        <div className="ablation-job-top"><div><span className="panel-label">LẦN CHẠY {job.run_id?.slice(0, 10)}</span><strong>{job.status === 'complete' ? 'Hoàn tất' : job.status === 'failed' ? 'Có lỗi' : job.status === 'queued' ? 'Đang chờ' : 'Đang xử lý'}</strong></div><span>{job.completed || 0} / {job.total || 0} cặp bài-seed</span></div>
        <progress max="100" value={report ? 100 : progress} />
        {job.error && <div className="alert alert-error">{job.error}</div>}
      </div>}

      {report && <div className="result-stack ablation-results">
        <div className="result-heading"><div><span className="panel-label">KẾT QUẢ GHÉP CẶP</span><h3>{report.valid_articles.toLocaleString()} bài · {report.config.seeds.length} seed</h3></div><div className="ablation-downloads">
          <a href={ablationArtifactUrl(runId, 'metrics')}>CSV metrics</a>
          <a href={ablationArtifactUrl(runId, 'predictions')}>JSONL predictions</a>
          <a href={ablationArtifactUrl(runId, 'report')}>JSON report</a>
        </div></div>
        <div className="panel table-panel ablation-table-panel"><div className="table-heading"><div><span className="panel-label">ROUGE F1 VÀ CHÊNH LỆCH CẶP</span><h3>So với KL không L1</h3></div><small>ROUGE: mean ± SD giữa các seed thành công</small></div>
          <div className="table-scroll"><table className="data-table ablation-table"><thead><tr>
            <th>Cấu hình</th><th>α W / H</th><th>ROUGE-1</th><th>ROUGE-2</th><th>ROUGE-L</th><th>Δ R-1</th><th>Δ R-2</th><th>Δ R-L</th><th>W gần 0</th><th>H gần 0</th><th>Topic/câu</th><th>Từ/topic</th><th>Fallback</th>
          </tr></thead><tbody>{CONFIGS.map(([id, label]) => {
            const stats = report.methods[id];
            const delta = report.paired_deltas_vs_kl_no_l1[id];
            const params = report.config.configurations[id];
            return <tr key={id} className={id === 'kl_no_l1' ? 'ablation-baseline-row' : ''}>
              <th scope="row"><span className="ablation-row-label">{label}</span><small>{id}</small></th>
              <td>{params.alpha_w.toFixed(2)} / {params.alpha_h.toFixed(2)}</td>
              {ROUGE_FIELDS.map(([field]) => <td key={field}>{meanStd(stats.rouge[field])}</td>)}
              {ROUGE_FIELDS.map(([field]) => <td key={field} className="ablation-delta">{score(delta.rouge[field].mean)}</td>)}
              <td>{score(stats.w_zero_fraction.mean)}</td><td>{score(stats.h_zero_fraction.mean)}</td>
              <td>{score(stats.active_topics_per_sentence.mean)}</td><td>{score(stats.active_terms_per_topic.mean)}</td>
              <td>{stats.n_fallback} / {report.expected_runs_per_configuration}</td>
            </tr>;
          })}</tbody></table></div>
        </div>
        <div className="ablation-result-notes">
          <p>Δ ROUGE là chênh lệch theo cùng <code>ID bài + seed</code> so với <code>kl_no_l1</code>; số dương nghĩa là tăng trên mẫu này, không khẳng định tổng quát.</p>
          <p>W/H gần 0 là tỷ lệ phần tử ≤ {report.config.factor_zero_threshold}. Active topic/câu và active từ/topic là số lượng trung bình vượt ngưỡng đó.</p>
          {report.run_error_count > 0 && <p>{report.run_error_count} lượt lỗi; xem <code>ablation_report.json</code> để biết chi tiết.</p>}
        </div>
      </div>}
    </section>
  );
}
