import { useEffect, useState } from 'react';
import { health } from './api.js';
import Visualizer from './pages/Visualizer.jsx';
import Summarizer from './pages/Summarizer.jsx';
import Evaluation from './pages/Evaluation.jsx';

const pages = [
  { id: 'visualizer', number: '01', label: 'Mô phỏng NMF', caption: 'KL loss & ma trận' },
  { id: 'summarizer', number: '02', label: 'Tóm tắt văn bản', caption: 'Chủ đề & câu trích' },
  { id: 'evaluation', number: '03', label: 'Đánh giá', caption: 'Đối chứng & ROUGE' },
];

export default function App() {
  const [active, setActive] = useState('visualizer');
  const [serverStatus, setServerStatus] = useState('checking');

  useEffect(() => {
    let mounted = true;
    const check = () => health()
      .then(() => { if (mounted) setServerStatus('online'); })
      .catch(() => { if (mounted) setServerStatus('offline'); });
    check();
    const timer = window.setInterval(check, 30_000);
    return () => { mounted = false; window.clearInterval(timer); };
  }, []);

  const current = pages.find((page) => page.id === active);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-block">
          <div className="brand-mark" aria-hidden="true"><span /><span /><span /><span /></div>
          <div>
            <div className="brand-name">TopicSum<span className="brand-dot">.</span></div>
            <div className="brand-subtitle">Vietnamese NMF Lab</div>
          </div>
        </div>

        <div className="sidebar-section-label">KHÔNG GIAN LÀM VIỆC</div>
        <nav className="side-nav" aria-label="Các màn hình TopicSum">
          {pages.map((page) => (
            <button
              key={page.id}
              type="button"
              className={`nav-item ${active === page.id ? 'is-active' : ''}`}
              onClick={() => setActive(page.id)}
              aria-current={active === page.id ? 'page' : undefined}
            >
              <span className="nav-number">{page.number}</span>
              <span className="nav-copy"><strong>{page.label}</strong><small>{page.caption}</small></span>
              <span className="nav-arrow" aria-hidden="true">→</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="sidebar-note"><span className="note-line" />Luồng tóm tắt dùng TF-IDF cục bộ, NMF KL divergence, L1 sparsity và MMR.</div>
          <div className={`server-status status-${serverStatus}`} aria-live="polite">
            <span className="status-indicator" />
            {serverStatus === 'online' ? 'Backend đang kết nối' : serverStatus === 'offline' ? 'Backend chưa kết nối' : 'Đang kiểm tra backend'}
          </div>
        </div>
      </aside>

      <main className="main-content">
        <div className="topline">
          <span>TOPICSUM / {current.number} - {current.label.toUpperCase()}</span>
          <span className="topline-tag">TRÍCH XUẤT · TIẾNG VIỆT</span>
        </div>
        <header className="hero">
          <div className="hero-copy">
            <div className="eyebrow"><span className="eyebrow-line" />PHÂN TÍCH CHỦ ĐỀ BẰNG NMF</div>
            <h1>Từ văn bản đến <em>ý chính.</em></h1>
            <p>Quan sát ma trận, chọn câu bằng MMR và đo chất lượng tóm tắt với cấu hình NMF phù hợp văn bản thưa.</p>
          </div>
          <div className="hero-visual" aria-hidden="true">
            <div className="hero-grid"><span /><span /><span /><span /><span /><span /><span /><span /><span /></div>
            <div className="hero-formula">V <span>≈</span> W x H</div>
            <div className="hero-formula-note">SENTENCES x TERMS</div>
          </div>
        </header>

        <div className="page-content">
          {active === 'visualizer' && <Visualizer />}
          {active === 'summarizer' && <Summarizer />}
          {active === 'evaluation' && <Evaluation />}
        </div>
        <footer className="site-footer"><span>TOPICSUM / NPL-NMF</span><span>Giữ nguyên câu gốc · Không sinh văn bản bằng LLM</span></footer>
      </main>
    </div>
  );
}
