import { useState, useRef, useCallback, useEffect } from 'react';
import './App.css';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const HISTORY_KEY = 'agritech_history';
const MAX_HISTORY = 12;

const SHORT_NAMES = {
  'Tomato___Bacterial_spot': 'Bacterial Spot',
  'Tomato___Early_blight': 'Early Blight',
  'Tomato___Late_blight': 'Late Blight',
  'Tomato___Septoria_leaf_spot': 'Septoria',
  'Tomato___healthy': 'Healthy',
};

function createThumbnail(file) {
  return new Promise((resolve) => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const img = new window.Image();
    img.onload = () => {
      canvas.width = 120;
      canvas.height = 90;
      ctx.drawImage(img, 0, 0, 120, 90);
      resolve(canvas.toDataURL('image/jpeg', 0.6));
      URL.revokeObjectURL(img.src);
    };
    img.src = URL.createObjectURL(file);
  });
}

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [mode, setMode] = useState('ensemble');
  const [dragging, setDragging] = useState(false);
  const [activeTab, setActiveTab] = useState('diagnosis');
  const [heatmapData, setHeatmapData] = useState(null);
  const [heatmapMethod, setHeatmapMethod] = useState(null);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [isLoadingHeatmap, setIsLoadingHeatmap] = useState(false);
  const [history, setHistory] = useState([]);
  const [similarCases, setSimilarCases] = useState(null);
  const [isLoadingSimilar, setIsLoadingSimilar] = useState(false);
  const [advisory, setAdvisory] = useState(null);
  const [isLoadingAdvisory, setIsLoadingAdvisory] = useState(false);
  const [userContext, setUserContext] = useState('');
  const fileRef = useRef(null);

  // Load history on mount
  useEffect(() => {
    try {
      const saved = localStorage.getItem(HISTORY_KEY);
      if (saved) setHistory(JSON.parse(saved));
    } catch { /* ignore */ }
  }, []);

  const handleFile = useCallback((file) => {
    if (!file) return;
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResults(null);
    setError(null);
    setHeatmapData(null);
    setHeatmapMethod(null);
    setShowHeatmap(false);
    setActiveTab('diagnosis');
    setSimilarCases(null);
    setAdvisory(null);
  }, []);

  const analyze = async () => {
    if (!selectedFile) return;
    setIsAnalyzing(true);
    setError(null);
    setResults(null);
    setHeatmapData(null);
    setHeatmapMethod(null);

    const fd = new FormData();
    fd.append('file', selectedFile);

    try {
      const res = await fetch(`${API_URL}/predict?mode=${mode}`, { method: 'POST', body: fd });
      if (!res.ok) throw new Error(`Server responded with ${res.status}`);
      const data = await res.json();
      setResults(data);
      setActiveTab('diagnosis');

      // Save to history
      const thumb = await createThumbnail(selectedFile);
      const entry = {
        id: Date.now(),
        thumbnail: thumb,
        prediction: data.prediction,
        confidence: data.confidence,
        severity: data.severity,
        mode,
      };
      setHistory(prev => {
        const next = [entry, ...prev].slice(0, MAX_HISTORY);
        localStorage.setItem(HISTORY_KEY, JSON.stringify(next));
        return next;
      });

      // Fetch Grad-CAM + similar cases in background
      fetchHeatmap();
      fetchSimilar();
    } catch (e) {
      setError(e.message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const fetchHeatmap = async () => {
    if (!selectedFile) return;
    setIsLoadingHeatmap(true);
    const fd = new FormData();
    fd.append('file', selectedFile);
    const heatmapModel = mode === 'ensemble' ? 'efnet' : mode;
    try {
      const res = await fetch(`${API_URL}/gradcam?model=${heatmapModel}`, { method: 'POST', body: fd });
      if (res.ok) {
        const data = await res.json();
        if (data.success) {
          setHeatmapData(data.gradcam_base64 || data.heatmap_base64);
          setHeatmapMethod(data.method || 'gradcam');
        }
      }
    } catch (e) {
      console.warn('Grad-CAM fetch failed:', e);
    } finally {
      setIsLoadingHeatmap(false);
    }
  };

  const fetchSimilar = async () => {
    if (!selectedFile) return;
    setIsLoadingSimilar(true);
    const fd = new FormData();
    fd.append('file', selectedFile);
    try {
      const res = await fetch(`${API_URL}/similar?top_k=3`, { method: 'POST', body: fd });
      if (res.ok) {
        const data = await res.json();
        if (data.success) setSimilarCases(data.similar_cases);
      }
    } catch (e) {
      console.warn('Similar cases fetch failed:', e);
    } finally {
      setIsLoadingSimilar(false);
    }
  };

  const fetchAdvisory = async () => {
    if (!results) return;
    setIsLoadingAdvisory(true);
    setAdvisory(null);
    const params = new URLSearchParams({
      disease_class: results.prediction_class,
      confidence: results.confidence,
      severity: results.severity,
      models_agree: results.models_agree,
      user_context: userContext,
    });
    try {
      const res = await fetch(`${API_URL}/advisor?${params}`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        if (data.success) setAdvisory(data.advisory);
      } else {
        const err = await res.json().catch(() => ({}));
        console.warn('Advisor failed:', err.detail || res.status);
      }
    } catch (e) {
      console.warn('Advisor fetch failed:', e);
    } finally {
      setIsLoadingAdvisory(false);
    }
  };

  const renderMarkdown = (md) => {
    if (!md) return null;
    // Lightweight markdown renderer: headings, bold, lists, line breaks
    return md.split('\n').map((line, i) => {
      if (line.startsWith('## ')) return <h4 key={i} className="md-h2">{line.replace('## ', '')}</h4>;
      if (line.startsWith('### ')) return <h5 key={i} className="md-h3">{line.replace('### ', '')}</h5>;
      if (/^\d+\.\s/.test(line)) {
        const text = line.replace(/^\d+\.\s/, '');
        return <div key={i} className="md-list-item"><span className="md-num">{line.match(/^(\d+)/)[1]}.</span> <span dangerouslySetInnerHTML={{__html: text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')}} /></div>;
      }
      if (line.startsWith('- ')) {
        const text = line.replace('- ', '');
        return <div key={i} className="md-list-item"><span className="md-bullet">•</span> <span dangerouslySetInnerHTML={{__html: text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')}} /></div>;
      }
      if (line.trim() === '') return <div key={i} className="md-spacer" />;
      return <p key={i} className="md-para" dangerouslySetInnerHTML={{__html: line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')}} />;
    });
  };

  const clearAll = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResults(null);
    setError(null);
    setHeatmapData(null);
    setHeatmapMethod(null);
    setShowHeatmap(false);
    setSimilarCases(null);
    setAdvisory(null);
  };

  const clearHistory = () => {
    setHistory([]);
    localStorage.removeItem(HISTORY_KEY);
  };

  const severityClass = (sev) => {
    if (!sev || sev === 'None') return 'healthy';
    if (sev === 'Critical') return 'critical';
    return 'moderate';
  };

  return (
    <div className="app-wrapper">
      {/* Navbar */}
      <nav className="navbar">
        <div className="nav-brand">
          <div className="nav-logo">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 3c.132 0 .263 0 .393 0a7.5 7.5 0 0 0 7.92 12.446A9 9 0 1 1 12 3z" />
            </svg>
          </div>
          <div>
            <div className="nav-title"><span>AgriTech</span></div>
            <div className="nav-subtitle">Diagnostic Center</div>
          </div>
        </div>
        <div className="nav-status">
          <div className="pulse-dot"></div>
          <span className="status-text">System Online</span>
        </div>
      </nav>

      <main className="main-grid">
        {/* Left Column */}
        <section className="left-panel">
          <div
            className={`upload-zone ${previewUrl ? 'has-image' : ''} ${dragging ? 'dragging' : ''}`}
            onClick={() => !previewUrl && fileRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]); }}
          >
            {isAnalyzing && <div className="scan-overlay"><div className="scan-laser"></div></div>}

            {previewUrl ? (
              <div className="preview-container">
                <img src={previewUrl} className="preview-img" alt="Leaf sample" />
                {showHeatmap && heatmapData && (
                  <div className="heatmap-overlay">
                    <img src={`data:image/png;base64,${heatmapData}`} alt="Grad-CAM overlay" />
                  </div>
                )}
                <button className="reupload-btn" onClick={(e) => { e.stopPropagation(); fileRef.current?.click(); }}>
                  <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182m0-4.991v4.99" />
                  </svg>
                </button>
              </div>
            ) : (
              <div className="upload-content">
                <div className="upload-icon-ring fade-up">
                  <svg fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 16.5V9.75m0 0l3 3m-3-3l-3 3M6.75 19.5a4.5 4.5 0 01-1.41-8.775 5.25 5.25 0 0110.338-2.32 3 3 0 013.613 3.853A4 4 0 0118 19.5H6.75z" />
                  </svg>
                </div>
                <h2 className="fade-up" style={{animationDelay: '0.05s'}}>Upload Specimen</h2>
                <p className="fade-up" style={{animationDelay: '0.1s'}}>Drop a tomato leaf image here for AI-powered disease analysis.</p>
                <div className="browse-btn fade-up" style={{animationDelay: '0.15s'}}>Select File</div>
              </div>
            )}
            <input ref={fileRef} type="file" accept="image/*" style={{ display: 'none' }} onChange={(e) => handleFile(e.target.files[0])} />
          </div>

          {/* Action Bar */}
          <div className="action-bar fade-in">
            <div className="mode-toggle">
              {['ensemble', 'efnet', 'resnet'].map((m) => (
                <button key={m} className={`toggle-btn ${mode === m ? 'active' : ''}`} onClick={() => setMode(m)}>
                  {m === 'ensemble' ? 'Ensemble' : m === 'efnet' ? 'EfNet' : 'ResNet'}
                </button>
              ))}
            </div>
            <div className="action-buttons">
              {previewUrl && <button className="clear-btn" onClick={clearAll}>Reset</button>}
              <button className="analyze-btn" disabled={!selectedFile || isAnalyzing} onClick={analyze}>
                {isAnalyzing ? (
                  <><span className="spinner"></span> Analyzing...</>
                ) : (
                  <>Analyze <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" /></svg></>
                )}
              </button>
            </div>
          </div>

          {/* Grad-CAM Controls */}
          {results && (
            <div className="heatmap-controls fade-up">
              <div className="heatmap-label">
                <svg fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M15.362 5.214A8.252 8.252 0 0112 21 8.25 8.25 0 016.038 7.047 8.287 8.287 0 009 9.601a8.983 8.983 0 013.361-6.867 8.21 8.21 0 003 2.48z" />
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 18a3.75 3.75 0 00.495-7.468 5.99 5.99 0 00-1.925 3.547 5.975 5.975 0 01-2.133-1.001A3.75 3.75 0 0012 18z" />
                </svg>
                Grad-CAM Overlay
              </div>
              {isLoadingHeatmap ? (
                <div className="heatmap-loading"><span className="spinner-dark"></span> Generating...</div>
              ) : heatmapData ? (
                <div className="heatmap-toggle-wrap">
                  {heatmapMethod && <span className="heatmap-method">{heatmapMethod === 'classifier-weighted-cam' ? 'CAM' : heatmapMethod}</span>}
                  <div className={`toggle-switch ${showHeatmap ? 'active' : ''}`} onClick={() => setShowHeatmap(!showHeatmap)} />
                </div>
              ) : (
                <span style={{fontSize: '0.75rem', color: 'var(--text-tertiary)'}}>Unavailable</span>
              )}
            </div>
          )}

          {error && (
            <div className="error-card fade-up">
              <div className="error-icon">!</div>
              <div className="error-text">
                <strong>Analysis Failed</strong>
                <p>{error}. Is the backend running?</p>
              </div>
            </div>
          )}

          {/* Model Comparison */}
          {results && results.diagnostics && Object.keys(results.diagnostics).length > 1 && (
            <div className="comparison-section fade-up">
              <h3 className="section-title">Model Comparison</h3>
              <div className="comparison-grid">
                {Object.entries(results.diagnostics).map(([model, data], idx) => (
                  <div key={model} className={`model-card ${results.models_agree ? 'agree' : 'disagree'}`} style={{animationDelay: `${idx * 0.05}s`}}>
                    <div className="model-header">
                      <span className="model-name">{model === 'efnet' ? 'EfficientNet-B0' : 'ResNet-50'}</span>
                      <span className={`badge ${model === 'efnet' ? 'fast' : 'deep'}`}>
                        {model === 'efnet' ? 'Fast' : 'Deep'}
                      </span>
                    </div>
                    <div className="model-prediction">{SHORT_NAMES[data.prediction] || data.prediction}</div>
                    <div className="metric-row">
                      <span>Confidence</span>
                      <span className="metric-val">{data.confidence}%</span>
                    </div>
                    <div className="progress-track">
                      <div className="progress-fill" style={{ width: `${data.confidence}%` }}></div>
                    </div>
                    <div className="dist-list">
                      {Object.entries(data.distribution).sort(([,a],[,b]) => b - a).slice(0,3).map(([cls, val]) => (
                        <div key={cls} className="dist-row">
                          <span className="dist-label">{SHORT_NAMES[cls] || cls}</span>
                          <div className="dist-track"><div className="dist-fill" style={{ width: `${val}%` }}></div></div>
                          <span className="dist-val">{val}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Similar Cases */}
          {results && (isLoadingSimilar || similarCases) && (
            <div className="similar-section fade-up">
              <h3 className="section-title">
                <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z" />
                </svg>
                Similar Reference Cases
              </h3>
              {isLoadingSimilar ? (
                <div className="similar-loading">
                  <div className="shimmer-card"></div>
                  <div className="shimmer-card"></div>
                  <div className="shimmer-card"></div>
                </div>
              ) : similarCases && similarCases.length > 0 ? (
                <div className="similar-grid">
                  {similarCases.map((c) => (
                    <div key={c.rank} className={`similar-card ${c.class === results.prediction_class ? 'match' : 'diff'}`}>
                      {c.thumbnail_base64 ? (
                        <img src={`data:image/jpeg;base64,${c.thumbnail_base64}`} alt={c.label} className="similar-thumb" />
                      ) : (
                        <div className="similar-thumb-placeholder" />
                      )}
                      <div className="similar-info">
                        <div className="similar-label">{c.label}</div>
                        <div className="similar-score">
                          <div className={`similarity-badge ${c.similarity >= 90 ? 'high' : c.similarity >= 70 ? 'med' : 'low'}`}>
                            {c.similarity}% match
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : null}
            </div>
          )}
        </section>

        {/* Right Column — Clinical Report */}
        <aside className="right-panel">
          <div className="report-card">
            <div className="report-header">
              <div className="icon-box">
                <svg fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
                </svg>
              </div>
              <h2>Clinical Report</h2>
            </div>

            {results ? (
              <div className="report-content fade-in">
                {/* Tabs */}
                <div className="tabs">
                  {[['diagnosis','Diagnosis'],['models','Models'],['details','Details'],['advisor','AI Advisor']].map(([key, label]) => (
                    <button key={key} className={`tab-btn ${activeTab === key ? 'active' : ''}`} onClick={() => setActiveTab(key)}>
                      {label}
                    </button>
                  ))}
                </div>

                {/* OOD Warning */}
                {results.validation && !results.validation.is_leaf && (
                  <div className="ood-warning fade-up">
                    <div className="ood-warning-icon">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
                      </svg>
                    </div>
                    <div>
                      <strong>Not a Tomato Leaf</strong>
                      <p>{results.validation.reason || 'This image does not appear to contain a tomato leaf. Results may be unreliable.'}</p>
                    </div>
                  </div>
                )}

                {/* Consensus banner */}
                {results.diagnostics && Object.keys(results.diagnostics).length > 1 && (
                  <div className={`consensus-banner ${results.models_agree ? 'agree' : 'conflict'}`}>
                    <div className="consensus-icon">
                      {results.models_agree ? (
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="20 6 9 17 4 12" /></svg>
                      ) : (
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>
                      )}
                    </div>
                    {results.models_agree ? 'Models Agree' : 'Models Disagree — Review Needed'}
                  </div>
                )}

                {/* Tab: Diagnosis */}
                {activeTab === 'diagnosis' && (
                  <>
                    <div className={`diagnosis-hero ${severityClass(results.severity)}`}>
                      <div className="hero-bg"></div>
                      <div className="hero-content">
                        <span className="pre-label">Detected Condition</span>
                        <h3>{results.prediction}</h3>
                        <div className="severity-pill">Severity: {results.severity}</div>
                      </div>
                    </div>
                    <div className="data-block">
                      <div className="block-title core-green">Ensemble Confidence</div>
                      <div className="big-metric">{results.confidence}%</div>
                      <div className="confidence-sub">
                        {results.is_confident === false ? '⚠ Below confidence threshold' : 'High confidence prediction'}
                      </div>
                    </div>
                  </>
                )}

                {/* Tab: Models */}
                {activeTab === 'models' && results.diagnostics && (
                  <div>
                    {Object.entries(results.diagnostics).map(([model, data]) => (
                      <div key={model} className="data-block cascade-in-1" style={{marginBottom: '0.75rem'}}>
                        <div className="block-title" style={{color: 'var(--text-primary)'}}>
                          {model === 'efnet' ? 'EfficientNet-B0' : 'ResNet-50'}
                        </div>
                        <div style={{fontFamily: 'var(--font-display)', fontSize: '1.25rem', fontWeight: 700, marginBottom: '0.5rem'}}>
                          {SHORT_NAMES[data.prediction] || data.prediction}
                          <span style={{fontSize: '0.875rem', fontWeight: 500, color: 'var(--green-600)', marginLeft: '0.5rem'}}>{data.confidence}%</span>
                        </div>
                        <div className="dist-list">
                          {Object.entries(data.distribution).sort(([,a],[,b]) => b - a).map(([cls, val]) => (
                            <div key={cls} className="dist-row">
                              <span className="dist-label">{SHORT_NAMES[cls] || cls}</span>
                              <div className="dist-track"><div className="dist-fill" style={{ width: `${val}%` }}></div></div>
                              <span className="dist-val">{val}%</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {/* Tab: Details */}
                {activeTab === 'details' && (
                  <>
                    {results.symptoms?.length > 0 && (
                      <div className="data-block cascade-in-1">
                        <div className="block-title core-yellow">Symptoms</div>
                        <ul className="bullet-list yellow-bullets">
                          {results.symptoms.map((s, i) => <li key={i}>{s}</li>)}
                        </ul>
                      </div>
                    )}
                    {results.treatment?.length > 0 && (
                      <div className="data-block cascade-in-2">
                        <div className="block-title core-green">Treatment</div>
                        <ul className="bullet-list green-bullets">
                          {results.treatment.map((t, i) => <li key={i}>{t}</li>)}
                        </ul>
                      </div>
                    )}
                    {results.prevention?.length > 0 && (
                      <div className="data-block cascade-in-3">
                        <div className="block-title core-cyan">Prevention</div>
                        <ul className="bullet-list cyan-bullets">
                          {results.prevention.map((p, i) => <li key={i}>{p}</li>)}
                        </ul>
                      </div>
                    )}
                    {results.precautions?.length > 0 && (
                      <div className="data-block cascade-in-4">
                        <div className="block-title core-purple">Precautions</div>
                        <ul className="bullet-list purple-bullets">
                          {results.precautions.map((p, i) => <li key={i}>{p}</li>)}
                        </ul>
                      </div>
                    )}
                  </>
                )}

                {/* Tab: AI Advisor */}
                {activeTab === 'advisor' && (
                  <div className="advisor-section cascade-in-1">
                    <div className="advisor-intro">
                      <div className="advisor-icon">
                        <svg fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.455 2.456L21.75 6l-1.036.259a3.375 3.375 0 00-2.455 2.456zM16.894 20.567L16.5 21.75l-.394-1.183a2.25 2.25 0 00-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 001.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 001.423 1.423l1.183.394-1.183.394a2.25 2.25 0 00-1.423 1.423z" />
                        </svg>
                      </div>
                      <div>
                        <strong>Personalised AI Advisor</strong>
                        <p>Get tailored treatment advice powered by Llama AI. Optionally describe your growing conditions for more specific guidance.</p>
                      </div>
                    </div>

                    <textarea
                      className="context-input"
                      placeholder="Describe your growing conditions (optional)...&#10;e.g., outdoor garden, humid climate, organic farming, container plants"
                      value={userContext}
                      onChange={(e) => setUserContext(e.target.value)}
                      rows={3}
                    />

                    <button
                      className="advisor-btn"
                      onClick={fetchAdvisory}
                      disabled={isLoadingAdvisory}
                    >
                      {isLoadingAdvisory ? (
                        <><span className="spinner"></span> Generating Advice...</>
                      ) : (
                        <>Get Personalised Advice <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" /></svg></>
                      )}
                    </button>

                    {isLoadingAdvisory && (
                      <div className="advisory-skeleton">
                        <div className="skeleton-line w80"></div>
                        <div className="skeleton-line w60"></div>
                        <div className="skeleton-line w90"></div>
                        <div className="skeleton-line w70"></div>
                        <div className="skeleton-line w85"></div>
                      </div>
                    )}

                    {advisory && (
                      <div className="advisory-content fade-up">
                        <div className="advisory-badge">
                          <svg width="12" height="12" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" /></svg>
                          Llama AI · Personalised
                        </div>
                        {renderMarkdown(advisory)}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ) : (
              <div className="empty-state">
                <div className="empty-icon"></div>
                <p>Upload a tomato leaf image and run analysis to generate a clinical report.</p>
              </div>
            )}
          </div>
        </aside>
      </main>

      {/* History Strip */}
      {history.length > 0 && (
        <div className="history-section fade-in">
          <div className="history-header">
            <span className="history-title">Recent Analyses</span>
            <button className="history-clear" onClick={clearHistory}>Clear All</button>
          </div>
          <div className="history-scroll">
            {history.map((item) => (
              <div key={item.id} className="history-item" title={`${item.prediction} — ${item.confidence}%`}>
                <img src={item.thumbnail} alt={item.prediction} />
                <div className="history-item-info">
                  <div className="history-item-label">{item.prediction}</div>
                  <div className="history-item-conf">{item.confidence}% · {item.mode}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
