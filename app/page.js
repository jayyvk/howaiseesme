'use client';
import { useState, useEffect, useRef, useCallback, useMemo } from 'react';

const STATES = { LOADING: 'loading', CAMERA: 'camera', LIVE: 'live', ERROR: 'error' };
const COLS = 32;
const ROWS = 16;
const TOTAL = 512;

export default function Home() {
  const [state, setState] = useState(STATES.LOADING);
  const [loadStage, setLoadStage] = useState('Initializing...');
  const [loadPct, setLoadPct] = useState(0);
  const [embedding, setEmbedding] = useState(() => new Array(TOTAL).fill(0));
  const [personMask, setPersonMask] = useState(() => new Float32Array(TOTAL));
  const [tick, setTick] = useState(0);
  const [inferMs, setInferMs] = useState(0);
  const [errorMsg, setErrorMsg] = useState('');
  const [camMinimized, setCamMinimized] = useState(false);
  const [segMode, setSegMode] = useState('brightness');

  const [queryText, setQueryText] = useState('');
  const [queryLoading, setQueryLoading] = useState(false);
  // Each item: { text, scaledScore, rawScore }
  const [queries, setQueries] = useState([]);

  const workerRef = useRef(null);
  const videoRef = useRef(null);
  const previewRef = useRef(null);
  const canvasRef = useRef(null);
  const silCanvasRef = useRef(null);
  const intervalRef = useRef(null);
  const silIntervalRef = useRef(null);
  const inferStartRef = useRef(0);
  const maskRef = useRef(new Float32Array(TOTAL));
  const segRef = useRef(null);
  const useMLSegRef = useRef(false);
  const queriesRef = useRef([]);

  // Keep ref in sync
  useEffect(() => { queriesRef.current = queries; }, [queries]);

  // MediaPipe (optional)
  useEffect(() => {
    const s = document.createElement('script');
    s.src = 'https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation@0.1/selfie_segmentation.js';
    s.crossOrigin = 'anonymous';
    s.onload = () => {
      try {
        if (window.SelfieSegmentation) {
          const seg = new window.SelfieSegmentation({
            locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation@0.1/${f}`
          });
          seg.setOptions({ modelSelection: 1 });
          seg.onResults((results) => {
            if (!results.segmentationMask) return;
            const c = document.createElement('canvas');
            c.width = COLS; c.height = ROWS;
            const ctx = c.getContext('2d', { willReadFrequently: true });
            ctx.save(); ctx.translate(COLS, 0); ctx.scale(-1, 1);
            ctx.drawImage(results.segmentationMask, 0, 0, COLS, ROWS);
            ctx.restore();
            const d = ctx.getImageData(0, 0, COLS, ROWS).data;
            const m = new Float32Array(TOTAL);
            for (let i = 0; i < TOTAL; i++) m[i] = d[i * 4] / 255;
            maskRef.current = m;
            setPersonMask(new Float32Array(m));
          });
          segRef.current = seg;
          useMLSegRef.current = true;
          setSegMode('mediapipe');
        }
      } catch (e) {}
    };
    document.head.appendChild(s);
  }, []);

  const sampleBrightness = useCallback(() => {
    const vid = videoRef.current;
    const c = silCanvasRef.current;
    if (!vid || !c || vid.readyState < 2) return;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    c.width = COLS; c.height = ROWS;
    ctx.save(); ctx.translate(COLS, 0); ctx.scale(-1, 1);
    ctx.drawImage(vid, 0, 0, COLS, ROWS);
    ctx.restore();
    const d = ctx.getImageData(0, 0, COLS, ROWS).data;
    const bright = [];
    for (let i = 0; i < TOTAL; i++) bright.push((d[i*4] + d[i*4+1] + d[i*4+2]) / 765);
    const sorted = [...bright].sort((a,b) => a - b);
    const q30 = sorted[Math.floor(TOTAL * 0.3)];
    const q70 = sorted[Math.floor(TOTAL * 0.7)];
    const spread = q70 - q30;
    const threshold = q30 + spread * 0.6;
    const m = new Float32Array(TOTAL);
    for (let i = 0; i < TOTAL; i++) {
      if (bright[i] < threshold) {
        m[i] = 0.4 + Math.min(1, (threshold - bright[i]) / (spread + 0.08)) * 0.6;
      }
    }
    if (!useMLSegRef.current) {
      maskRef.current = m;
      setPersonMask(new Float32Array(m));
    }
  }, []);

  const runMLSeg = useCallback(async () => {
    if (!segRef.current || !videoRef.current || videoRef.current.readyState < 2) return;
    try { await segRef.current.send({ image: videoRef.current }); } catch(e) {}
  }, []);

  // Worker
  useEffect(() => {
    const w = new Worker('/worker.js', { type: 'module' });
    workerRef.current = w;
    w.onmessage = (e) => {
      const { type, data } = e.data;
      if (type === 'progress') { setLoadStage(data.stage); setLoadPct(data.pct); }
      if (type === 'ready') startCamera();
      if (type === 'result') {
        setInferMs(Math.round(performance.now() - inferStartRef.current));
        setEmbedding(data.embedding);
        setTick(t => t + 1);
      }
      if (type === 'text_result') {
        setQueryLoading(false);
        setQueries(prev => {
          const exists = prev.find(q => q.text === data.text);
          if (exists) return prev.map(q => q.text === data.text ? { ...q, scaledScore: data.scaledScore, rawScore: data.rawScore } : q);
          return [{ text: data.text, scaledScore: data.scaledScore, rawScore: data.rawScore }, ...prev].slice(0, 12);
        });
      }
      if (type === 'batch_result') {
        setQueries(prev => {
          const map = {};
          data.results.forEach(r => { map[r.text] = r; });
          return prev.map(q => map[q.text] ? { ...q, scaledScore: map[q.text].scaledScore, rawScore: map[q.text].rawScore } : q);
        });
      }
      if (type === 'error') { setErrorMsg(data); setState(STATES.ERROR); }
    };
    w.postMessage({ type: 'load' });
    return () => { w.terminate(); };
  }, []);

  const startCamera = useCallback(async () => {
    setState(STATES.CAMERA);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } }
      });
      const vid = videoRef.current;
      vid.srcObject = stream;
      vid.onloadeddata = () => {
        vid.play();
        setState(STATES.LIVE);
        startInference();
        silIntervalRef.current = setInterval(() => {
          sampleBrightness();
          if (useMLSegRef.current) runMLSeg();
        }, 150);
      };
    } catch (err) {
      setErrorMsg('Camera access denied. Please allow camera and reload.');
      setState(STATES.ERROR);
    }
  }, [sampleBrightness, runMLSeg]);

  useEffect(() => {
    if (state === STATES.LIVE && previewRef.current && videoRef.current?.srcObject) {
      previewRef.current.srcObject = videoRef.current.srcObject;
      previewRef.current.play().catch(() => {});
    }
  }, [state]);

  const runInference = useCallback(() => {
    const vid = videoRef.current;
    const c = canvasRef.current;
    if (!vid || !c || !workerRef.current || vid.readyState < 2) return;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    c.width = 224; c.height = 224;
    ctx.drawImage(vid, 0, 0, 224, 224);
    const img = ctx.getImageData(0, 0, 224, 224);
    inferStartRef.current = performance.now();
    workerRef.current.postMessage(
      { type: 'infer', data: { imageData: img.data.buffer, width: 224, height: 224 } },
      [img.data.buffer]
    );
  }, []);

  const startInference = useCallback(() => {
    setTimeout(runInference, 600);
    intervalRef.current = setInterval(runInference, 1500); // slightly slower to save CPU
    return () => { clearInterval(intervalRef.current); clearInterval(silIntervalRef.current); };
  }, [runInference]);

  // Batch re-score queries when embedding updates
  useEffect(() => {
    if (tick > 0 && queriesRef.current.length > 0 && workerRef.current) {
      workerRef.current.postMessage({
        type: 'score_batch',
        data: { texts: queriesRef.current.map(q => q.text) }
      });
    }
  }, [tick]);

  const submitQuery = useCallback(() => {
    const text = queryText.trim();
    if (!text || !workerRef.current) return;
    setQueryLoading(true);
    workerRef.current.postMessage({ type: 'encode_text', data: { text, id: Date.now() } });
    setQueryText('');
  }, [queryText]);

  const handleKeyDown = useCallback((e) => {
    if (e.key === 'Enter') submitQuery();
  }, [submitQuery]);

  const removeQuery = useCallback((text) => {
    setQueries(prev => prev.filter(q => q.text !== text));
  }, []);

  // Compute softmax probabilities across all active queries for relative comparison
  const queriesWithProb = useMemo(() => {
    if (queries.length === 0) return [];
    const scores = queries.map(q => q.scaledScore || 0);
    const maxScore = Math.max(...scores);
    const exps = scores.map(s => Math.exp(s - maxScore)); // subtract max for numerical stability
    const sumExp = exps.reduce((a, b) => a + b, 0);
    return queries.map((q, i) => ({
      ...q,
      probability: (exps[i] / sumExp) * 100,
    })).sort((a, b) => b.scaledScore - a.scaledScore);
  }, [queries]);

  // Cell styles
  const cellData = useMemo(() => {
    return embedding.map((val, i) => {
      const p = personMask[i] || 0;
      if (p < 0.3) {
        // Background: faint grey numbers
        return {
          color: `rgba(190,190,190, ${Math.min(0.15 + Math.abs(val) * 2, 0.4)})`,
          bg: 'transparent', weight: 400,
        };
      }
      // Person: uniform blue tint, stronger where mask is more confident
      const alpha = 0.2 + p * 0.45;
      return {
        color: '#fff',
        bg: `rgba(37, 99, 235, ${alpha})`,
        weight: 600,
      };
    });
  }, [embedding, personMask]);

  return (
    <div style={{
      background: '#fff', height: '100vh', width: '100vw',
      fontFamily: "'IBM Plex Mono', monospace",
      position: 'relative', overflow: 'hidden',
    }}>
      <video ref={videoRef} style={{ display: 'none' }} playsInline muted />
      <canvas ref={canvasRef} style={{ display: 'none' }} />
      <canvas ref={silCanvasRef} style={{ display: 'none' }} />

      {/* Top bar */}
      {state === STATES.LIVE && (
        <div style={{
          position: 'fixed', top: 0, left: 0, right: 0, zIndex: 30,
          background: 'rgba(255,255,255,0.94)',
          backdropFilter: 'blur(16px)',
          borderBottom: '1px solid rgba(0,0,0,0.06)',
        }}>
          {/* Input */}
          <div style={{ padding: '10px 20px', display: 'flex', alignItems: 'center', gap: '10px' }}>
            <input
              type="text"
              value={queryText}
              onChange={(e) => setQueryText(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder='Describe what you see — "a happy person", "holding a phone"...'
              style={{
                flex: 1, padding: '9px 14px', fontSize: '13px', fontFamily: 'inherit',
                background: '#f8f9fa', border: '1.5px solid #e5e7eb',
                borderRadius: '8px', outline: 'none', color: '#1f2937',
              }}
              onFocus={(e) => e.target.style.borderColor = '#3b82f6'}
              onBlur={(e) => e.target.style.borderColor = '#e5e7eb'}
            />
            <button
              onClick={submitQuery}
              disabled={queryLoading || !queryText.trim()}
              style={{
                padding: '9px 18px', fontSize: '12px', fontWeight: 600, fontFamily: 'inherit',
                background: queryLoading ? '#e5e7eb' : '#2563eb',
                color: queryLoading ? '#9ca3af' : '#fff',
                border: 'none', borderRadius: '8px', cursor: 'pointer',
                flexShrink: 0, transition: 'background 0.2s',
              }}
            >
              {queryLoading ? '...' : 'Match'}
            </button>
          </div>

          {/* Results: sorted by score, showing both scaled score and softmax % */}
          {queriesWithProb.length > 0 && (
            <div style={{ padding: '0 20px 12px 20px', display: 'flex', flexDirection: 'column', gap: '5px' }}>
              {queriesWithProb.map((q, i) => {
                const isTop = i === 0 && queriesWithProb.length > 1;
                return (
                  <div key={q.text} style={{
                    display: 'flex', alignItems: 'center', gap: '10px',
                    padding: '4px 8px',
                    background: isTop ? 'rgba(37,99,235,0.04)' : 'transparent',
                    borderRadius: '6px',
                    transition: 'background 0.3s',
                  }}>
                    {/* Probability (relative to other queries) */}
                    <div style={{
                      fontSize: '16px', fontWeight: 700,
                      color: isTop ? '#2563eb' : '#94a3b8',
                      minWidth: '52px', textAlign: 'right',
                      fontVariantNumeric: 'tabular-nums',
                    }}>
                      {queriesWithProb.length > 1 ? `${q.probability.toFixed(0)}%` : q.scaledScore.toFixed(1)}
                    </div>

                    {/* Bar + text */}
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{
                        display: 'flex', alignItems: 'baseline', gap: '8px',
                      }}>
                        <span style={{
                          fontSize: '12px',
                          color: isTop ? '#1e3a5f' : '#64748b',
                          fontWeight: isTop ? 600 : 400,
                          whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                        }}>
                          {q.text}
                        </span>
                        <span style={{
                          fontSize: '9px', color: '#c4c9d1',
                          flexShrink: 0,
                        }}>
                          {q.scaledScore.toFixed(1)}
                        </span>
                      </div>
                      <div style={{
                        height: '3px', background: '#f1f5f9', borderRadius: '2px',
                        overflow: 'hidden', marginTop: '3px',
                      }}>
                        <div style={{
                          height: '100%',
                          width: `${Math.min(q.probability, 100)}%`,
                          background: isTop ? '#2563eb' : '#cbd5e1',
                          borderRadius: '2px',
                          transition: 'width 0.5s ease',
                        }} />
                      </div>
                    </div>

                    {/* Remove */}
                    <button
                      onClick={() => removeQuery(q.text)}
                      style={{
                        background: 'none', border: 'none', cursor: 'pointer',
                        color: '#d1d5db', fontSize: '14px', padding: '0 2px',
                        fontFamily: 'inherit', lineHeight: 1, flexShrink: 0,
                      }}
                    >×</button>
                  </div>
                );
              })}

              {queriesWithProb.length === 1 && (
                <div style={{ fontSize: '10px', color: '#b4b9c2', padding: '0 8px' }}>
                  Add more descriptions to compare — scores become relative percentages
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Fullscreen grid */}
      <div style={{
        width: '100%', height: '100%',
        display: 'grid',
        gridTemplateColumns: `repeat(${COLS}, 1fr)`,
        gridTemplateRows: `repeat(${ROWS}, 1fr)`,
        opacity: state === STATES.LIVE ? 1 : 0.04,
        transition: 'opacity 1s ease',
      }}>
        {cellData.map((s, d) => (
          <div key={d} style={{
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 'clamp(5px, 1vw, 10px)',
            color: s.color, background: s.bg, fontWeight: s.weight,
            transition: 'color 0.2s, background 0.2s',
            whiteSpace: 'nowrap', overflow: 'hidden', userSelect: 'none',
            letterSpacing: '-0.3px',
          }}>
            {embedding[d] >= 0 ? '\u00A0' : ''}{embedding[d].toFixed(3)}
          </div>
        ))}
      </div>

      {/* Webcam preview */}
      {state === STATES.LIVE && (
        <div style={{ position: 'fixed', bottom: '16px', right: '16px', zIndex: 20 }}>
          <button
            onClick={() => setCamMinimized(m => !m)}
            style={{
              position: 'absolute', top: '-10px', left: '-10px', zIndex: 21,
              width: '24px', height: '24px', borderRadius: '50%',
              background: '#fff', border: '1px solid #e0e0e0',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              cursor: 'pointer', fontSize: '11px', color: '#6b7280',
              boxShadow: '0 2px 6px rgba(0,0,0,0.1)',
              fontFamily: 'inherit', lineHeight: 1, padding: 0,
            }}
          >{camMinimized ? '◻' : '—'}</button>
          <div style={{
            width: camMinimized ? '48px' : '220px',
            aspectRatio: '4/3', background: '#000',
            borderRadius: camMinimized ? '8px' : '12px',
            overflow: 'hidden', boxShadow: '0 4px 24px rgba(0,0,0,0.2)',
            border: '3px solid #fff', transition: 'width 0.3s ease',
            position: 'relative',
          }}>
            <video ref={previewRef} playsInline muted style={{
              width: '100%', height: '100%',
              objectFit: 'cover', transform: 'scaleX(-1)', display: 'block',
            }} />
            {!camMinimized && (
              <div style={{
                position: 'absolute', top: '6px', left: '8px',
                display: 'flex', alignItems: 'center', gap: '4px',
                background: 'rgba(0,0,0,0.6)', padding: '2px 7px', borderRadius: '4px',
              }}>
                <div style={{
                  width: '5px', height: '5px', borderRadius: '50%',
                  background: '#ef4444', animation: 'pulse 1.5s infinite',
                }} />
                <span style={{ fontSize: '8px', color: '#fff', letterSpacing: '1px' }}>LIVE</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Info */}
      {state === STATES.LIVE && (
        <div style={{
          position: 'fixed', bottom: '16px', left: '20px', zIndex: 10,
          fontSize: '8px', color: '#c4c9d1', letterSpacing: '0.5px', pointerEvents: 'none',
        }}>
          CLIP ViT-B/16 · 512d · {inferMs}ms · f{tick} · {segMode} · 100% local
        </div>
      )}

      {/* Overlays */}
      {state === STATES.LOADING && (
        <div style={{
          position: 'fixed', inset: 0, zIndex: 50,
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
          background: '#fff',
        }}>
          <div style={{
            width: '32px', height: '32px', borderRadius: '50%',
            border: '2.5px solid #e5e7eb', borderTopColor: '#3b82f6',
            animation: 'spin 0.8s linear infinite', marginBottom: '20px',
          }} />
          <div style={{ fontSize: '13px', color: '#374151', fontWeight: 500, marginBottom: '14px' }}>{loadStage}</div>
          <div style={{ width: '240px', height: '3px', background: '#f3f4f6', borderRadius: '2px', overflow: 'hidden' }}>
            <div style={{
              height: '100%', width: `${Math.min(loadPct, 100)}%`,
              background: 'linear-gradient(to right, #93c5fd, #3b82f6)',
              borderRadius: '2px', transition: 'width 0.3s',
            }} />
          </div>
          <div style={{ fontSize: '10px', color: '#9ca3af', marginTop: '10px' }}>{Math.min(Math.round(loadPct), 100)}%</div>
        </div>
      )}

      {state === STATES.CAMERA && (
        <div style={{
          position: 'fixed', inset: 0, zIndex: 50,
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
          background: '#fff',
        }}>
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#9ca3af" strokeWidth="1.5" style={{ marginBottom: '16px' }}>
            <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
            <circle cx="12" cy="13" r="4" />
          </svg>
          <div style={{ fontSize: '14px', color: '#374151', fontWeight: 500, marginBottom: '6px' }}>Allow camera access</div>
          <div style={{ fontSize: '12px', color: '#9ca3af' }}>to see how AI sees you</div>
        </div>
      )}

      {state === STATES.ERROR && (
        <div style={{
          position: 'fixed', inset: 0, zIndex: 50,
          display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#fff',
        }}>
          <div style={{ fontSize: '12px', color: '#dc2626', textAlign: 'center', padding: '0 40px', maxWidth: '600px', whiteSpace: 'pre-wrap' }}>{errorMsg}</div>
        </div>
      )}

      <style>{`
        @keyframes spin { 0%{transform:rotate(0deg)} 100%{transform:rotate(360deg)} }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
      `}</style>
    </div>
  );
}
