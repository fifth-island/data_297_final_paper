import { useState, useEffect, useCallback } from 'react';
import { theme } from './theme';
import Slide09 from './slides/Slide09';
import Slide10 from './slides/Slide10';
import Slide11_Rhythm from './slides/Slide11_Rhythm';
import Slide12_Spectral from './slides/Slide12_Spectral';
import Slide11 from './slides/Slide11';
import Slide12 from './slides/Slide12';
import Slide13 from './slides/Slide13';
import Slide14 from './slides/Slide14';
import Slide15 from './slides/Slide15';
import Slide16 from './slides/Slide16';
import Slide17 from './slides/Slide17';
import Slide18_Architecture from './slides/Slide18_Architecture';
import Slide21_Inputs from './slides/Slide21_Inputs';
import Slide19_Training from './slides/Slide19_Training';
import Slide23_CrossChannel from './slides/Slide23_CrossChannel';
import Slide24_ProbingSetup from './slides/Slide24_ProbingSetup';
import Slide20_WhamProbing from './slides/Slide20_WhamProbing';
import Slide21_YearConfound from './slides/Slide21_YearConfound';
import Slide27_WhamUmaps from './slides/Slide27_WhamUmaps';
import Slide28_Baselines from './slides/Slide28_Baselines';
import Slide22_Ablations from './slides/Slide22_Ablations';
import Slide23_Headline from './slides/Slide23_Headline';
import Slide31_UmapComparison from './slides/Slide31_UmapComparison';
import Slide32_SynthPipeline from './slides/Slide32_SynthPipeline';
import Slide24_Augmentation from './slides/Slide24_Augmentation';
import Slide25_Discussion from './slides/Slide25_Discussion';

const slides = [
  { component: Slide09, label: '09 — The Population' },
  { component: Slide10, label: '10 — Anatomy of a Coda' },
  { component: Slide11_Rhythm, label: '11 — The Rhythm Channel' },
  { component: Slide12_Spectral, label: '12 — The Spectral Channel' },
  { component: Slide11, label: '13 — The Imbalance Trap' },
  { component: Slide12, label: '14 — Reading a Coda Type' },
  { component: Slide13, label: '15 — The Shared Vocabulary' },
  { component: Slide14, label: '16 — The Data Funnel' },
  { component: Slide15, label: '17 — 12 Voices in the Data' },
  { component: Slide16, label: '18 — The Year Problem' },
  { component: Slide17, label: '19 — From Data to Design' },
  { component: Slide18_Architecture, label: '20 — DCCE Architecture' },
  { component: Slide21_Inputs, label: '21 — Input Representations' },
  { component: Slide19_Training, label: '22 — Training Objective' },
  { component: Slide23_CrossChannel, label: '23 — Cross-Channel Pairing' },
  { component: Slide24_ProbingSetup, label: '24 — Probing WhAM' },
  { component: Slide20_WhamProbing, label: '25 — Inside WhAM\'s Brain' },
  { component: Slide21_YearConfound, label: '26 — The Year Confound' },
  { component: Slide27_WhamUmaps, label: '27 — WhAM Embedding Space' },
  { component: Slide28_Baselines, label: '28 — Baselines Comparison' },
  { component: Slide22_Ablations, label: '29 — DCCE Ablations' },
  { component: Slide23_Headline, label: '30 — The Headline' },
  { component: Slide31_UmapComparison, label: '31 — WhAM vs DCCE UMAPs' },
  { component: Slide32_SynthPipeline, label: '32 — Synthetic Generation' },
  { component: Slide24_Augmentation, label: '33 — Augmentation Results' },
  { component: Slide25_Discussion, label: '34 — Discussion & Takeaways' },
];

function App() {
  const [current, setCurrent] = useState(0);
  const [scale, setScale] = useState(1);

  const next = useCallback(() => setCurrent((c) => Math.min(c + 1, slides.length - 1)), []);
  const prev = useCallback(() => setCurrent((c) => Math.max(c - 1, 0)), []);

  useEffect(() => {
    const handler = (e) => {
      if (e.key === 'ArrowRight' || e.key === ' ') { e.preventDefault(); next(); }
      if (e.key === 'ArrowLeft') { e.preventDefault(); prev(); }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [next, prev]);

  useEffect(() => {
    const resize = () => {
      const sx = window.innerWidth / 1320;
      const sy = window.innerHeight / 800;
      setScale(Math.min(sx, sy, 1));
    };
    resize();
    window.addEventListener('resize', resize);
    return () => window.removeEventListener('resize', resize);
  }, []);

  const SlideComponent = slides[current].component;

  return (
    <div style={{
      minHeight: '100vh',
      background: '#2a2a2a',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      fontFamily: theme.fontBody,
    }}>
      <div style={{
        transform: `scale(${scale})`,
        transformOrigin: 'center center',
      }}>
      <div style={{
        width: '1280px',
        height: '720px',
        borderRadius: 4,
        overflow: 'hidden',
        boxShadow: '0 8px 40px rgba(0,0,0,0.4)',
      }}>
        <SlideComponent />
      </div>
      </div>

      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 16,
        marginTop: 20,
        color: '#999',
        fontSize: 13,
      }}>
        <button
          onClick={prev}
          disabled={current === 0}
          style={{
            background: 'none',
            border: '1px solid #555',
            color: current === 0 ? '#555' : '#ccc',
            padding: '6px 14px',
            borderRadius: 4,
            cursor: current === 0 ? 'default' : 'pointer',
            fontFamily: theme.fontBody,
            fontSize: 12,
          }}
        >
          &#8592; Prev
        </button>

        <div style={{ display: 'flex', gap: 6 }}>
          {slides.map((_, i) => (
            <div
              key={i}
              onClick={() => setCurrent(i)}
              style={{
                width: i === current ? 20 : 8,
                height: 8,
                borderRadius: 4,
                background: i === current ? theme.periwinkle : '#555',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
              }}
            />
          ))}
        </div>

        <button
          onClick={next}
          disabled={current === slides.length - 1}
          style={{
            background: 'none',
            border: '1px solid #555',
            color: current === slides.length - 1 ? '#555' : '#ccc',
            padding: '6px 14px',
            borderRadius: 4,
            cursor: current === slides.length - 1 ? 'default' : 'pointer',
            fontFamily: theme.fontBody,
            fontSize: 12,
          }}
        >
          Next &#8594;
        </button>

        <span style={{ fontSize: 11, color: '#666', marginLeft: 8 }}>
          {slides[current].label}
        </span>
      </div>
    </div>
  );
}

export default App;
