import SlideLayout from '../components/SlideLayout';
import { InsightBox, SectionLabel } from '../components/Shared';
import { theme } from '../theme';
import { baselineComparison } from '../data';

const TASK_COLORS = {
  unit: theme.unitA,
  type: '#2ca02c',
  indivID: '#d62728',
};

const FBar = ({ value, max, color }) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
    <div style={{
      height: 14,
      width: `${(value / max) * 100}%`,
      background: color,
      borderRadius: '0 3px 3px 0',
      minWidth: 2,
    }} />
    <span style={{ fontFamily: 'monospace', fontSize: 10, fontWeight: 700, color }}>{value.toFixed(3)}</span>
  </div>
);

const Slide28_Baselines = () => {
  const dcce = { model: 'DCCE-full', unit: 0.878, type: 0.578, indivID: 0.834 };
  const allModels = [...baselineComparison, dcce];

  return (
    <SlideLayout number="28" title="Baselines Comparison" subtitle="Four representations × three tasks — where does each method shine?">
      <div style={{ display: 'flex', gap: 16, flex: 1 }}>
        {/* Left: table with bars */}
        <div style={{ flex: 1.4 }}>
          <div style={{
            background: theme.white,
            borderRadius: 10,
            overflow: 'hidden',
            boxShadow: '0 2px 6px rgba(0,0,0,0.06)',
          }}>
            {/* Header */}
            <div style={{ display: 'grid', gridTemplateColumns: '110px 1fr 1fr 1fr', background: theme.text, color: theme.white, fontSize: 10, fontWeight: 700 }}>
              <div style={{ padding: '8px 12px' }}>Model</div>
              <div style={{ padding: '8px 12px', color: TASK_COLORS.unit }}>Social Unit</div>
              <div style={{ padding: '8px 12px', color: TASK_COLORS.type }}>Coda Type</div>
              <div style={{ padding: '8px 12px', color: TASK_COLORS.indivID }}>Individual ID</div>
            </div>
            {/* Rows */}
            {allModels.map((m, i) => {
              const isDCCE = m.model === 'DCCE-full';
              return (
                <div key={i} style={{
                  display: 'grid',
                  gridTemplateColumns: '110px 1fr 1fr 1fr',
                  fontSize: 10,
                  background: isDCCE ? '#e8f5e915' : i % 2 ? theme.bgLight : 'transparent',
                  borderLeft: isDCCE ? '3px solid #2ca02c' : '3px solid transparent',
                }}>
                  <div style={{ padding: '6px 10px', fontWeight: isDCCE ? 700 : 500, fontFamily: 'monospace' }}>{m.model}</div>
                  <div style={{ padding: '6px 10px' }}><FBar value={m.unit} max={1} color={TASK_COLORS.unit} /></div>
                  <div style={{ padding: '6px 10px' }}><FBar value={m.type} max={1} color={TASK_COLORS.type} /></div>
                  <div style={{ padding: '6px 10px' }}><FBar value={m.indivID} max={1} color={TASK_COLORS.indivID} /></div>
                </div>
              );
            })}
          </div>

          <div style={{ marginTop: 12 }}>
            <img
              src="/figures/baseline_comparison.png"
              alt="Baseline comparison chart"
              style={{ width: '100%', borderRadius: 8, boxShadow: '0 1px 4px rgba(0,0,0,0.06)' }}
            />
          </div>
        </div>

        {/* Right: interpretation cards */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 10 }}>
          <SectionLabel>Key patterns</SectionLabel>

          <div style={{
            background: '#e8f5e9',
            borderRadius: 8,
            padding: '10px 14px',
            border: '1.5px solid #a5d6a7',
          }}>
            <div style={{ fontSize: 11, fontWeight: 700, marginBottom: 4 }}>🏆 ICI excels at coda type</div>
            <div style={{ fontSize: 10, color: theme.textSecondary, lineHeight: 1.5 }}>
              Raw ICI achieves <strong>0.931</strong> F1 on type — no neural model beats a simple rhythm vector for classifying coda patterns.
            </div>
          </div>

          <div style={{
            background: '#fce4ec',
            borderRadius: 8,
            padding: '10px 14px',
            border: '1.5px solid #ef9a9a',
          }}>
            <div style={{ fontSize: 11, fontWeight: 700, marginBottom: 4 }}>⚠ WhAM fails at individual ID</div>
            <div style={{ fontSize: 10, color: theme.textSecondary, lineHeight: 1.5 }}>
              Despite 30M parameters and 10K codas, WhAM only reaches <strong>0.454</strong> on individual ID — barely above Raw ICI (0.493).
            </div>
          </div>

          <div style={{
            background: '#8172B215',
            borderRadius: 8,
            padding: '10px 14px',
            border: '1.5px solid #8172B240',
          }}>
            <div style={{ fontSize: 11, fontWeight: 700, marginBottom: 4 }}>💡 Mel → Identity signal</div>
            <div style={{ fontSize: 10, color: theme.textSecondary, lineHeight: 1.5 }}>
              Raw mel spectrograms get 0.740 unit F1 and 0.272 indivID — spectral data has identity info, but needs better encoding.
            </div>
          </div>

          <div style={{
            background: theme.white,
            borderRadius: 8,
            padding: '10px 14px',
            boxShadow: '0 1px 4px rgba(0,0,0,0.06)',
          }}>
            <div style={{ fontSize: 11, fontWeight: 700, marginBottom: 4, color: '#2ca02c' }}>→ Motivation for DCCE</div>
            <div style={{ fontSize: 10, color: theme.textSecondary, lineHeight: 1.5 }}>
              Combine ICI's type strength with mel's identity potential, using contrastive learning to align them. The best of both channels.
            </div>
          </div>
        </div>
      </div>
    </SlideLayout>
  );
};

export default Slide28_Baselines;
