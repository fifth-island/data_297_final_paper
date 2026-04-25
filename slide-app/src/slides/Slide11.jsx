import SlideLayout from '../components/SlideLayout';
import { InsightBox } from '../components/Shared';
import { theme, UNIT_COLORS } from '../theme';
import { confusionMatrixAlwaysF } from '../data';

const WaffleChart = () => {
  const grid = [];
  const counts = { F: 59, D: 22, A: 18 };
  let idx = 0;
  for (const [unit, count] of Object.entries(counts)) {
    for (let i = 0; i < count; i++) {
      grid.push({ unit, idx: idx++ });
    }
  }
  // fill to 100
  while (grid.length < 100) grid.push({ unit: null, idx: grid.length });

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(10, 1fr)', gap: 3, width: 220 }}>
      {grid.map((cell) => (
        <div key={cell.idx} style={{
          width: 18,
          height: 18,
          borderRadius: 3,
          background: cell.unit ? UNIT_COLORS[cell.unit] : '#e0ddd8',
          opacity: cell.unit ? 0.85 : 0.3,
        }} />
      ))}
    </div>
  );
};

const ConfusionMatrix = () => {
  const { values } = confusionMatrixAlwaysF;
  const labels = ['A', 'D', 'F'];
  const maxVal = 821;
  return (
    <div>
      <div style={{ display: 'grid', gridTemplateColumns: '50px repeat(3, 56px)', gap: 0, fontSize: 11 }}>
        <div />
        {labels.map((l) => (
          <div key={l} style={{ textAlign: 'center', fontWeight: 700, padding: 4, fontSize: 10 }}>Pred {l}</div>
        ))}
        {values.map((row, i) => (
          <>
            <div key={`label-${i}`} style={{ fontWeight: 700, padding: '4px 6px', fontSize: 10, display: 'flex', alignItems: 'center' }}>
              Act {labels[i]}
            </div>
            {row.map((val, j) => (
              <div key={`${i}-${j}`} style={{
                background: val > 0 ? `rgba(76, 114, 176, ${val / maxVal})` : '#f5f3f0',
                textAlign: 'center',
                padding: '8px 4px',
                fontWeight: 700,
                fontSize: 13,
                color: val > 200 ? '#fff' : theme.text,
                border: '1px solid rgba(0,0,0,0.05)',
              }}>
                {val}
              </div>
            ))}
          </>
        ))}
      </div>
      <div style={{ marginTop: 8, fontSize: 10, color: theme.textSecondary }}>
        Per-class F1: A = <b>0.00</b> · D = <b>0.00</b> · F = <b>0.745</b>
      </div>
    </div>
  );
};

const Slide11 = () => (
  <SlideLayout number="13" title="The Imbalance Trap" subtitle={'A model that always says "Unit F" looks 59% accurate — but knows nothing'}>
    <div style={{ display: 'flex', gap: 32, flex: 1 }}>
      {/* Left: The Trap */}
      <div style={{ flex: 1 }}>
        <div style={{
          background: '#fce4ec',
          borderRadius: 8,
          padding: '10px 14px',
          marginBottom: 14,
        }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: theme.red, textTransform: 'uppercase', letterSpacing: 1 }}>
            The Naive Metric
          </div>
        </div>
        <WaffleChart />
        <div style={{ marginTop: 12, fontSize: 10, color: theme.textSecondary }}>Each square ≈ 1% of dataset</div>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginTop: 14 }}>
          <span style={{ fontFamily: theme.fontHeader, fontSize: 36, fontWeight: 700 }}>59.4%</span>
          <span style={{ fontSize: 12, color: theme.textSecondary }}>Accuracy of "always predict Unit F"</span>
        </div>
        <div style={{ fontSize: 13, color: theme.red, fontWeight: 600, marginTop: 4 }}>
          ✕ Looks reasonable. Completely useless.
        </div>
      </div>

      {/* Right: The Fix */}
      <div style={{ flex: 1 }}>
        <div style={{
          background: '#e8f5e9',
          borderRadius: 8,
          padding: '10px 14px',
          marginBottom: 14,
        }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: theme.green, textTransform: 'uppercase', letterSpacing: 1 }}>
            The Real Metric
          </div>
        </div>
        <ConfusionMatrix />
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginTop: 14 }}>
          <span style={{ fontFamily: theme.fontHeader, fontSize: 36, fontWeight: 700, color: theme.green }}>0.248</span>
          <span style={{ fontSize: 12, color: theme.textSecondary }}>Macro-F1 of same model</span>
        </div>
        <div style={{ fontSize: 13, color: theme.green, fontWeight: 600, marginTop: 4 }}>
          ✓ Penalizes ignoring minority classes equally.
        </div>
      </div>
    </div>

    {/* Bottom banner */}
    <InsightBox>
      <strong>Three consequences for every experiment:</strong>{' '}
      ① Primary metric: Macro-F1, not accuracy{' · '}
      ② Training: WeightedRandomSampler — each batch ~equal across units{' · '}
      ③ Probe: class_weight="balanced" — minority errors penalized more
    </InsightBox>
  </SlideLayout>
);

export default Slide11;
