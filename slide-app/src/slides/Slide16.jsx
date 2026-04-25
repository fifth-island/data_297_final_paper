import SlideLayout from '../components/SlideLayout';
import { InsightBox } from '../components/Shared';
import { theme, UNIT_COLORS } from '../theme';
import { yearByUnitChart } from '../data';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, Cell, LabelList } from 'recharts';

const CramerScale = () => {
  const levels = [
    { label: '< 0.1', desc: 'Negligible', pct: 16.7 },
    { label: '0.1–0.3', desc: 'Small', pct: 33.3 },
    { label: '0.3–0.5', desc: 'Moderate', pct: 33.3 },
    { label: '> 0.5', desc: 'Strong', pct: 16.7 },
  ];
  return (
    <div style={{ background: theme.white, borderRadius: 8, padding: '14px 16px', boxShadow: '0 1px 3px rgba(0,0,0,0.06)' }}>
      <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1, color: theme.textSecondary, marginBottom: 8 }}>
        Cramér's V (Unit × Year)
      </div>
      <div style={{ fontFamily: theme.fontHeader, fontSize: 36, fontWeight: 700, textAlign: 'center', marginBottom: 8 }}>
        0.51
      </div>
      <div style={{ display: 'flex', height: 12, borderRadius: 6, overflow: 'hidden', marginBottom: 8 }}>
        <div style={{ flex: 1, background: '#c8e6c9' }} />
        <div style={{ flex: 2, background: '#fff9c4' }} />
        <div style={{ flex: 2, background: '#ffcc80' }} />
        <div style={{ flex: 1, background: '#ef9a9a' }} />
      </div>
      <div style={{ display: 'flex', fontSize: 9, color: theme.textSecondary }}>
        <span style={{ flex: 1 }}>Negligible</span>
        <span style={{ flex: 2, textAlign: 'center' }}>Small</span>
        <span style={{ flex: 2, textAlign: 'center' }}>Moderate</span>
        <span style={{ flex: 1, textAlign: 'right', fontWeight: 700, color: theme.red }}>← Here</span>
      </div>
    </div>
  );
};

const Slide16 = () => (
  <SlideLayout number="18" title="When Were They Recorded?" subtitle="Units A, D, and F were not recorded at the same time">
    <div style={{ display: 'flex', gap: 20, flex: 1 }}>
      {/* Timeline chart */}
      <div style={{ flex: 2 }}>
        <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1, color: theme.textSecondary, marginBottom: 8 }}>
          Recording Distribution by Year
        </div>
        <div style={{ height: 308 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={yearByUnitChart} margin={{ top: 18, right: 10, bottom: 5, left: 10 }}>
              <XAxis dataKey="year" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="A" fill={UNIT_COLORS.A} name="Unit A">
                <LabelList dataKey="A" position="top" style={{ fontSize: 9, fontWeight: 600, fill: UNIT_COLORS.A }} formatter={(v) => v > 0 ? v : ''} />
              </Bar>
              <Bar dataKey="D" fill={UNIT_COLORS.D} name="Unit D">
                <LabelList dataKey="D" position="top" style={{ fontSize: 9, fontWeight: 600, fill: UNIT_COLORS.D }} formatter={(v) => v > 0 ? v : ''} />
              </Bar>
              <Bar dataKey="F" fill={UNIT_COLORS.F} name="Unit F">
                <LabelList dataKey="F" position="top" style={{ fontSize: 9, fontWeight: 600, fill: '#9e9a2e' }} formatter={(v) => v > 0 ? v : ''} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* What this means table */}
        <div style={{ marginTop: 12, fontSize: 11 }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 0, borderTop: `1.5px solid ${theme.text}` }}>
            <div style={{ padding: '6px 8px', fontWeight: 700, background: theme.text, color: theme.white }}>If model learns…</div>
            <div style={{ padding: '6px 8px', fontWeight: 700, background: theme.text, color: theme.white }}>It might actually be learning…</div>
            {[
              ['Unit A features', '2005 recording conditions'],
              ['Unit D features', '2010 recording conditions'],
              ['Unit F features', 'Mixed year equipment drift'],
            ].map(([left, right], i) => (
              <>
                <div key={`l-${i}`} style={{ padding: '5px 8px', background: i % 2 ? theme.bgLight : 'transparent' }}>{left}</div>
                <div key={`r-${i}`} style={{ padding: '5px 8px', background: i % 2 ? theme.bgLight : 'transparent', fontStyle: 'italic', color: theme.red }}>{right}</div>
              </>
            ))}
          </div>
        </div>
      </div>

      {/* Right panel */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 14 }}>
        <CramerScale />

        <InsightBox variant="red">
          <strong>⚠ Warning:</strong> WhAM's unit F1 = 0.895 at layer 19. Year F1 at the same layer = 0.875.
          The model cannot reliably tell whether it learned whale voices or microphone calibration.
        </InsightBox>

        <InsightBox variant="green">
          <strong>✓ DCCE advantage:</strong> The rhythm encoder uses pre-computed ICI values from the field database —
          not extracted from audio. ICI is invariant to recording equipment drift.
        </InsightBox>
      </div>
    </div>
  </SlideLayout>
);

export default Slide16;
