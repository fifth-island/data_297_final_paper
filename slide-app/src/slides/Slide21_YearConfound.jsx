import SlideLayout from '../components/SlideLayout';
import { InsightBox, KPIBox } from '../components/Shared';
import { theme, UNIT_COLORS } from '../theme';
import { yearByUnitChart } from '../data';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, LabelList } from 'recharts';

const ConfoundTable = () => {
  const rows = [
    { stat: "Cramér's V (unit × year)", value: '0.51', interp: 'Strong association', color: theme.red },
    { stat: 'Spearman ρ', value: '0.63', interp: 'Strong monotone correlation', color: theme.red },
    { stat: 'p-value', value: '0.003', interp: 'Highly significant', color: theme.red },
    { stat: 'WhAM Year F1 (L18)', value: '0.906', interp: 'Highest of all probes', color: '#d62728' },
    { stat: 'WhAM Unit F1 (L19)', value: '0.895', interp: 'Tracks year almost perfectly', color: theme.unitA },
  ];
  return (
    <div style={{ background: theme.white, borderRadius: 8, overflow: 'hidden', boxShadow: '0 1px 4px rgba(0,0,0,0.06)' }}>
      <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 0.6fr 1.2fr', fontSize: 10 }}>
        <div style={{ padding: '6px 10px', fontWeight: 700, background: theme.text, color: theme.white }}>Statistic</div>
        <div style={{ padding: '6px 10px', fontWeight: 700, background: theme.text, color: theme.white, textAlign: 'center' }}>Value</div>
        <div style={{ padding: '6px 10px', fontWeight: 700, background: theme.text, color: theme.white }}>Interpretation</div>
        {rows.map((r, i) => (
          <>
            <div key={`s-${i}`} style={{ padding: '5px 10px', background: i % 2 ? theme.bgLight : 'transparent', fontWeight: 500 }}>{r.stat}</div>
            <div key={`v-${i}`} style={{ padding: '5px 10px', background: i % 2 ? theme.bgLight : 'transparent', textAlign: 'center', fontWeight: 700, fontFamily: 'monospace', color: r.color }}>{r.value}</div>
            <div key={`i-${i}`} style={{ padding: '5px 10px', background: i % 2 ? theme.bgLight : 'transparent', fontStyle: 'italic', color: theme.textSecondary }}>{r.interp}</div>
          </>
        ))}
      </div>
    </div>
  );
};

const Slide21_YearConfound = () => (
  <SlideLayout number="26" title="The Year Confound" subtitle="Not reported in the original WhAM paper — a novel finding of this work">
    <div style={{ display: 'flex', gap: 16, flex: 1 }}>
      {/* Left: chart */}
      <div style={{ flex: 1 }}>
        <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1, color: theme.textSecondary, marginBottom: 6 }}>
          Recording Timeline by Unit
        </div>
        <div style={{ height: 200 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={yearByUnitChart} margin={{ top: 12, right: 8, bottom: 0, left: 8 }}>
              <XAxis dataKey="year" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 9 }} />
              <Tooltip />
              <Legend wrapperStyle={{ fontSize: 10 }} />
              <Bar dataKey="A" fill={UNIT_COLORS.A} name="Unit A">
                <LabelList dataKey="A" position="top" style={{ fontSize: 8, fontWeight: 600, fill: UNIT_COLORS.A }} formatter={(v) => v > 0 ? v : ''} />
              </Bar>
              <Bar dataKey="D" fill={UNIT_COLORS.D} name="Unit D">
                <LabelList dataKey="D" position="top" style={{ fontSize: 8, fontWeight: 600, fill: UNIT_COLORS.D }} formatter={(v) => v > 0 ? v : ''} />
              </Bar>
              <Bar dataKey="F" fill={UNIT_COLORS.F} name="Unit F">
                <LabelList dataKey="F" position="top" style={{ fontSize: 8, fontWeight: 600, fill: '#9e9a2e' }} formatter={(v) => v > 0 ? v : ''} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={{ marginTop: 10 }}>
          <ConfoundTable />
        </div>
      </div>

      {/* Right: implications */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 12 }}>
        <div style={{
          background: '#fce4ec',
          borderRadius: 10,
          padding: '14px 18px',
          border: '2px solid #ef9a9a',
        }}>
          <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 8 }}>⚠ The Problem</div>
          <div style={{ fontSize: 11, lineHeight: 1.6 }}>
            Units A, D, F were recorded at <strong>systematically different times</strong> during 2005–2010.
            A model trained on raw audio may learn <em>recording-year artifacts</em> (equipment calibration, ocean noise, hydrophone positioning) rather than biological unit identity.
          </div>
        </div>

        <div style={{
          background: '#e8f5e9',
          borderRadius: 10,
          padding: '14px 18px',
          border: '2px solid #a5d6a7',
        }}>
          <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 8 }}>✓ DCCE's Immunity</div>
          <div style={{ fontSize: 11, lineHeight: 1.6 }}>
            <strong>Rhythm encoder:</strong> Uses pre-computed ICI values from field databases — timing ratios don't change with equipment.<br /><br />
            <strong>Spectral encoder:</strong> Processes per-coda mel spectrograms — captures click-level acoustic structure, not accumulated waveform drift.
          </div>
        </div>

        <InsightBox>
          <strong>Implication:</strong> WhAM's 0.895 unit F1 should be reported with this caveat.
          DCCE's 0.878 unit F1 is more biologically grounded — it can't exploit year confounds.
        </InsightBox>
      </div>
    </div>
  </SlideLayout>
);

export default Slide21_YearConfound;
