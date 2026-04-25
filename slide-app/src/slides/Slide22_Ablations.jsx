import SlideLayout from '../components/SlideLayout';
import { InsightBox } from '../components/Shared';
import { theme } from '../theme';
import { dcceAblations } from '../data';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, LabelList, Legend } from 'recharts';

const TASK_COLORS = {
  unit: theme.unitA,
  type: '#2ca02c',
  indivID: '#d62728',
};

const AblationTable = () => {
  const variants = dcceAblations;
  const headers = ['Variant', 'Unit F1', 'Type F1', 'IndivID F1'];
  return (
    <div style={{ background: theme.white, borderRadius: 8, overflow: 'hidden', boxShadow: '0 1px 4px rgba(0,0,0,0.06)' }}>
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr repeat(3, 0.8fr)', fontSize: 11 }}>
        {headers.map((h, i) => (
          <div key={h} style={{ padding: '7px 10px', fontWeight: 700, background: theme.text, color: theme.white, textAlign: i > 0 ? 'center' : 'left' }}>
            {h}
          </div>
        ))}
        {variants.map((v, ri) => {
          const isFull = v.variant === 'full';
          const bg = isFull ? '#e8f5e920' : ri % 2 ? theme.bgLight : 'transparent';
          const fw = isFull ? 700 : 500;
          return (
            <>
              <div key={`n-${ri}`} style={{ padding: '6px 10px', background: bg, fontWeight: fw, fontFamily: 'monospace', fontSize: 10, borderLeft: isFull ? '3px solid #2ca02c' : 'none' }}>
                {v.variant}
              </div>
              <div key={`u-${ri}`} style={{ padding: '6px 10px', background: bg, fontWeight: fw, textAlign: 'center', fontFamily: 'monospace', fontSize: 10, color: TASK_COLORS.unit }}>
                {v.unit.toFixed(3)}
              </div>
              <div key={`t-${ri}`} style={{ padding: '6px 10px', background: bg, fontWeight: fw, textAlign: 'center', fontFamily: 'monospace', fontSize: 10, color: TASK_COLORS.type }}>
                {v.type.toFixed(3)}
              </div>
              <div key={`i-${ri}`} style={{ padding: '6px 10px', background: bg, fontWeight: fw, textAlign: 'center', fontFamily: 'monospace', fontSize: 10, color: TASK_COLORS.indivID }}>
                {v.indivID.toFixed(3)}
              </div>
            </>
          );
        })}
      </div>
    </div>
  );
};

const Slide22_Ablations = () => {
  const chartData = dcceAblations.map((d) => ({
    ...d,
    unitPct: Math.round(d.unit * 100),
    typePct: Math.round(d.type * 100),
    idPct: Math.round(d.indivID * 100),
  }));

  return (
    <SlideLayout number="29" title="DCCE Ablations" subtitle="What does each component contribute?">
      <div style={{ display: 'flex', gap: 20, flex: 1 }}>
        {/* Chart */}
        <div style={{ flex: 1.3 }}>
          <div style={{ height: 300 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
                <XAxis dataKey="variant" tick={{ fontSize: 10, fontFamily: 'monospace' }} />
                <YAxis domain={[0, 100]} tick={{ fontSize: 9 }} label={{ value: 'Macro-F1 (%)', angle: -90, position: 'insideLeft', offset: 10, fontSize: 10 }} />
                <Tooltip formatter={(v) => `${v}%`} />
                <Legend wrapperStyle={{ fontSize: 10 }} />
                <Bar dataKey="unitPct" name="Social Unit" fill={TASK_COLORS.unit} radius={[3, 3, 0, 0]}>
                  <LabelList dataKey="unitPct" position="top" style={{ fontSize: 8, fontWeight: 600, fill: TASK_COLORS.unit }} />
                </Bar>
                <Bar dataKey="typePct" name="Coda Type" fill={TASK_COLORS.type} radius={[3, 3, 0, 0]}>
                  <LabelList dataKey="typePct" position="top" style={{ fontSize: 8, fontWeight: 600, fill: TASK_COLORS.type }} />
                </Bar>
                <Bar dataKey="idPct" name="Individual ID" fill={TASK_COLORS.indivID} radius={[3, 3, 0, 0]}>
                  <LabelList dataKey="idPct" position="top" style={{ fontSize: 8, fontWeight: 600, fill: TASK_COLORS.indivID }} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Table + insights */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 12 }}>
          <AblationTable />

          <div style={{ display: 'flex', gap: 8 }}>
            <div style={{
              background: '#e8f5e9',
              borderRadius: 8,
              padding: '8px 12px',
              fontSize: 10,
              lineHeight: 1.5,
              flex: 1,
              border: '1.5px solid #a5d6a7',
            }}>
              <strong>Cross-channel effect</strong><br />
              <span style={{ fontFamily: 'monospace' }}>full</span> vs <span style={{ fontFamily: 'monospace' }}>late_fusion</span>:<br />
              Unit: <strong style={{ color: TASK_COLORS.unit }}>+0.222</strong><br />
              IndivID: <strong style={{ color: TASK_COLORS.indivID }}>+0.009</strong>
            </div>
            <div style={{
              background: '#fff3e0',
              borderRadius: 8,
              padding: '8px 12px',
              fontSize: 10,
              lineHeight: 1.5,
              flex: 1,
              border: '1.5px solid #ffcc80',
            }}>
              <strong>Channel specialization</strong><br />
              Rhythm → type (0.878)<br />
              Spectral → ID (0.787)<br />
              Confirms biology!
            </div>
          </div>
        </div>
      </div>

      <InsightBox>
        <strong>Full model wins on every task except coda type</strong> (where raw ICI = 0.931 and no neural model comes close).
        The cross-channel pairing contributes +0.222 on unit F1 — the largest single ablation effect.
      </InsightBox>
    </SlideLayout>
  );
};

export default Slide22_Ablations;
