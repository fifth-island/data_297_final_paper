import SlideLayout from '../components/SlideLayout';
import { KPIBox, InsightBox } from '../components/Shared';
import { theme } from '../theme';
import { baselineComparison } from '../data';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, LabelList, ReferenceLine } from 'recharts';

const Slide23_Headline = () => {
  const idData = [
    ...baselineComparison.map((b) => ({ model: b.model, f1: b.indivID, pct: Math.round(b.indivID * 100) })),
    { model: 'DCCE-full', f1: 0.834, pct: 83 },
  ];

  const unitData = [
    ...baselineComparison.map((b) => ({ model: b.model, f1: b.unit, pct: Math.round(b.unit * 100) })),
    { model: 'DCCE-full', f1: 0.878, pct: 88 },
  ];

  return (
    <SlideLayout number="30" title="The Headline" subtitle="Domain knowledge beats scale on individual identity">
      <div style={{ display: 'flex', gap: 20, flex: 1 }}>
        {/* Individual ID chart */}
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1, color: '#d62728', marginBottom: 6 }}>
            Individual ID — Macro-F1
          </div>
          <div style={{ height: 280 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={idData} layout="vertical" margin={{ top: 0, right: 50, bottom: 0, left: 0 }}>
                <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 9 }} />
                <YAxis dataKey="model" type="category" width={75} tick={{ fontSize: 9 }} />
                <Bar dataKey="pct" radius={[0, 4, 4, 0]}>
                  {idData.map((entry, i) => (
                    <Cell key={i} fill={i === idData.length - 1 ? '#d62728' : theme.dustyBlue} opacity={i === idData.length - 1 ? 1 : 0.5} />
                  ))}
                  <LabelList dataKey="f1" position="right" formatter={(v) => v.toFixed(3)} style={{ fontSize: 10, fontWeight: 700 }} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Social Unit chart */}
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1, color: theme.unitA, marginBottom: 6 }}>
            Social Unit — Macro-F1
          </div>
          <div style={{ height: 280 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={unitData} layout="vertical" margin={{ top: 0, right: 50, bottom: 0, left: 0 }}>
                <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 9 }} />
                <YAxis dataKey="model" type="category" width={75} tick={{ fontSize: 9 }} />
                <Bar dataKey="pct" radius={[0, 4, 4, 0]}>
                  {unitData.map((entry, i) => (
                    <Cell key={i} fill={i === unitData.length - 1 ? theme.unitA : theme.dustyBlue} opacity={i === unitData.length - 1 ? 1 : 0.5} />
                  ))}
                  <LabelList dataKey="f1" position="right" formatter={(v) => v.toFixed(3)} style={{ fontSize: 10, fontWeight: 700 }} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Big KPIs */}
      <div style={{ display: 'flex', gap: 16, justifyContent: 'center', margin: '8px 0' }}>
        <KPIBox value="+0.380" label="IndivID F1 gain over WhAM" color="#d62728" />
        <KPIBox value="83.7%" label="Relative improvement" color="#d62728" />
        <KPIBox value="−0.017" label="Unit F1 gap (near parity)" color={theme.unitA} />
        <KPIBox value="6.7×" label="Less training data" color="#2ca02c" />
      </div>

      <InsightBox>
        <strong>DCCE-full: 0.834 indivID F1 vs WhAM 0.454.</strong> A laptop-scale model with 1,501 codas beats a 10,000-coda
        VampNet transformer on individual identity — because it encodes the right biological structure.
      </InsightBox>
    </SlideLayout>
  );
};

export default Slide23_Headline;
