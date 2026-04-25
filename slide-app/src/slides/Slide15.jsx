import SlideLayout from '../components/SlideLayout';
import { InsightBox } from '../components/Shared';
import { theme, UNIT_COLORS } from '../theme';
import { individuals, individualIdResults } from '../data';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, LabelList } from 'recharts';

const WhaleCard = ({ ind }) => {
  const color = UNIT_COLORS[ind.unit];
  return (
    <div style={{
      background: theme.white,
      borderRadius: 8,
      overflow: 'hidden',
      boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
    }}>
      <div style={{ background: color, padding: '4px 10px', color: '#fff', fontSize: 10, fontWeight: 700 }}>
        Unit {ind.unit}
      </div>
      <div style={{ padding: '6px 10px' }}>
        <div style={{ fontFamily: theme.fontHeader, fontSize: 14, fontWeight: 700 }}>IDN {ind.idn}</div>
        <div style={{ fontSize: 9, color: theme.textSecondary, lineHeight: 1.5, marginTop: 2 }}>
          {ind.codas} codas · {ind.topType} · {ind.meanICI}ms
        </div>
      </div>
    </div>
  );
};

const Slide15 = () => {
  const barData = individualIdResults.map((r) => ({
    ...r,
    f1pct: Math.round(r.f1 * 100),
  }));

  return (
    <SlideLayout number="17" title="12 Voices in the Data" subtitle="Individual ID is the hardest classification task — and the most biologically meaningful">
      <div style={{ display: 'flex', gap: 20, flex: 1 }}>
        {/* Card grid */}
        <div style={{ flex: 2 }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8 }}>
            {individuals.map((ind) => (
              <WhaleCard key={ind.idn} ind={ind} />
            ))}
          </div>
        </div>

        {/* Results bar chart */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1, color: theme.textSecondary, marginBottom: 8 }}>
            Individual ID Macro-F1
          </div>
          <div style={{ flex: 1 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={barData} layout="vertical" margin={{ top: 0, right: 40, bottom: 0, left: 0 }}>
                <XAxis type="number" domain={[0, 100]} hide />
                <YAxis dataKey="model" type="category" width={85} tick={{ fontSize: 9 }} />
                <Bar dataKey="f1pct" radius={[0, 4, 4, 0]}>
                  {barData.map((entry, i) => (
                    <Cell key={i} fill={i >= 3 ? theme.periwinkle : theme.dustyBlue} opacity={i >= 3 ? 1 : 0.6} />
                  ))}
                  <LabelList dataKey="f1" position="right" formatter={(v) => v.toFixed(3)} style={{ fontSize: 10, fontWeight: 600 }} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Bottom KPIs */}
      <div style={{ display: 'flex', gap: 24, justifyContent: 'center', margin: '10px 0' }}>
        {[
          { v: '12 classes', sub: '~63 codas/class' },
          { v: 'Rhythm shared', sub: 'ICI ≠ ID' },
          { v: '762 total codas', sub: 'tiny dataset' },
        ].map((k) => (
          <div key={k.v} style={{
            background: theme.white,
            borderRadius: 8,
            padding: '8px 18px',
            textAlign: 'center',
            boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
          }}>
            <div style={{ fontSize: 13, fontWeight: 700 }}>{k.v}</div>
            <div style={{ fontSize: 10, color: theme.textSecondary }}>{k.sub}</div>
          </div>
        ))}
      </div>

      <InsightBox>
        The DCCE spectral-only model achieves F1=0.787 — nearly 2x the WhAM baseline (0.454).
        The fusion model reaches 0.834: both channels together outperform either alone.
      </InsightBox>
    </SlideLayout>
  );
};

export default Slide15;
