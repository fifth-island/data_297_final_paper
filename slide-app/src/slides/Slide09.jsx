import SlideLayout from '../components/SlideLayout';
import { KPIBox, InsightBox } from '../components/Shared';
import { theme, UNIT_COLORS } from '../theme';
import { unitDistribution, idn0ByUnit, unitICI, yearByUnit } from '../data';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const sparkData = (unitYears) =>
  Object.entries(unitYears).map(([y, c]) => ({ year: y, count: c }));

const UnitCard = ({ unit, data, idn, ici }) => {
  const color = UNIT_COLORS[unit];
  return (
    <div style={{
      background: theme.white,
      borderRadius: 10,
      overflow: 'hidden',
      flex: 1,
      boxShadow: '0 1px 4px rgba(0,0,0,0.06)',
    }}>
      <div style={{ background: color, padding: '10px 16px', color: '#fff' }}>
        <div style={{ fontFamily: theme.fontHeader, fontSize: 18, fontWeight: 700 }}>Unit {unit}</div>
      </div>
      <div style={{ padding: '12px 16px' }}>
        <div style={{ display: 'flex', gap: 16, marginBottom: 8 }}>
          <div>
            <div style={{ fontSize: 24, fontWeight: 700, fontFamily: theme.fontHeader }}>{data.total}</div>
            <div style={{ fontSize: 10, color: theme.textSecondary, fontWeight: 600 }}>TOTAL CODAS</div>
          </div>
          <div>
            <div style={{ fontSize: 24, fontWeight: 700, fontFamily: theme.fontHeader }}>{data.clean}</div>
            <div style={{ fontSize: 10, color: theme.textSecondary, fontWeight: 600 }}>CLEAN</div>
          </div>
        </div>
        <div style={{ fontSize: 12, marginBottom: 4 }}>
          <span style={{ fontWeight: 600 }}>IDN=0 rate:</span>{' '}
          <span style={{ color: idn.unknownPct > 50 ? theme.red : theme.textSecondary, fontWeight: 700 }}>
            {idn.unknownPct}%
          </span>
        </div>
        <div style={{ fontSize: 12, marginBottom: 4 }}>
          <span style={{ fontWeight: 600 }}>Median ICI:</span> {ici.medianICI}ms
        </div>
        <div style={{ fontSize: 12, marginBottom: 8 }}>
          <span style={{ fontWeight: 600 }}>Dominant type:</span> 1+1+3
        </div>
        {/* Sparkline */}
        <div style={{ height: 60 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={sparkData(yearByUnit[unit])} margin={{ top: 0, right: 0, bottom: 0, left: 0 }}>
              <XAxis dataKey="year" tick={{ fontSize: 8 }} tickLine={false} axisLine={false} />
              <Bar dataKey="count" radius={[2, 2, 0, 0]}>
                {sparkData(yearByUnit[unit]).map((_, i) => (
                  <Cell key={i} fill={color} opacity={0.7} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

const Slide09 = () => (
  <SlideLayout number="09" title="The Population" subtitle="3 matrilineal family units · 12 named individuals · 5 years of field work">
    {/* Clan banner */}
    <div style={{
      background: theme.text,
      color: theme.white,
      borderRadius: 8,
      padding: '10px 20px',
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      marginBottom: 20,
    }}>
      <span style={{ fontFamily: theme.fontHeader, fontSize: 16 }}>EC1 Vocal Clan</span>
      <span style={{ fontSize: 13, opacity: 0.8 }}>1,501 codas · 35 coda types · 2005–2010</span>
    </div>

    {/* Unit cards */}
    <div style={{ display: 'flex', gap: 16, marginBottom: 16, flex: 1 }}>
      {['A', 'D', 'F'].map((u) => (
        <UnitCard
          key={u}
          unit={u}
          data={unitDistribution.find((d) => d.unit === u)}
          idn={idn0ByUnit.find((d) => d.unit === u)}
          ici={unitICI.find((d) => d.unit === u)}
        />
      ))}

      {/* KPI strip */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10, width: 130 }}>
        <KPIBox value="12" label="Named individuals" />
        <KPIBox value="36" label="Unique IDNs total" />
        <KPIBox value="49.2%" label="Unknown speaker" color={theme.red} />
      </div>
    </div>

    {/* Insight */}
    <InsightBox>
      These are not three random groups — they are matrilineal whale families. Unit A, D, and F are separate family
      lines within the same clan. They share a coda repertoire but have distinct voices.
    </InsightBox>
  </SlideLayout>
);

export default Slide09;
