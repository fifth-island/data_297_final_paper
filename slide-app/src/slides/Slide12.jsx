import SlideLayout from '../components/SlideLayout';
import { InsightBox } from '../components/Shared';
import { theme } from '../theme';
import { codaTypeDefs } from '../data';

const ICITimeline = ({ icis, clicks }) => {
  const total = icis.reduce((a, b) => a + b, 0);
  const positions = [0];
  icis.forEach((v) => positions.push(positions[positions.length - 1] + v));
  const scale = 220 / total;

  const getColor = (ms) => (ms < 100 ? theme.unitF : ms < 200 ? theme.unitD : theme.unitA);

  return (
    <svg width={260} height={50} viewBox="0 0 260 50">
      {/* Click dots */}
      {positions.map((p, i) => (
        <circle key={i} cx={20 + p * scale} cy={15} r={5} fill={theme.text} />
      ))}
      {/* ICI brackets */}
      {icis.map((ms, i) => {
        const x1 = 20 + positions[i] * scale;
        const x2 = 20 + positions[i + 1] * scale;
        return (
          <g key={i}>
            <line x1={x1} y1={25} x2={x2} y2={25} stroke={getColor(ms)} strokeWidth={2} />
            <text x={(x1 + x2) / 2} y={42} textAnchor="middle" fontSize={8} fill={theme.textSecondary} fontFamily="Open Sans">
              {ms}ms
            </text>
          </g>
        );
      })}
    </svg>
  );
};

const CodaCard = ({ def }) => (
  <div style={{
    background: theme.white,
    borderRadius: 10,
    padding: '14px 16px',
    boxShadow: '0 1px 4px rgba(0,0,0,0.06)',
    flex: 1,
  }}>
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 6 }}>
      <span style={{ fontFamily: theme.fontHeader, fontSize: 20, fontWeight: 700 }}>{def.type}</span>
      <span style={{ fontSize: 11, color: theme.textSecondary, fontWeight: 600 }}>
        {def.count} codas · {def.pct}%
      </span>
    </div>
    <ICITimeline icis={def.icis} clicks={def.clicks} />
    <div style={{ fontSize: 11, color: theme.text, marginTop: 4, lineHeight: 1.4 }}>
      {def.description}
    </div>
    <div style={{ fontSize: 10, color: theme.dustyBlue, fontWeight: 600, marginTop: 4, fontStyle: 'italic' }}>
      {def.role}
    </div>
  </div>
);

const Slide12 = () => (
  <SlideLayout number="14" title="Reading a Coda Type" subtitle="The click pattern is the word — ICI is the pronunciation">
    <div style={{
      fontSize: 12,
      color: theme.textSecondary,
      fontWeight: 500,
      marginBottom: 14,
      fontStyle: 'italic',
    }}>
      Coda types are named by their click count and rhythm. The name is literally the pattern.
    </div>

    {/* 2x2 card grid */}
    <div style={{ display: 'flex', gap: 14, marginBottom: 14, flex: 1 }}>
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 14 }}>
        <CodaCard def={codaTypeDefs[0]} />
        <CodaCard def={codaTypeDefs[2]} />
      </div>
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 14 }}>
        <CodaCard def={codaTypeDefs[1]} />
        <CodaCard def={codaTypeDefs[3]} />
      </div>
    </div>

    {/* KPI row */}
    <div style={{ display: 'flex', gap: 24, justifyContent: 'center', marginBottom: 10 }}>
      {[
        { v: '22', l: 'active coda types' },
        { v: '75%', l: 'covered by top 4' },
        { v: '9', l: 'shared across all units' },
      ].map((kpi) => (
        <div key={kpi.l} style={{ textAlign: 'center' }}>
          <span style={{ fontFamily: theme.fontHeader, fontSize: 22, fontWeight: 700 }}>{kpi.v}</span>{' '}
          <span style={{ fontSize: 11, color: theme.textSecondary }}>{kpi.l}</span>
        </div>
      ))}
    </div>

    <InsightBox>
      The top 4 types (1+1+3, 5R1, 4D, 7D1) = 75% of all codas — and all 4 appear in every unit.
      Unit A, D, and F speak the same "words." The identity is in the <em>accent</em>, not the word.
    </InsightBox>
  </SlideLayout>
);

export default Slide12;
