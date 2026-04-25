import SlideLayout from '../components/SlideLayout';
import { KPIBox, InsightBox } from '../components/Shared';
import { theme } from '../theme';
import { funnelStages } from '../data';

const FunnelStage = ({ stage, index, maxCount }) => {
  const widthPct = (stage.count / maxCount) * 100;
  const colors = ['#9e9e9e', theme.unitA, theme.purple, theme.purple];
  const color = colors[index];

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: index < 3 ? 0 : 0 }}>
      {/* Funnel box */}
      <div style={{
        width: `${Math.max(widthPct, 40)}%`,
        background: color,
        borderRadius: 8,
        padding: '14px 20px',
        color: '#fff',
        transition: 'width 0.3s ease',
      }}>
        <div style={{ fontFamily: theme.fontHeader, fontSize: 16, fontWeight: 700 }}>{stage.label}</div>
        <div style={{ fontSize: 11, opacity: 0.85, marginTop: 2 }}>{stage.sub}</div>
      </div>

      {/* Loss annotation */}
      {stage.removed > 0 && (
        <div style={{ fontSize: 11, color: theme.textSecondary }}>
          <span style={{ fontWeight: 700, color: theme.red }}>−{stage.removed}</span>{' '}
          ({((stage.removed / (stage.count + stage.removed)) * 100).toFixed(1)}%){' '}
          <span style={{ fontStyle: 'italic' }}>{stage.reason}</span>
        </div>
      )}
    </div>
  );
};

const Arrow = () => (
  <div style={{ textAlign: 'left', paddingLeft: 40, color: theme.textSecondary, fontSize: 18, margin: '4px 0' }}>
    ↓
  </div>
);

const Slide14 = () => (
  <SlideLayout number="16" title="The Data Funnel" subtitle="From 1,501 recordings to the usable subsets for each task">
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: 0 }}>
      <FunnelStage stage={funnelStages[0]} index={0} maxCount={1501} />
      <Arrow />
      <FunnelStage stage={funnelStages[1]} index={1} maxCount={1501} />
      <Arrow />
      <FunnelStage stage={funnelStages[2]} index={2} maxCount={1501} />
      <Arrow />
      <FunnelStage stage={funnelStages[3]} index={3} maxCount={1501} />
    </div>

    {/* KPI row */}
    <div style={{ display: 'flex', gap: 16, justifyContent: 'center', margin: '16px 0' }}>
      <KPIBox value="1,383" label="Clean codas · Unit/Type tasks" color={theme.unitA} />
      <KPIBox value="762" label="Identified · ID-task codas" color={theme.purple} />
      <KPIBox value="12" label="Individuals in ID task" />
    </div>

    <InsightBox variant="light">
      The 621-coda loss (IDN=0, mostly Unit F) is not a data quality problem — it is a field reality.
      When multiple whales vocalize simultaneously, attribution is impossible without bioacoustic localization equipment.
    </InsightBox>
  </SlideLayout>
);

export default Slide14;
