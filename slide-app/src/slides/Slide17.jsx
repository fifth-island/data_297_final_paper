import SlideLayout from '../components/SlideLayout';
import { theme } from '../theme';
import { designDecisions } from '../data';

const cardColors = [
  theme.unitA,
  theme.unitD,
  theme.unitF,
  theme.periwinkle,
  theme.dustyBlue,
  theme.purple,
];

const DecisionCard = ({ decision, index }) => {
  const color = cardColors[index % cardColors.length];
  return (
    <div style={{
      background: theme.white,
      borderRadius: 8,
      overflow: 'hidden',
      boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
    }}>
      <div style={{ background: color, padding: '6px 12px', color: '#fff' }}>
        <div style={{ fontSize: 11, fontWeight: 700 }}>{decision.title}</div>
      </div>
      <div style={{ padding: '8px 12px', fontSize: 10, lineHeight: 1.5 }}>
        <div style={{ marginBottom: 4 }}>
          <span style={{ fontWeight: 700, color: theme.textSecondary, fontSize: 9, textTransform: 'uppercase' }}>Finding: </span>
          {decision.finding}
        </div>
        <div style={{ marginBottom: 4 }}>
          <span style={{ fontWeight: 700, color: theme.textSecondary, fontSize: 9, textTransform: 'uppercase' }}>Why: </span>
          {decision.why}
        </div>
        <div style={{
          background: `${color}15`,
          borderRadius: 4,
          padding: '4px 8px',
          fontWeight: 600,
          fontSize: 10,
        }}>
          → {decision.decision}
        </div>
      </div>
    </div>
  );
};

const Slide17 = () => (
  <SlideLayout number="19" title="From Data to Design" subtitle="Every architectural decision has a data justification">
    <div style={{
      display: 'grid',
      gridTemplateColumns: 'repeat(3, 1fr)',
      gap: 12,
      flex: 1,
    }}>
      {designDecisions.map((d, i) => (
        <DecisionCard key={d.title} decision={d} index={i} />
      ))}
    </div>

    <div style={{
      textAlign: 'center',
      marginTop: 12,
      fontSize: 12,
      fontWeight: 600,
      color: theme.textSecondary,
      borderTop: `1.5px solid ${theme.text}`,
      paddingTop: 10,
    }}>
      6 architectural decisions · All traceable to EDA findings · None arbitrary
    </div>
  </SlideLayout>
);

export default Slide17;
