import SlideLayout from '../components/SlideLayout';
import { InsightBox, KPIBox, SectionLabel } from '../components/Shared';
import { theme, UNIT_COLORS } from '../theme';

const ProbeStep = ({ number, label, detail, color }) => (
  <div style={{
    display: 'flex',
    alignItems: 'flex-start',
    gap: 12,
    padding: '10px 0',
    borderBottom: `1px solid ${theme.bgLight}`,
  }}>
    <div style={{
      width: 28, height: 28, borderRadius: '50%', background: color,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      color: '#fff', fontSize: 13, fontWeight: 700, flexShrink: 0,
    }}>
      {number}
    </div>
    <div>
      <div style={{ fontSize: 12, fontWeight: 700 }}>{label}</div>
      <div style={{ fontSize: 10, color: theme.textSecondary, lineHeight: 1.5, marginTop: 2 }}>{detail}</div>
    </div>
  </div>
);

const TargetTag = ({ label, color, icon }) => (
  <div style={{
    background: `${color}15`,
    border: `1.5px solid ${color}`,
    borderRadius: 8,
    padding: '8px 12px',
    flex: 1,
    textAlign: 'center',
  }}>
    <div style={{ fontSize: 16, marginBottom: 4 }}>{icon}</div>
    <div style={{ fontSize: 11, fontWeight: 700, color }}>{label}</div>
  </div>
);

const Slide24_ProbingSetup = () => (
  <SlideLayout number="24" title="Probing WhAM" subtitle="Experimental setup: what does a pre-trained audio model actually learn about whales?">
    <div style={{ display: 'flex', gap: 20, flex: 1 }}>
      {/* Left: pipeline */}
      <div style={{ flex: 1 }}>
        <SectionLabel>Probing Pipeline</SectionLabel>
        <div style={{
          background: theme.white,
          borderRadius: 10,
          padding: '14px 18px',
          boxShadow: '0 2px 6px rgba(0,0,0,0.06)',
        }}>
          <ProbeStep
            number="1" color={theme.purple}
            label="Extract hidden states"
            detail="Pass each coda's mel spectrogram through WhAM (frozen). Collect the 1,280-d hidden vector at every transformer layer (20 total)."
          />
          <ProbeStep
            number="2" color={theme.dustyBlue}
            label="Train linear probe"
            detail="For each of the 20 layers × 4 targets, fit a logistic regression (no hidden layers) on the frozen embeddings. Stratified 5-fold CV."
          />
          <ProbeStep
            number="3" color="#2ca02c"
            label="Measure macro-F1"
            detail="Report macro-averaged F1 to handle class imbalance (Unit F has 59% of data). Compare each layer's profile across targets."
          />
        </div>

        <div style={{
          background: theme.white,
          borderRadius: 10,
          padding: '10px 14px',
          boxShadow: '0 1px 4px rgba(0,0,0,0.06)',
          marginTop: 12,
          display: 'flex',
          alignItems: 'center',
          gap: 12,
        }}>
          <div style={{ fontFamily: 'monospace', fontSize: 11, fontWeight: 600, color: theme.purple }}>
            20 layers × 1,280 dims × 4 targets = 80 probes
          </div>
        </div>
      </div>

      {/* Right: targets + model card */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 12 }}>
        <SectionLabel>Four probe targets</SectionLabel>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
          <TargetTag label="Social Unit" color={theme.unitA} icon="👥" />
          <TargetTag label="Coda Type" color="#2ca02c" icon="🎵" />
          <TargetTag label="Individual ID" color="#d62728" icon="🐋" />
          <TargetTag label="Rec. Year" color={theme.periwinkle} icon="📅" />
        </div>

        <div style={{
          background: theme.white,
          borderRadius: 10,
          padding: '14px 18px',
          boxShadow: '0 2px 6px rgba(0,0,0,0.06)',
        }}>
          <SectionLabel>WhAM Model Card</SectionLabel>
          {[
            ['Architecture', 'VampNet transformer (bidirectional)'],
            ['Pre-training', '~10,000 codas (external corpus)'],
            ['Parameters', '~30M'],
            ['Hidden dim', '1,280 per layer'],
            ['Layers', '20 transformer blocks'],
            ['Task', 'Audio codec token prediction (MLM)'],
          ].map(([k, v], i) => (
            <div key={i} style={{
              display: 'flex',
              justifyContent: 'space-between',
              padding: '4px 0',
              fontSize: 11,
              borderBottom: i < 5 ? `1px solid ${theme.bgLight}` : 'none',
            }}>
              <span style={{ color: theme.textSecondary }}>{k}</span>
              <span style={{ fontWeight: 600, fontFamily: 'monospace', fontSize: 10 }}>{v}</span>
            </div>
          ))}
        </div>

        <InsightBox variant="light">
          <strong>Key question:</strong> Does WhAM's self-supervised objective (audio token prediction) give rise to representations that encode whale identity — even though it was never trained on labels?
        </InsightBox>
      </div>
    </div>
  </SlideLayout>
);

export default Slide24_ProbingSetup;
