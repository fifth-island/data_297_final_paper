import SlideLayout from '../components/SlideLayout';
import { InsightBox } from '../components/Shared';
import { theme } from '../theme';

const EncoderBox = ({ title, color, icon, specs, input }) => (
  <div style={{
    background: theme.white,
    borderRadius: 10,
    overflow: 'hidden',
    boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
    flex: 1,
  }}>
    <div style={{ background: color, padding: '10px 16px', color: '#fff' }}>
      <div style={{ fontSize: 14, fontWeight: 700 }}>{icon} {title}</div>
    </div>
    <div style={{ padding: '12px 16px' }}>
      <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1, color: theme.textSecondary, marginBottom: 6 }}>
        Input
      </div>
      <div style={{
        background: `${color}15`,
        borderRadius: 6,
        padding: '6px 10px',
        fontSize: 11,
        fontWeight: 600,
        marginBottom: 12,
      }}>
        {input}
      </div>
      {specs.map((s, i) => (
        <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 0', fontSize: 11, borderBottom: i < specs.length - 1 ? `1px solid ${theme.bgLight}` : 'none' }}>
          <span style={{ color: theme.textSecondary }}>{s.label}</span>
          <span style={{ fontWeight: 700, fontFamily: 'monospace', fontSize: 10 }}>{s.value}</span>
        </div>
      ))}
    </div>
  </div>
);

const FusionBox = () => (
  <div style={{
    background: theme.white,
    borderRadius: 10,
    overflow: 'hidden',
    boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
  }}>
    <div style={{ background: '#2ca02c', padding: '8px 16px', color: '#fff' }}>
      <div style={{ fontSize: 13, fontWeight: 700 }}>🔗 Fusion MLP</div>
    </div>
    <div style={{ padding: '10px 16px', display: 'flex', alignItems: 'center', gap: 12 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flex: 1 }}>
        {['r_emb (64d)', 's_emb (64d)'].map((l, i) => (
          <div key={i} style={{
            background: i === 0 ? '#4a90d915' : '#f5a62315',
            border: `1.5px solid ${i === 0 ? '#4a90d9' : '#f5a623'}`,
            borderRadius: 6,
            padding: '4px 10px',
            fontSize: 10,
            fontWeight: 700,
          }}>
            {l}
          </div>
        ))}
        <span style={{ fontSize: 16, color: theme.textSecondary }}>→</span>
        <div style={{ fontSize: 10, color: theme.textSecondary }}>concat(128d)</div>
        <span style={{ fontSize: 16, color: theme.textSecondary }}>→</span>
        <div style={{ fontSize: 10, color: theme.textSecondary }}>LayerNorm → Linear → ReLU</div>
        <span style={{ fontSize: 16, color: theme.textSecondary }}>→</span>
        <div style={{
          background: '#2ca02c15',
          border: '1.5px solid #2ca02c',
          borderRadius: 6,
          padding: '4px 10px',
          fontSize: 10,
          fontWeight: 700,
        }}>
          z (64d, L2-norm)
        </div>
      </div>
    </div>
  </div>
);

const Slide18_Architecture = () => (
  <SlideLayout number="20" title="DCCE Architecture" subtitle="Dual-Channel Contrastive Encoder — biology as an architectural prior">
    <div style={{ display: 'flex', gap: 16, marginBottom: 16 }}>
      <EncoderBox
        title="Rhythm Encoder"
        color="#4a90d9"
        icon="🎵"
        input="ICI vector (length 9, zero-padded, StandardScaler)"
        specs={[
          { label: 'Architecture', value: '2-layer BiGRU' },
          { label: 'Hidden size', value: '64' },
          { label: 'Output', value: 'r_emb ∈ ℝ⁶⁴' },
          { label: 'Encodes', value: 'Coda type (rhythm)' },
        ]}
      />
      <EncoderBox
        title="Spectral Encoder"
        color="#f5a623"
        icon="🔊"
        input="Mel spectrogram (64 bins × 128 frames, fmax=8kHz)"
        specs={[
          { label: 'Architecture', value: '3-block CNN' },
          { label: 'Blocks', value: 'Conv2d+BN+ReLU+Pool' },
          { label: 'Output', value: 's_emb ∈ ℝ⁶⁴' },
          { label: 'Encodes', value: 'Identity (spectral)' },
        ]}
      />
    </div>

    <FusionBox />

    <div style={{ display: 'flex', gap: 14, marginTop: 16 }}>
      <InsightBox>
        <strong>Design principle:</strong> Process rhythm and spectral channels through separate encoders —
        no information blending at the input. The known biological structure becomes the model's inductive bias.
      </InsightBox>
      <InsightBox variant="light">
        <strong>vs. WhAM:</strong> Waveform → codec → transformer. No inductive bias about two-channel structure.
        10,000 codas + 5 days GPU training. DCCE: 1,501 codas on a laptop.
      </InsightBox>
    </div>
  </SlideLayout>
);

export default Slide18_Architecture;
