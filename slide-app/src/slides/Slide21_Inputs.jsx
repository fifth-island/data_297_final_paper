import SlideLayout from '../components/SlideLayout';
import { InsightBox, SectionLabel } from '../components/Shared';
import { theme } from '../theme';

const ChannelCard = ({ color, icon, title, inputDesc, shape, example, imgSrc, encodes }) => (
  <div style={{
    background: theme.white,
    borderRadius: 10,
    overflow: 'hidden',
    boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
  }}>
    <div style={{ background: color, padding: '10px 16px', color: '#fff' }}>
      <div style={{ fontSize: 14, fontWeight: 700 }}>{icon} {title}</div>
    </div>
    <div style={{ padding: '12px 16px', flex: 1 }}>
      <SectionLabel>Input</SectionLabel>
      <div style={{ fontSize: 11, lineHeight: 1.6, marginBottom: 10 }}>{inputDesc}</div>

      <div style={{
        background: `${color}10`,
        borderRadius: 6,
        padding: '8px 12px',
        fontFamily: 'monospace',
        fontSize: 10,
        marginBottom: 10,
        border: `1px solid ${color}30`,
      }}>
        <strong>Shape:</strong> {shape}
      </div>

      {imgSrc && (
        <img
          src={imgSrc}
          alt={title}
          style={{ width: '100%', borderRadius: 6, marginBottom: 10, border: `1px solid ${theme.bgDark}` }}
        />
      )}

      <div style={{ fontSize: 10, color: theme.textSecondary, lineHeight: 1.5 }}>
        <strong>Example:</strong> {example}
      </div>
    </div>
    <div style={{ padding: '8px 16px', background: `${color}10`, fontSize: 10, fontWeight: 600 }}>
      Encodes → {encodes}
    </div>
  </div>
);

const Slide21_Inputs = () => (
  <SlideLayout number="21" title="Input Representations" subtitle="Two biological channels → two input modalities">
    <div style={{ display: 'flex', gap: 16, flex: 1 }}>
      <ChannelCard
        color="#4a90d9"
        icon="🎵"
        title="Rhythm Channel"
        inputDesc={<>The <strong>Inter-Click Interval (ICI)</strong> vector: time gaps between consecutive clicks. Zero-padded to length 9, then StandardScaler-normalized.</>}
        shape="ℝ⁹ (float vector)"
        imgSrc="/figures/ici_rhythm_patterns.png"
        example="1+1+3 → [218, 226, 81, 76, 0, 0, 0, 0, 0] ms"
        encodes="Coda type (rhythm pattern)"
      />
      <ChannelCard
        color="#f5a623"
        icon="🔊"
        title="Spectral Channel"
        inputDesc={<>A <strong>mel spectrogram</strong> extracted from the raw audio waveform. 64 mel bins, 128 time frames, fmax = 8 kHz. Captures click spectral shape and micro-timing.</>}
        shape="ℝ⁶⁴ˣ¹²⁸ (2D image)"
        imgSrc="/figures/mel_spectrogram.png"
        example="Each coda becomes a 64×128 grayscale 'image' of acoustic energy"
        encodes="Individual identity (vocal signature)"
      />
    </div>

    <div style={{ display: 'flex', gap: 12, marginTop: 14 }}>
      <InsightBox>
        <strong>Why two channels?</strong> Biology tells us coda type lives in rhythm (ICI), while individual identity lives in spectral texture.
        Separate inputs let each encoder specialize.
      </InsightBox>
      <InsightBox variant="light">
        <strong>Preprocessing:</strong> All mel spectrograms use the same extraction pipeline (librosa, sr=22050, n_fft=2048, hop=512).
        ICIs come from pre-annotated field databases — no audio processing needed.
      </InsightBox>
    </div>
  </SlideLayout>
);

export default Slide21_Inputs;
