import SlideLayout from '../components/SlideLayout';
import { theme } from '../theme';

const Slide10 = () => {
  return (
    <SlideLayout number="10" title="Anatomy of a Coda" subtitle="One WAV file → two information channels">
      {/* Flow stages */}
      <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start', marginBottom: 12 }}>
        {/* Stage 1: Waveform */}
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1, color: theme.unitA, marginBottom: 4 }}>
            Stage 1 — Raw Audio
          </div>
          <div style={{
            background: theme.white,
            borderRadius: 8,
            padding: '6px',
            boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
          }}>
            <img
              src="/figures/waveform.png"
              alt="Waveform of coda 1077"
              style={{ width: '100%', borderRadius: 4, display: 'block' }}
            />
            <div style={{ fontSize: 9, color: theme.unitA, fontWeight: 600, textAlign: 'center', marginTop: 3 }}>
              coda 1077.wav · Unit F · Type: 1+1+3 · 1.50s · 5 clicks
            </div>
          </div>
        </div>

        <div style={{ fontSize: 22, color: theme.textSecondary, alignSelf: 'center', marginTop: 20 }}>→</div>

        {/* Stage 2: ICI */}
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1, color: theme.unitD, marginBottom: 4 }}>
            Stage 2 — ICI Extraction (Rhythm)
          </div>
          <div style={{
            background: theme.white,
            borderRadius: 8,
            padding: '6px',
            boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
          }}>
            <img
              src="/figures/ici_timeline.png"
              alt="ICI timeline for coda 1077"
              style={{ width: '100%', borderRadius: 4, display: 'block' }}
            />
            <div style={{ fontSize: 9, color: theme.unitD, fontWeight: 600, textAlign: 'center', marginTop: 3 }}>
              ICI vector [530, 485, 240, 250] ms → zero-pad → GRU Encoder
            </div>
          </div>
        </div>

        <div style={{ fontSize: 22, color: theme.textSecondary, alignSelf: 'center', marginTop: 20 }}>→</div>

        {/* Stage 3: Mel */}
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1, color: theme.unitFText, marginBottom: 4 }}>
            Stage 3 — Mel-Spectrogram (Spectral)
          </div>
          <div style={{
            background: theme.white,
            borderRadius: 8,
            padding: '6px',
            boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
          }}>
            <img
              src="/figures/mel_spectrogram.png"
              alt="Mel-spectrogram of coda 1077"
              style={{ width: '100%', borderRadius: 4, display: 'block' }}
            />
            <div style={{ fontSize: 9, color: theme.unitFText, fontWeight: 600, textAlign: 'center', marginTop: 3 }}>
              64 × T mel-spectrogram → CNN Encoder
            </div>
          </div>
        </div>
      </div>

      {/* Output row */}
      <div style={{ display: 'flex', gap: 12, marginBottom: 12 }}>
        <div style={{ flex: 1, background: `${theme.unitA}15`, border: `1.5px solid ${theme.unitA}`, borderRadius: 8, padding: '8px 16px', textAlign: 'center' }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: theme.unitA }}>CODA TYPE (rhythm)</div>
          <div style={{ fontSize: 11, color: theme.textSecondary, marginTop: 2 }}>22 categories · F1 = 0.931</div>
        </div>
        <div style={{ flex: 1, background: `${theme.unitF}15`, border: `1.5px solid ${theme.unitF}`, borderRadius: 8, padding: '8px 16px', textAlign: 'center' }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: theme.unitFText }}>SPEAKER IDENTITY (spectral)</div>
          <div style={{ fontSize: 11, color: theme.textSecondary, marginTop: 2 }}>Unit A / D / F + Individual ID · F1 = 0.740</div>
        </div>
      </div>

      {/* Bottom comparison table */}
      <div style={{ display: 'grid', gridTemplateColumns: '120px 1fr 1fr', gap: 0, fontSize: 11, borderTop: `1.5px solid ${theme.text}` }}>
        {[
          ['', 'Rhythm (ICI)', 'Spectral (Mel)'],
          ['Feature dim', '9 values', '64 × T matrix'],
          ['Source', 'Pre-computed CSV', 'Extracted from WAV'],
          ['Predicts', 'Coda type F1 = 0.931', 'Unit F1 = 0.740'],
          ['Fails at', 'Unit / Individual ID', 'Coda type classification'],
        ].map((row, i) => row.map((cell, j) => (
          <div key={`${i}-${j}`} style={{
            padding: '5px 8px',
            fontWeight: i === 0 || j === 0 ? 700 : 400,
            background: i === 0 ? theme.text : i % 2 === 0 ? theme.bgLight : 'transparent',
            color: i === 0 ? theme.white : theme.text,
            borderBottom: `0.5px solid ${theme.bgDark}`,
          }}>
            {cell}
          </div>
        )))}
      </div>
    </SlideLayout>
  );
};

export default Slide10;
