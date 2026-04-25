import SlideLayout from '../components/SlideLayout';
import { KPIBox, InsightBox, SectionLabel } from '../components/Shared';
import { theme } from '../theme';

/* ─── Image panel helper ─── */
const FigurePanel = ({ src, alt, caption }) => (
  <div>
    <div style={{
      background: theme.white,
      borderRadius: 8,
      padding: '4px',
      boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
    }}>
      <img src={src} alt={alt} style={{ width: '100%', borderRadius: 4, display: 'block' }} />
    </div>
    {caption && (
      <div style={{ fontSize: 9, color: theme.textSecondary, textAlign: 'center', marginTop: 3 }}>
        {caption}
      </div>
    )}
  </div>
);

const Slide12_Spectral = () => (
  <SlideLayout number="12" title="The Spectral Channel" subtitle="Mel-spectrograms encode voice identity — orthogonal to rhythm">
    <div style={{ display: 'flex', gap: 20, flex: 1 }}>
      {/* Left: real mel-spectrogram grid */}
      <div style={{ flex: 1.2, display: 'flex', flexDirection: 'column', gap: 6 }}>
        <SectionLabel>Sample Mel-Spectrograms (64 mel × time, magma)</SectionLabel>
        <FigurePanel
          src="/figures/mel_grid.png"
          alt="3×2 grid of mel-spectrograms by unit"
          caption="Vertical striations = click events · Energy concentrated 2–8 kHz · Shape = voice identity"
        />
      </div>

      {/* Center: centroid violin + independence scatter */}
      <div style={{ flex: 0.9, display: 'flex', flexDirection: 'column', gap: 10 }}>
        <SectionLabel>Spectral Centroid by Unit</SectionLabel>
        <FigurePanel
          src="/figures/centroid_violin.png"
          alt="Violin plot of spectral centroids by unit"
          caption="Centroids overlap — no unit separation from global frequency"
        />
        <SectionLabel>Rhythm vs. Spectral Independence</SectionLabel>
        <FigurePanel
          src="/figures/ici_vs_centroid.png"
          alt="ICI vs spectral centroid scatter"
          caption="Pearson r ≈ 0 — the two channels are statistically independent"
        />
      </div>

      {/* Right: KPIs + insights */}
      <div style={{ flex: 0.7, display: 'flex', flexDirection: 'column', gap: 8 }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          <KPIBox value="0.740" label="Raw Mel → Unit F1" color={theme.unitA} />
          <div style={{ display: 'flex', gap: 6 }}>
            <div style={{ flex: 1, background: theme.white, borderRadius: 8, padding: '10px 8px', textAlign: 'center', boxShadow: '0 1px 3px rgba(0,0,0,0.06)' }}>
              <div style={{ fontFamily: theme.fontHeader, fontSize: 18, fontWeight: 700, color: theme.dustyBlue }}>64 × T</div>
              <div style={{ fontSize: 9, color: theme.textSecondary, fontWeight: 600, textTransform: 'uppercase' }}>Mel dims</div>
            </div>
            <div style={{ flex: 1, background: theme.white, borderRadius: 8, padding: '10px 8px', textAlign: 'center', boxShadow: '0 1px 3px rgba(0,0,0,0.06)' }}>
              <div style={{ fontFamily: theme.fontHeader, fontSize: 18, fontWeight: 700, color: theme.purple }}>r ≈ 0</div>
              <div style={{ fontSize: 9, color: theme.textSecondary, fontWeight: 600, textTransform: 'uppercase' }}>Independence</div>
            </div>
          </div>
        </div>

        <InsightBox variant="dark">
          <strong>Key finding:</strong> Spectral texture separates units (F1=0.740) while ICI cannot (0.599).
          The voice fingerprint is in the spectrogram.
        </InsightBox>

        <InsightBox variant="light">
          <strong>Implication:</strong> A CNN on the full mel-spectrogram is needed — global centroid cannot capture
          within-click vowel texture.
        </InsightBox>
      </div>
    </div>
  </SlideLayout>
);

export default Slide12_Spectral;
