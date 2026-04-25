import SlideLayout from '../components/SlideLayout';
import { InsightBox, SectionLabel, KPIBox } from '../components/Shared';
import { theme } from '../theme';

const PipelineStep = ({ number, color, title, details, arrow = true }) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: 0, flex: 1 }}>
    <div style={{
      background: theme.white,
      borderRadius: 10,
      overflow: 'hidden',
      boxShadow: '0 2px 6px rgba(0,0,0,0.06)',
      flex: 1,
    }}>
      <div style={{ background: color, padding: '6px 12px', color: '#fff', display: 'flex', alignItems: 'center', gap: 6 }}>
        <div style={{
          width: 20, height: 20, borderRadius: '50%', background: 'rgba(255,255,255,0.3)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 11, fontWeight: 700,
        }}>{number}</div>
        <div style={{ fontSize: 11, fontWeight: 700 }}>{title}</div>
      </div>
      <div style={{ padding: '8px 12px', fontSize: 10, lineHeight: 1.6, color: theme.textSecondary }}>
        {details}
      </div>
    </div>
    {arrow && <div style={{ fontSize: 20, color: theme.textSecondary, padding: '0 4px' }}>→</div>}
  </div>
);

const Slide32_SynthPipeline = () => (
  <SlideLayout number="32" title="Synthetic Generation Pipeline" subtitle="Using WhAM's VampNet to generate new codas — and testing if they help">
    {/* Pipeline steps */}
    <div style={{ display: 'flex', gap: 0, marginBottom: 14 }}>
      <PipelineStep
        number="1" color={theme.purple}
        title="Select prompt"
        details="Pick a real coda (with known unit + type labels) as the conditioning prompt."
      />
      <PipelineStep
        number="2" color={theme.dustyBlue}
        title="WhAM generates"
        details="VampNet fills masked audio tokens conditioned on the prompt. Produces a novel waveform."
      />
      <PipelineStep
        number="3" color="#f5a623"
        title="Extract features"
        details="Compute mel spectrogram from generated audio. Copy ICI from the original prompt (pseudo-ICI)."
      />
      <PipelineStep
        number="4" color="#2ca02c"
        title="Add to training"
        details="Append synthetic codas (with unit+type labels) to DCCE training set. No individual ID labels."
        arrow={false}
      />
    </div>

    <div style={{ display: 'flex', gap: 16, flex: 1 }}>
      {/* Left: spectrogram comparison */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 10 }}>
        <SectionLabel>Real vs synthetic spectrograms</SectionLabel>
        <div style={{
          background: theme.white,
          borderRadius: 10,
          padding: 8,
          boxShadow: '0 2px 6px rgba(0,0,0,0.06)',
          flex: 1,
          display: 'flex',
          alignItems: 'center',
        }}>
          <img
            src="/figures/synth_spectrograms.png"
            alt="Real vs synthetic spectrograms"
            style={{ width: '100%', borderRadius: 6 }}
          />
        </div>
      </div>

      {/* Right: quality metrics + issues */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 10 }}>
        <SectionLabel>Generation quality</SectionLabel>
        <div style={{ display: 'flex', gap: 8 }}>
          <KPIBox value="81%" label="Expert 2AFC accuracy" color="#2ca02c" />
          <KPIBox value="1,000" label="Synthetic codas generated" color={theme.purple} />
        </div>

        <div style={{
          background: '#fff3e0',
          borderRadius: 10,
          padding: '12px 16px',
          border: '1.5px solid #ffcc80',
        }}>
          <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 6 }}>⚠ Three critical limitations</div>
          <div style={{ fontSize: 10, lineHeight: 1.8 }}>
            <div style={{ display: 'flex', gap: 6, alignItems: 'flex-start', marginBottom: 4 }}>
              <span style={{ fontWeight: 700, color: theme.red }}>1.</span>
              <span><strong>Pseudo-ICI:</strong> ICI is copied from prompt → rhythm encoder sees zero new information</span>
            </div>
            <div style={{ display: 'flex', gap: 6, alignItems: 'flex-start', marginBottom: 4 }}>
              <span style={{ fontWeight: 700, color: theme.red }}>2.</span>
              <span><strong>No individual labels:</strong> L_id cannot train on synthetics — spectral encoder gets no identity supervision</span>
            </div>
            <div style={{ display: 'flex', gap: 6, alignItems: 'flex-start' }}>
              <span style={{ fontWeight: 700, color: theme.red }}>3.</span>
              <span><strong>Contrastive dilution:</strong> Synthetics enter L_contrastive but dilute the individual-level geometry</span>
            </div>
          </div>
        </div>

        <InsightBox variant="light">
          <strong>Mel profile comparison:</strong> Synthetic codas have similar mean mel energy profiles to real codas — they sound convincing but lack the fine-grained spectral variation that carries individual identity.
        </InsightBox>
      </div>
    </div>
  </SlideLayout>
);

export default Slide32_SynthPipeline;
