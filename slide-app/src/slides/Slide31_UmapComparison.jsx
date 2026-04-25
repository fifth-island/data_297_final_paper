import SlideLayout from '../components/SlideLayout';
import { InsightBox, SectionLabel } from '../components/Shared';
import { theme } from '../theme';

const Slide31_UmapComparison = () => (
  <SlideLayout number="31" title="Embedding Space: WhAM vs DCCE" subtitle="Same codas, radically different representations">
    <div style={{ display: 'flex', gap: 16, flex: 1 }}>
      {/* Side-by-side UMAPs */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 8 }}>
        <SectionLabel>WhAM L19 (1,280-d)</SectionLabel>
        <div style={{
          background: theme.white,
          borderRadius: 10,
          padding: 8,
          boxShadow: '0 2px 6px rgba(0,0,0,0.06)',
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
        }}>
          <img
            src="/figures/wham_tsne.png"
            alt="WhAM UMAP"
            style={{ width: '100%', borderRadius: 6, flex: 1, objectFit: 'contain' }}
          />
          <div style={{ display: 'flex', gap: 8, marginTop: 8, justifyContent: 'center' }}>
            <div style={{ background: '#fce4ec', borderRadius: 6, padding: '4px 10px', fontSize: 10, fontWeight: 600 }}>
              Units overlap heavily
            </div>
            <div style={{ background: '#fce4ec', borderRadius: 6, padding: '4px 10px', fontSize: 10, fontWeight: 600 }}>
              No individual sub-clusters
            </div>
          </div>
        </div>
      </div>

      <div style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '0 4px',
      }}>
        <div style={{ fontSize: 24, color: theme.textSecondary }}>→</div>
        <div style={{
          fontSize: 9,
          fontWeight: 700,
          color: theme.textSecondary,
          textTransform: 'uppercase',
          letterSpacing: 1,
          writingMode: 'vertical-lr',
          marginTop: 8,
        }}>
          vs
        </div>
      </div>

      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 8 }}>
        <SectionLabel>DCCE-full (64-d)</SectionLabel>
        <div style={{
          background: theme.white,
          borderRadius: 10,
          padding: 8,
          boxShadow: '0 2px 6px rgba(0,0,0,0.06)',
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
        }}>
          <img
            src="/figures/dcce_umap.png"
            alt="DCCE UMAP"
            style={{ width: '100%', borderRadius: 6, flex: 1, objectFit: 'contain' }}
          />
          <div style={{ display: 'flex', gap: 8, marginTop: 8, justifyContent: 'center' }}>
            <div style={{ background: '#e8f5e9', borderRadius: 6, padding: '4px 10px', fontSize: 10, fontWeight: 600 }}>
              Clean unit separation
            </div>
            <div style={{ background: '#e8f5e9', borderRadius: 6, padding: '4px 10px', fontSize: 10, fontWeight: 600 }}>
              Individual sub-clusters visible
            </div>
          </div>
        </div>
      </div>
    </div>

    {/* Comparison metrics */}
    <div style={{
      display: 'flex',
      gap: 16,
      marginTop: 12,
      background: theme.white,
      borderRadius: 10,
      padding: '12px 20px',
      boxShadow: '0 1px 4px rgba(0,0,0,0.06)',
    }}>
      <div style={{ flex: 1, textAlign: 'center' }}>
        <div style={{ fontSize: 9, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1, color: theme.textSecondary, marginBottom: 4 }}>Dimensionality</div>
        <div style={{ fontSize: 11 }}><span style={{ color: theme.textSecondary }}>1,280-d</span> → <strong>64-d</strong> (20× smaller)</div>
      </div>
      <div style={{ width: 1, background: theme.bgDark }} />
      <div style={{ flex: 1, textAlign: 'center' }}>
        <div style={{ fontSize: 9, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1, color: theme.textSecondary, marginBottom: 4 }}>Individual ID F1</div>
        <div style={{ fontSize: 11 }}><span style={{ color: theme.textSecondary }}>0.454</span> → <strong style={{ color: '#d62728' }}>0.834</strong> (+83.7%)</div>
      </div>
      <div style={{ width: 1, background: theme.bgDark }} />
      <div style={{ flex: 1, textAlign: 'center' }}>
        <div style={{ fontSize: 9, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1, color: theme.textSecondary, marginBottom: 4 }}>Unit F1</div>
        <div style={{ fontSize: 11 }}><span style={{ color: theme.textSecondary }}>0.895</span> → <strong style={{ color: theme.unitA }}>0.878</strong> (−0.017, near parity)</div>
      </div>
      <div style={{ width: 1, background: theme.bgDark }} />
      <div style={{ flex: 1, textAlign: 'center' }}>
        <div style={{ fontSize: 9, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1, color: theme.textSecondary, marginBottom: 4 }}>Training data</div>
        <div style={{ fontSize: 11 }}><span style={{ color: theme.textSecondary }}>~10K codas</span> → <strong style={{ color: '#2ca02c' }}>1,501 codas</strong> (6.7×)</div>
      </div>
    </div>

    <InsightBox>
      <strong>20× fewer dimensions, 6.7× less data — yet DCCE produces far more structured embeddings.</strong>
      The dual-channel design + contrastive objective creates a space where both unit and individual identity are linearly separable.
    </InsightBox>
  </SlideLayout>
);

export default Slide31_UmapComparison;
