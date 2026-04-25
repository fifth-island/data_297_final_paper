import SlideLayout from '../components/SlideLayout';
import { InsightBox, SectionLabel } from '../components/Shared';
import { theme } from '../theme';

const Slide27_WhamUmaps = () => (
  <SlideLayout number="27" title="WhAM Embedding Space" subtitle="UMAP visualizations reveal what WhAM learns — and what it misses">
    <div style={{ display: 'flex', gap: 16, flex: 1 }}>
      {/* Left: UMAP image */}
      <div style={{ flex: 1.2, display: 'flex', flexDirection: 'column' }}>
        <SectionLabel>WhAM L19 UMAP — colored by Social Unit</SectionLabel>
        <div style={{
          background: theme.white,
          borderRadius: 10,
          padding: 10,
          boxShadow: '0 2px 6px rgba(0,0,0,0.06)',
          flex: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}>
          <img
            src="/figures/wham_tsne.png"
            alt="WhAM t-SNE by unit"
            style={{ width: '100%', borderRadius: 6 }}
          />
        </div>
      </div>

      {/* Right: interpretations */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 12 }}>
        <SectionLabel>What the embeddings show</SectionLabel>
        
        <div style={{
          background: '#e8f5e9',
          borderRadius: 10,
          padding: '12px 16px',
          border: '1.5px solid #a5d6a7',
        }}>
          <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 6 }}>✓ Units partially separate</div>
          <div style={{ fontSize: 11, lineHeight: 1.6, color: theme.textSecondary }}>
            Unit A and D form loose clusters, but with significant overlap.
            Unit F dominates the center — consistent with its 59% data share.
          </div>
        </div>

        <div style={{
          background: '#fce4ec',
          borderRadius: 10,
          padding: '12px 16px',
          border: '1.5px solid #ef9a9a',
        }}>
          <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 6 }}>⚠ Individuals not visible</div>
          <div style={{ fontSize: 11, lineHeight: 1.6, color: theme.textSecondary }}>
            No sub-clusters appear within units — consistent with WhAM's low individual ID probe (F1 = 0.454).
            The MLM objective collapses individual-level variation.
          </div>
        </div>

        <div style={{
          background: '#fff3e0',
          borderRadius: 10,
          padding: '12px 16px',
          border: '1.5px solid #ffcc80',
        }}>
          <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 6 }}>🔍 Coda types cluster by rhythm</div>
          <div style={{ fontSize: 11, lineHeight: 1.6, color: theme.textSecondary }}>
            Points with similar coda types (e.g., all 5R1) tend to be neighbors — 
            but WhAM's probe F1 for type is only 0.26.
            The signal exists but is weak and entangled.
          </div>
        </div>

        <div style={{
          background: theme.white,
          borderRadius: 10,
          padding: '10px 14px',
          boxShadow: '0 1px 4px rgba(0,0,0,0.06)',
          display: 'flex',
          gap: 16,
        }}>
          <div style={{ textAlign: 'center', flex: 1 }}>
            <div style={{ fontFamily: theme.fontHeader, fontSize: 20, fontWeight: 700, color: theme.unitA }}>0.895</div>
            <div style={{ fontSize: 9, color: theme.textSecondary, fontWeight: 600 }}>UNIT F1</div>
          </div>
          <div style={{ width: 1, background: theme.bgDark }} />
          <div style={{ textAlign: 'center', flex: 1 }}>
            <div style={{ fontFamily: theme.fontHeader, fontSize: 20, fontWeight: 700, color: '#d62728' }}>0.454</div>
            <div style={{ fontSize: 9, color: theme.textSecondary, fontWeight: 600 }}>INDIV ID F1</div>
          </div>
          <div style={{ width: 1, background: theme.bgDark }} />
          <div style={{ textAlign: 'center', flex: 1 }}>
            <div style={{ fontFamily: theme.fontHeader, fontSize: 20, fontWeight: 700, color: theme.periwinkle }}>0.906</div>
            <div style={{ fontSize: 9, color: theme.textSecondary, fontWeight: 600 }}>YEAR F1</div>
          </div>
        </div>
      </div>
    </div>

    <InsightBox>
      <strong>Bottom line:</strong> WhAM's embedding space is organized around unit/year — not individual identity.
      This motivates DCCE: we need an architecture that explicitly separates what WhAM entangles.
    </InsightBox>
  </SlideLayout>
);

export default Slide27_WhamUmaps;
