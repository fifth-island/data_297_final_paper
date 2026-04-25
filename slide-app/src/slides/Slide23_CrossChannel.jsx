import SlideLayout from '../components/SlideLayout';
import { InsightBox, SectionLabel } from '../components/Shared';
import { theme } from '../theme';

const PairDiagram = ({ codaA, codaB, unitLabel, resultA, resultB }) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: 12, justifyContent: 'center', padding: '8px 0' }}>
    <div style={{ textAlign: 'center' }}>
      <div style={{ fontSize: 10, fontWeight: 700, marginBottom: 6 }}>{codaA} ({unitLabel})</div>
      <div style={{ display: 'flex', gap: 4 }}>
        <div style={{ background: '#4a90d930', border: '2px solid #4a90d9', borderRadius: 6, padding: '5px 8px', fontSize: 9, fontWeight: 700 }}>R(A)</div>
        <div style={{ background: '#f5a62330', border: '2px solid #f5a623', borderRadius: 6, padding: '5px 8px', fontSize: 9, fontWeight: 700 }}>S(A)</div>
      </div>
    </div>
    <div style={{ fontSize: 18, color: theme.textSecondary }}>⇄</div>
    <div style={{ textAlign: 'center' }}>
      <div style={{ fontSize: 10, fontWeight: 700, marginBottom: 6 }}>{codaB} ({unitLabel})</div>
      <div style={{ display: 'flex', gap: 4 }}>
        <div style={{ background: '#4a90d930', border: '2px solid #4a90d9', borderRadius: 6, padding: '5px 8px', fontSize: 9, fontWeight: 700 }}>R(B)</div>
        <div style={{ background: '#f5a62330', border: '2px solid #f5a623', borderRadius: 6, padding: '5px 8px', fontSize: 9, fontWeight: 700 }}>S(B)</div>
      </div>
    </div>
    <div style={{ fontSize: 18, color: theme.textSecondary }}>→</div>
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      <div style={{ background: '#e8f5e9', border: '2px solid #2ca02c', borderRadius: 6, padding: '5px 10px', fontSize: 9, fontWeight: 700 }}>
        {resultA}
      </div>
      <div style={{ textAlign: 'center', fontSize: 9, fontWeight: 700, color: '#2ca02c' }}>positive pair</div>
      <div style={{ background: '#e8f5e9', border: '2px solid #2ca02c', borderRadius: 6, padding: '5px 10px', fontSize: 9, fontWeight: 700 }}>
        {resultB}
      </div>
    </div>
  </div>
);

const ComparisonRow = ({ method, how, limitation }) => (
  <div style={{ display: 'flex', gap: 0, fontSize: 10, borderBottom: `1px solid ${theme.bgLight}` }}>
    <div style={{ padding: '6px 10px', fontWeight: 700, width: 100 }}>{method}</div>
    <div style={{ padding: '6px 10px', flex: 1 }}>{how}</div>
    <div style={{ padding: '6px 10px', flex: 1, color: theme.red, fontStyle: 'italic' }}>{limitation}</div>
  </div>
);

const Slide23_CrossChannel = () => (
  <SlideLayout number="23" title="Cross-Channel Pairing" subtitle="The key novelty — creating multi-view pairs from biological structure">
    <div style={{ display: 'flex', gap: 16, flex: 1 }}>
      {/* Left: mechanism */}
      <div style={{ flex: 1.2, display: 'flex', flexDirection: 'column', gap: 12 }}>
        <SectionLabel>Mechanism</SectionLabel>
        <div style={{
          background: theme.white,
          borderRadius: 10,
          padding: '14px 16px',
          boxShadow: '0 2px 6px rgba(0,0,0,0.06)',
        }}>
          <div style={{ fontSize: 11, lineHeight: 1.6, marginBottom: 12 }}>
            For each pair of codas from the <strong>same social unit</strong>, swap the rhythm and spectral channels to create two "views":
          </div>
          <PairDiagram
            codaA="Coda A" codaB="Coda B" unitLabel="Unit F"
            resultA="R(A) ⊕ S(B)" resultB="R(B) ⊕ S(A)"
          />
          <div style={{ fontSize: 10, color: theme.textSecondary, marginTop: 8, lineHeight: 1.5 }}>
            NT-Xent pulls these views together in embedding space — forcing the model to learn <em>what's shared across channels</em> (unit identity) rather than surface features.
          </div>
        </div>

        <div style={{
          background: theme.white,
          borderRadius: 10,
          padding: '14px 16px',
          boxShadow: '0 2px 6px rgba(0,0,0,0.06)',
        }}>
          <SectionLabel>Why different from standard augmentation?</SectionLabel>
          <div style={{ fontSize: 10, lineHeight: 1.6, color: theme.textSecondary }}>
            Standard contrastive learning creates views via <strong>random crops, flips, color jitter</strong> — generic transforms.
            Cross-channel pairing exploits <strong>domain-specific structure</strong>: the biological fact that rhythm and spectral channels encode <em>different</em> identity levels.
          </div>
        </div>
      </div>

      {/* Right: CLIP analogy + comparison */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 12 }}>
        <div style={{
          background: '#8172B215',
          borderRadius: 10,
          padding: '14px 16px',
          border: '2px solid #8172B240',
        }}>
          <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 8, color: theme.purple }}>💡 CLIP Analogy</div>
          <div style={{ fontSize: 11, lineHeight: 1.6 }}>
            CLIP pairs <strong>images ↔ text</strong> — two modalities describing the same concept.
            <br /><br />
            DCCE pairs <strong>rhythm ↔ spectral</strong> — two biological channels from the same social unit.
            <br /><br />
            Both learn <em>cross-modal alignment</em>, but DCCE's pairing is biologically motivated.
          </div>
        </div>

        <div style={{
          background: theme.white,
          borderRadius: 10,
          overflow: 'hidden',
          boxShadow: '0 2px 6px rgba(0,0,0,0.06)',
        }}>
          <div style={{ display: 'flex', gap: 0, fontSize: 10, background: theme.text, color: theme.white }}>
            <div style={{ padding: '6px 10px', fontWeight: 700, width: 100 }}>Method</div>
            <div style={{ padding: '6px 10px', flex: 1, fontWeight: 700 }}>View creation</div>
            <div style={{ padding: '6px 10px', flex: 1, fontWeight: 700 }}>Limitation</div>
          </div>
          <ComparisonRow method="SimCLR" how="Random augmentations of same input" limitation="No domain knowledge" />
          <ComparisonRow method="CLIP" how="Image ↔ text of same concept" limitation="Needs caption data" />
          <ComparisonRow method="DCCE" how="Rhythm(A) ⊕ Spectral(B) within unit" limitation="Needs channel annotations" />
        </div>

        <InsightBox variant="green">
          <strong>Ablation proof:</strong> Cross-channel pairing adds <strong>+0.222</strong> to unit F1 over late fusion — the single largest effect in any ablation.
        </InsightBox>
      </div>
    </div>
  </SlideLayout>
);

export default Slide23_CrossChannel;
