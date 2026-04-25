import SlideLayout from '../components/SlideLayout';
import { InsightBox } from '../components/Shared';
import { theme } from '../theme';

const LossCard = ({ color, title, formula, target, input, detail }) => (
  <div style={{
    background: theme.white,
    borderRadius: 10,
    overflow: 'hidden',
    boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
    flex: 1,
  }}>
    <div style={{ background: color, padding: '8px 14px', color: '#fff' }}>
      <div style={{ fontSize: 12, fontWeight: 700 }}>{title}</div>
    </div>
    <div style={{ padding: '10px 14px' }}>
      <div style={{
        fontFamily: 'monospace',
        fontSize: 11,
        background: `${color}10`,
        borderRadius: 4,
        padding: '6px 8px',
        marginBottom: 8,
        fontWeight: 600,
      }}>
        {formula}
      </div>
      <div style={{ fontSize: 10, lineHeight: 1.6 }}>
        <div><strong>Input:</strong> {input}</div>
        <div><strong>Target:</strong> {target}</div>
        <div style={{ color: theme.textSecondary, marginTop: 4 }}>{detail}</div>
      </div>
    </div>
  </div>
);

const CrossChannelDiagram = () => (
  <div style={{
    background: theme.white,
    borderRadius: 10,
    padding: '14px 20px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
  }}>
    <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1, color: theme.textSecondary, marginBottom: 10 }}>
      Cross-Channel Positive Pair Construction (Key Novelty)
    </div>
    <div style={{ display: 'flex', alignItems: 'center', gap: 20, justifyContent: 'center' }}>
      {/* Coda A */}
      <div style={{ textAlign: 'center' }}>
        <div style={{ fontSize: 11, fontWeight: 700, marginBottom: 6 }}>Coda A (Unit F)</div>
        <div style={{ display: 'flex', gap: 4 }}>
          <div style={{ background: '#4a90d930', border: '2px solid #4a90d9', borderRadius: 6, padding: '6px 10px', fontSize: 10, fontWeight: 700 }}>
            rhythm(A)
          </div>
          <div style={{ background: '#f5a62330', border: '2px solid #f5a623', borderRadius: 6, padding: '6px 10px', fontSize: 10, fontWeight: 700 }}>
            spectral(A)
          </div>
        </div>
      </div>

      {/* Swap arrows */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
        <div style={{ fontSize: 20, transform: 'rotate(-15deg)', color: '#4a90d9' }}>⤻</div>
        <div style={{ fontSize: 9, fontWeight: 700, color: theme.textSecondary }}>SWAP</div>
        <div style={{ fontSize: 20, transform: 'rotate(15deg)', color: '#f5a623' }}>⤻</div>
      </div>

      {/* Coda B */}
      <div style={{ textAlign: 'center' }}>
        <div style={{ fontSize: 11, fontWeight: 700, marginBottom: 6 }}>Coda B (Unit F)</div>
        <div style={{ display: 'flex', gap: 4 }}>
          <div style={{ background: '#4a90d930', border: '2px solid #4a90d9', borderRadius: 6, padding: '6px 10px', fontSize: 10, fontWeight: 700 }}>
            rhythm(B)
          </div>
          <div style={{ background: '#f5a62330', border: '2px solid #f5a623', borderRadius: 6, padding: '6px 10px', fontSize: 10, fontWeight: 700 }}>
            spectral(B)
          </div>
        </div>
      </div>

      <div style={{ fontSize: 22, color: theme.textSecondary }}>→</div>

      {/* Result */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        <div style={{ background: '#e8f5e9', border: '2px solid #2ca02c', borderRadius: 8, padding: '6px 12px', fontSize: 10, fontWeight: 700, textAlign: 'center' }}>
          View 1: rhythm(A) ⊕ spectral(B)
        </div>
        <div style={{ textAlign: 'center', fontSize: 10, fontWeight: 700, color: '#2ca02c' }}>positive pair</div>
        <div style={{ background: '#e8f5e9', border: '2px solid #2ca02c', borderRadius: 8, padding: '6px 12px', fontSize: 10, fontWeight: 700, textAlign: 'center' }}>
          View 2: rhythm(B) ⊕ spectral(A)
        </div>
      </div>
    </div>
  </div>
);

const Slide19_Training = () => (
  <SlideLayout number="22" title="Training Objective" subtitle="L = L_contrastive(z) + λ₁·L_type(r_emb) + λ₂·L_id(s_emb)">
    {/* Three loss cards */}
    <div style={{ display: 'flex', gap: 12, marginBottom: 14 }}>
      <LossCard
        color="#8172B2"
        title="L_contrastive (NT-Xent)"
        formula="−log exp(sim(zᵢ,zⱼ)/τ) / Σ exp(sim(zᵢ,zₖ)/τ)"
        input="Joint embedding z (64d)"
        target="Same-unit codas = positive pairs"
        detail="τ=0.07 · batch=64 · 50 epochs · AdamW lr=1e-3"
      />
      <LossCard
        color="#4a90d9"
        title="L_type (Auxiliary)"
        formula="CrossEntropy(r_emb → coda_type)"
        input="Rhythm embedding r_emb"
        target="22 coda type classes"
        detail="Prevents rhythm encoder collapse to unit-only signal"
      />
      <LossCard
        color="#f5a623"
        title="L_id (Auxiliary)"
        formula="CrossEntropy(s_emb → individual_id)"
        input="Spectral embedding s_emb"
        target="12 individual whales"
        detail="762 IDN-labeled codas only · Forces speaker discrimination"
      />
    </div>

    <CrossChannelDiagram />

    <div style={{ marginTop: 12 }}>
      <InsightBox>
        <strong>Inspired by CLIP</strong> (Radford et al., 2021): cross-modal positive pairs align different views of the same concept.
        Our twist — swap spectral context across same-unit codas. SimCLR augments the <em>same</em> sample; we pair <em>different</em> speakers from the same group.
      </InsightBox>
    </div>
  </SlideLayout>
);

export default Slide19_Training;
