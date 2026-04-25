import SlideLayout from '../components/SlideLayout';
import { InsightBox } from '../components/Shared';
import { theme } from '../theme';

const TakeawayCard = ({ number, color, title, body, evidence }) => (
  <div style={{
    background: theme.white,
    borderRadius: 10,
    overflow: 'hidden',
    boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
  }}>
    <div style={{ background: color, padding: '8px 14px', color: '#fff', display: 'flex', alignItems: 'center', gap: 8 }}>
      <div style={{
        width: 22, height: 22, borderRadius: '50%', background: 'rgba(255,255,255,0.25)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: 12, fontWeight: 700,
      }}>{number}</div>
      <div style={{ fontSize: 12, fontWeight: 700 }}>{title}</div>
    </div>
    <div style={{ padding: '10px 14px', flex: 1, fontSize: 11, lineHeight: 1.6 }}>
      {body}
    </div>
    <div style={{ padding: '8px 14px', background: `${color}10`, fontSize: 10, fontWeight: 600, color: theme.textSecondary }}>
      📊 {evidence}
    </div>
  </div>
);

const LimitationPill = ({ text }) => (
  <div style={{
    background: theme.bgLight,
    borderRadius: 6,
    padding: '6px 12px',
    fontSize: 10,
    fontWeight: 500,
    color: theme.textSecondary,
    lineHeight: 1.4,
  }}>
    {text}
  </div>
);

const Slide25_Discussion = () => (
  <SlideLayout number="34" title="Discussion & Takeaways" subtitle="What we learned, what it means, and what's left">
    {/* Top: 3 main takeaways */}
    <div style={{ display: 'flex', gap: 12, marginBottom: 14 }}>
      <TakeawayCard
        number="1"
        color="#8172B2"
        title="Domain Knowledge > Scale"
        body="DCCE's +0.380 indivID gain comes entirely from encoding known biology — not more data or bigger models. 6.7× less data, laptop-scale compute."
        evidence="0.834 vs 0.454 F1 on individual ID"
      />
      <TakeawayCard
        number="2"
        color={theme.unitA}
        title="Year Confound Matters"
        body="WhAM's unit advantage is partly driven by recording-year artifacts (V=0.51). DCCE uses recording-independent features and achieves near-parity (0.878 vs 0.895)."
        evidence="Cramér's V = 0.51, ρ = 0.63, p = 0.003"
      />
      <TakeawayCard
        number="3"
        color="#d62728"
        title="Fidelity ≠ Utility"
        body="WhAM generates FAD-indistinguishable codas, but synthetic augmentation fails. Augmentation must add task-relevant variation, not just acoustic realism."
        evidence="All metrics decline at N_synth = 1000"
      />
    </div>

    {/* Biological implications */}
    <div style={{
      background: theme.white,
      borderRadius: 10,
      padding: '12px 18px',
      boxShadow: '0 1px 4px rgba(0,0,0,0.06)',
      marginBottom: 12,
    }}>
      <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1, color: theme.textSecondary, marginBottom: 8 }}>
        Biological Implications
      </div>
      <div style={{ display: 'flex', gap: 16 }}>
        <div style={{ flex: 1, fontSize: 11, lineHeight: 1.5 }}>
          🐋 <strong>Individual identity is robustly encoded</strong> — a linear classifier on 64d embeddings achieves F1 = 0.834 for speaker ID.
        </div>
        <div style={{ flex: 1, fontSize: 11, lineHeight: 1.5 }}>
          🎵 <strong>Rhythm suffices for type</strong> — ICI F1 = 0.931 on coda type, but only 0.509 on individual ID. Spectral texture carries the identity signal.
        </div>
        <div style={{ flex: 1, fontSize: 11, lineHeight: 1.5 }}>
          🔗 <strong>Multi-level identity</strong> — unit and individual are simultaneously decodable from the same 64d embedding, consistent with biological hierarchy.
        </div>
      </div>
    </div>

    {/* Limitations */}
    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
      <LimitationPill text="⚠ Single population (Dominica, 3 units, 2005–2010)" />
      <LimitationPill text="⚠ No vowel labels for DSWP range" />
      <LimitationPill text="⚠ Pseudo-ICI for synthetic codas" />
      <LimitationPill text="⚠ Laptop-scale compute — scaling unexplored" />
    </div>
  </SlideLayout>
);

export default Slide25_Discussion;
