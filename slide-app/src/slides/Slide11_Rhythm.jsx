import SlideLayout from '../components/SlideLayout';
import { KPIBox, InsightBox, SectionLabel } from '../components/Shared';
import { theme, UNIT_COLORS } from '../theme';
import { iciByType, iciClassificationF1 } from '../data';
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, LabelList, Tooltip } from 'recharts';

/* ─── ICI Violin (real matplotlib figure) ─── */
const ICIViolin = () => (
  <div>
    <SectionLabel>Mean ICI by Social Unit</SectionLabel>
    <div style={{
      background: theme.white,
      borderRadius: 8,
      padding: '4px',
      boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
    }}>
      <img
        src="/figures/ici_violin.png"
        alt="Violin plot of ICI distributions by unit"
        style={{ width: '100%', borderRadius: 4, display: 'block' }}
      />
    </div>
  </div>
);

/* ─── ICI box-plot style by coda type ─── */
const ICIByType = () => (
  <div>
    <SectionLabel>ICI Distribution by Coda Type</SectionLabel>
    <div style={{ height: 170 }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={iciByType} layout="vertical" margin={{ top: 2, right: 35, bottom: 2, left: 40 }}>
          <XAxis type="number" tick={{ fontSize: 9 }} domain={[0, 300]} label={{ value: 'Median ICI (ms)', position: 'insideBottom', fontSize: 9, offset: -2 }} />
          <YAxis dataKey="type" type="category" tick={{ fontSize: 10, fontWeight: 600 }} width={50} />
          <Tooltip formatter={(v) => `${v} ms`} />
          <Bar dataKey="median" fill={theme.dustyBlue} radius={[0, 4, 4, 0]} barSize={14}>
            <LabelList dataKey="median" position="right" style={{ fontSize: 9, fontWeight: 600, fill: theme.textSecondary }} formatter={(v) => `${v}ms`} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  </div>
);

/* ─── t-SNE panel (real matplotlib figure) ─── */
const TSNEPanel = ({ title, subtitle, src, alt }) => (
  <div style={{ textAlign: 'center', flex: 1 }}>
    <div style={{ fontSize: 10, fontWeight: 700, color: theme.textSecondary, marginBottom: 4, textTransform: 'uppercase', letterSpacing: 0.8 }}>
      {title}
    </div>
    <div style={{
      background: theme.white,
      borderRadius: 8,
      padding: '4px',
      boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
    }}>
      <img src={src} alt={alt} style={{ width: '100%', borderRadius: 4, display: 'block' }} />
    </div>
    <div style={{ fontSize: 9, color: theme.textSecondary, marginTop: 4 }}>{subtitle}</div>
  </div>
);

/* ─── F1 bar chart ─── */
const ClassificationF1 = () => (
  <div>
    <SectionLabel>Raw ICI → Logistic Regression F1</SectionLabel>
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {iciClassificationF1.map((d) => (
        <div key={d.task} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{ fontSize: 11, fontWeight: 600, width: 90, textAlign: 'right' }}>{d.task}</div>
          <div style={{ flex: 1, height: 20, background: '#e0e0e0', borderRadius: 4, overflow: 'hidden', position: 'relative' }}>
            <div style={{
              width: `${d.f1 * 100}%`, height: '100%',
              background: d.color, borderRadius: 4,
            }} />
            <span style={{
              position: 'absolute', right: 6, top: '50%', transform: 'translateY(-50%)',
              fontSize: 11, fontWeight: 700, color: d.f1 > 0.6 ? '#fff' : theme.text,
              ...(d.f1 > 0.6 ? { right: 'auto', left: `${Math.min(d.f1 * 100 - 8, 85)}%` } : {}),
            }}>
              {d.f1.toFixed(3)}
            </span>
          </div>
        </div>
      ))}
    </div>
  </div>
);

const Slide11_Rhythm = () => (
  <SlideLayout number="11" title="The Rhythm Channel" subtitle="ICI sequences strongly encode coda type — but fail at speaker identity">
    <div style={{ display: 'flex', gap: 24, flex: 1 }}>
      {/* Left column */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 16 }}>
        <ICIViolin />
        <ICIByType />
      </div>

      {/* Center: t-SNE */}
      <div style={{ flex: 1.1, display: 'flex', flexDirection: 'column', gap: 10 }}>
        <SectionLabel>t-SNE of Standardised ICI Vectors (n=1,383)</SectionLabel>
        <div style={{ display: 'flex', gap: 12 }}>
          <TSNEPanel
            title="Coloured by Unit"
            subtitle="Units fully intermixed"
            src="/figures/tsne_by_unit.png"
            alt="t-SNE coloured by social unit"
          />
          <TSNEPanel
            title="Coloured by Coda Type"
            subtitle="Tight type clusters"
            src="/figures/tsne_by_codatype.png"
            alt="t-SNE coloured by coda type"
          />
        </div>
      </div>

      {/* Right column */}
      <div style={{ flex: 0.8, display: 'flex', flexDirection: 'column', gap: 14 }}>
        <ClassificationF1 />

        <InsightBox variant="dark">
          <strong>Key finding:</strong> Raw ICI achieves F1 = 0.931 for coda type but only 0.599 for unit and 0.493 for individual ID.
          The rhythm channel knows <em>what was said</em> but not <em>who said it</em>.
        </InsightBox>

        <InsightBox variant="light">
          <strong>Design implication:</strong> The rhythm encoder must learn <em>within-type micro-variation</em> via contrastive training
          — not just classify coda type.
        </InsightBox>
      </div>
    </div>
  </SlideLayout>
);

export default Slide11_Rhythm;
