import SlideLayout from '../components/SlideLayout';
import { KPIBox, InsightBox } from '../components/Shared';
import { theme } from '../theme';
import { whamProbingLayers } from '../data';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, ReferenceLine } from 'recharts';

const COLORS = {
  unit: theme.unitA,
  type: '#2ca02c',
  indivID: '#d62728',
  year: theme.periwinkle,
};

const Slide20_WhamProbing = () => (
  <SlideLayout number="25" title="Inside WhAM's Brain" subtitle="Layer-by-layer probing: what does each transformer layer encode?">
    <div style={{ display: 'flex', gap: 20, flex: 1 }}>
      {/* Chart */}
      <div style={{ flex: 2 }}>
        <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1, color: theme.textSecondary, marginBottom: 6 }}>
          Linear Probe F1 by Layer (20 layers × 1,280d)
        </div>
        <div style={{ height: 340 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={whamProbingLayers} margin={{ top: 10, right: 20, bottom: 10, left: 10 }}>
              <XAxis dataKey="layer" tick={{ fontSize: 10 }} label={{ value: 'Layer', position: 'insideBottom', offset: -4, fontSize: 10 }} />
              <YAxis domain={[0, 1]} tick={{ fontSize: 10 }} label={{ value: 'Macro-F1', angle: -90, position: 'insideLeft', offset: 10, fontSize: 10 }} />
              <Tooltip formatter={(v) => v.toFixed(3)} />
              <Legend wrapperStyle={{ fontSize: 10 }} />
              <ReferenceLine y={0.931} stroke="#2ca02c" strokeDasharray="3 3" label={{ value: 'Raw ICI type=0.931', position: 'right', fontSize: 8, fill: '#2ca02c' }} />
              <Line type="monotone" dataKey="unit" name="Social Unit" stroke={COLORS.unit} strokeWidth={2.5} dot={false} />
              <Line type="monotone" dataKey="year" name="Rec. Year" stroke={COLORS.year} strokeWidth={2} dot={false} strokeDasharray="6 3" />
              <Line type="monotone" dataKey="indivID" name="Individual ID" stroke={COLORS.indivID} strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="type" name="Coda Type" stroke={COLORS.type} strokeWidth={1.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Right panel */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 10 }}>
        <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1, color: theme.textSecondary }}>
          Peak performance per target
        </div>
        <KPIBox value="0.895" label="Unit (L19)" color={COLORS.unit} />
        <KPIBox value="0.906" label="Year (L18)" color={COLORS.year} />
        <KPIBox value="0.454" label="Indiv ID (L10)" color={COLORS.indivID} />
        <KPIBox value="< 0.26" label="Coda Type (all)" color={COLORS.type} />

        <InsightBox variant="red">
          <strong>⚠ Year F1 ≈ Unit F1.</strong><br />
          WhAM may be learning recording conditions, not whale voices.
        </InsightBox>
      </div>
    </div>

    <InsightBox>
      <strong>Three key findings:</strong> (1) Unit signal rises monotonically → WhAM is designed around group identity.
      (2) Individual ID peaks at L10 then degrades → overwritten by unit-level pressure.
      (3) Coda type is absent → WhAM's tokenization destroys ICI timing.
    </InsightBox>
  </SlideLayout>
);

export default Slide20_WhamProbing;
