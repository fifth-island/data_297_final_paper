import SlideLayout from '../components/SlideLayout';
import { InsightBox, KPIBox } from '../components/Shared';
import { theme } from '../theme';
import { augmentationSweep } from '../data';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, ReferenceLine } from 'recharts';

const TASK_COLORS = {
  unit: theme.unitA,
  type: '#2ca02c',
  indivID: '#d62728',
};

const FailureReason = ({ icon, title, detail }) => (
  <div style={{
    background: theme.white,
    borderRadius: 8,
    padding: '10px 14px',
    boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
    flex: 1,
  }}>
    <div style={{ fontSize: 18, marginBottom: 4 }}>{icon}</div>
    <div style={{ fontSize: 11, fontWeight: 700, marginBottom: 4 }}>{title}</div>
    <div style={{ fontSize: 10, color: theme.textSecondary, lineHeight: 1.5 }}>{detail}</div>
  </div>
);

const Slide24_Augmentation = () => {
  const chartData = augmentationSweep.map((d) => ({
    ...d,
    unitPct: +(d.unit * 100).toFixed(1),
    typePct: +(d.type * 100).toFixed(1),
    idPct: +(d.indivID * 100).toFixed(1),
  }));

  return (
    <SlideLayout number="33" title="Synthetic Augmentation" subtitle="Does WhAM-generated audio improve DCCE? No.">
      <div style={{ display: 'flex', gap: 20, flex: 1 }}>
        {/* Chart */}
        <div style={{ flex: 1.3 }}>
          <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1, color: theme.textSecondary, marginBottom: 6 }}>
            DCCE-full F1 vs. N_synth added to training
          </div>
          <div style={{ height: 280 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 10, right: 20, bottom: 10, left: 10 }}>
                <XAxis dataKey="label" tick={{ fontSize: 10 }} label={{ value: 'N_synth', position: 'insideBottom', offset: -4, fontSize: 10 }} />
                <YAxis domain={[45, 92]} tick={{ fontSize: 9 }} label={{ value: 'F1 (%)', angle: -90, position: 'insideLeft', offset: 10, fontSize: 10 }} />
                <Tooltip formatter={(v) => `${v}%`} />
                <Legend wrapperStyle={{ fontSize: 10 }} />
                <Line type="monotone" dataKey="unitPct" name="Social Unit" stroke={TASK_COLORS.unit} strokeWidth={2.5} dot={{ r: 4 }} />
                <Line type="monotone" dataKey="typePct" name="Coda Type" stroke={TASK_COLORS.type} strokeWidth={2} dot={{ r: 4 }} />
                <Line type="monotone" dataKey="idPct" name="Individual ID" stroke={TASK_COLORS.indivID} strokeWidth={2.5} dot={{ r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Deltas + reasons */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 10 }}>
          <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1, color: theme.textSecondary }}>
            Δ F1 at N=1000 vs baseline
          </div>
          <div style={{ display: 'flex', gap: 8, marginBottom: 4 }}>
            <KPIBox value="−0.009" label="Unit" color={theme.textSecondary} />
            <KPIBox value="−0.032" label="Type" color={theme.textSecondary} />
            <KPIBox value="−0.051" label="IndivID" color={theme.red} />
          </div>

          <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1, color: theme.textSecondary, marginTop: 4 }}>
            Why it fails
          </div>
          <FailureReason
            icon="🔄"
            title="Pseudo-ICI (copied)"
            detail="ICI is copied from prompt — adds zero new rhythm information to the rhythm encoder."
          />
          <FailureReason
            icon="🚫"
            title="No individual ID labels"
            detail="L_id can't train on synthetics. Spectral encoder gets no speaker-identity supervision."
          />
          <FailureReason
            icon="💧"
            title="Contrastive dilution"
            detail="Synthetics enter unit-level contrastive loss but dilute the individual-level geometry."
          />
        </div>
      </div>

      <InsightBox variant="red">
        <strong>Key insight: Acoustic fidelity ≠ representational benefit.</strong> WhAM produces FAD-indistinguishable codas (expert 2AFC = 81%),
        yet they provide zero benefit. Augmentation quality must match the <em>specific downstream loss</em>, not just sound realistic.
      </InsightBox>
    </SlideLayout>
  );
};

export default Slide24_Augmentation;
