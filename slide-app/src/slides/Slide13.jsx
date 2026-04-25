import React from 'react';
import SlideLayout from '../components/SlideLayout';
import { InsightBox } from '../components/Shared';
import { theme, UNIT_COLORS_TEXT } from '../theme';
import { codaTypeByUnit } from '../data';

/* ── colour constants ── */
// Seaborn-style unit colors (matching rhythm/spectral slides)
const UNIT_HEX = { A: '#4c72b0', D: '#dd8452', F: '#55a868' };

// Sharing category colors: slate blue / periwinkle / yellow (theme palette)
const SHARING_COLORS = {
  all: theme.unitA,       // slate blue
  partial: theme.unitD,   // periwinkle
  exclusive: theme.unitF, // yellow
};
const SHARING_TEXT_COLORS = {
  all: theme.unitA,
  partial: theme.unitD,
  exclusive: theme.unitFText,
};
const SHARING_LABELS = { all: 'All 3 units', partial: '2 units', exclusive: '1 unit only' };

/* ── Inline proportional bar for each row ── */
const ProportionBar = ({ A, D, F }) => {
  const total = A + D + F;
  if (total === 0) return null;
  const pA = (A / total) * 100;
  const pD = (D / total) * 100;
  const pF = (F / total) * 100;
  return (
    <div style={{
      display: 'flex',
      height: 8,
      borderRadius: 4,
      overflow: 'hidden',
      width: '100%',
    }}>
      {pA > 0 && <div style={{ width: `${pA}%`, background: UNIT_HEX.A }} />}
      {pD > 0 && <div style={{ width: `${pD}%`, background: UNIT_HEX.D }} />}
      {pF > 0 && <div style={{ width: `${pF}%`, background: UNIT_HEX.F }} />}
    </div>
  );
};

/* ── Count cell with unit-coloured background ── */
const CountCell = ({ value, maxVal, unit }) => {
  const opacity = maxVal > 0 ? Math.max(0.03, value / maxVal) : 0;
  const hex = UNIT_HEX[unit];
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return (
    <div style={{
      background: value > 0 ? `rgba(${r},${g},${b},${opacity * 0.5})` : 'transparent',
      padding: '4px 0',
      textAlign: 'center',
      fontSize: 13,
      fontWeight: value > 0 ? 600 : 400,
      color: value > 0 ? theme.text : theme.textSecondary,
      fontVariantNumeric: 'tabular-nums',
    }}>
      {value > 0 ? value : '—'}
    </div>
  );
};

/* ── Section header row spanning the full grid ── */
const SectionHeader = ({ label, color }) => (
  <div style={{
    gridColumn: '1 / -1',
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    padding: '6px 0 2px 0',
  }}>
    <div style={{ width: 8, height: 8, borderRadius: '50%', background: color, flexShrink: 0 }} />
    <span style={{
      fontSize: 10.5,
      fontWeight: 700,
      textTransform: 'uppercase',
      letterSpacing: 0.8,
      color: theme.textSecondary,
    }}>
      {label}
    </span>
    <div style={{ flex: 1, height: 1, background: `${color}44` }} />
  </div>
);

/* ── Small KPI stat ── */
const KPIStat = ({ value, label, color }) => (
  <div style={{ textAlign: 'center', flex: 1 }}>
    <div style={{
      fontFamily: theme.fontHeader,
      fontWeight: 700,
      fontSize: 22,
      lineHeight: 1,
      color: color || theme.text,
    }}>
      {value}
    </div>
    <div style={{ fontSize: 9, color: theme.textSecondary, marginTop: 4, lineHeight: 1.2 }}>{label}</div>
  </div>
);

/* ── Main slide ── */
const Slide13 = () => {
  const globalMax = Math.max(...codaTypeByUnit.map(r => Math.max(r.A, r.D, r.F)));

  // Compute stats from data
  const totalCodas = codaTypeByUnit.reduce((s, r) => s + r.A + r.D + r.F, 0);
  const totalA = codaTypeByUnit.reduce((s, r) => s + r.A, 0);
  const totalD = codaTypeByUnit.reduce((s, r) => s + r.D, 0);
  const totalF = codaTypeByUnit.reduce((s, r) => s + r.F, 0);
  const top4Total = codaTypeByUnit.slice(0, 4).reduce((s, r) => s + r.A + r.D + r.F, 0);
  const top4Pct = Math.round((top4Total / totalCodas) * 100);

  const groups = [
    { key: 'all', label: 'Shared by all 3 units', rows: codaTypeByUnit.filter(r => r.sharing === 'all') },
    { key: 'partial', label: 'Shared by 2 units', rows: codaTypeByUnit.filter(r => r.sharing === 'partial') },
    { key: 'exclusive', label: 'Unit-exclusive', rows: codaTypeByUnit.filter(r => r.sharing === 'exclusive') },
  ];

  return (
    <SlideLayout number="15" title="The Shared Vocabulary" subtitle="Coda types are clan property, not unit property">
      <div style={{ display: 'flex', gap: 24, flex: 1, minHeight: 0, overflow: 'hidden' }}>

        {/* ── Left column: full-height heatmap table ── */}
        <div style={{
          flex: 1,
          display: 'grid',
          gridTemplateColumns: '85px 78px 78px 78px 140px',
          alignContent: 'start',
          rowGap: 0,
          columnGap: 2,
        }}>
          {/* Column headers */}
          <div style={{ fontSize: 13, fontWeight: 700, padding: '0 0 6px 2px', color: theme.textSecondary }}>
            Coda Type
          </div>
          {['A', 'D', 'F'].map(u => (
            <div key={u} style={{
              fontSize: 13,
              fontWeight: 700,
              textAlign: 'center',
              padding: '0 0 6px 0',
              color: UNIT_HEX[u],
            }}>
              Unit {u}
            </div>
          ))}
          <div style={{ fontSize: 13, fontWeight: 700, padding: '0 0 6px 6px', color: theme.textSecondary }}>
            Proportion
          </div>

          {/* Grouped rows */}
          {groups.map(group => (
            group.rows.length > 0 && (
              <React.Fragment key={group.key}>
                <SectionHeader label={group.label} color={SHARING_COLORS[group.key]} />
                {group.rows.map(row => (
                  <React.Fragment key={row.type}>
                    <div style={{
                      padding: '4px 2px',
                      fontWeight: 600,
                      fontSize: 13,
                      fontFamily: theme.fontBody,
                      borderBottom: '0.5px solid rgba(0,0,0,0.04)',
                      display: 'flex',
                      alignItems: 'center',
                    }}>
                      {row.type}
                    </div>
                    {['A', 'D', 'F'].map(u => (
                      <div key={u} style={{ borderBottom: '0.5px solid rgba(0,0,0,0.04)' }}>
                        <CountCell value={row[u]} maxVal={globalMax} unit={u} />
                      </div>
                    ))}
                    <div style={{
                      padding: '3px 6px',
                      display: 'flex',
                      alignItems: 'center',
                      borderBottom: '0.5px solid rgba(0,0,0,0.04)',
                    }}>
                      <ProportionBar A={row.A} D={row.D} F={row.F} />
                    </div>
                  </React.Fragment>
                ))}
              </React.Fragment>
            )
          ))}
        </div>

        {/* ── Right area: two sub-columns ── */}
        <div style={{ flex: 1, display: 'flex', gap: 14, minWidth: 0 }}>

          {/* Sub-column A */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 8 }}>

            {/* Sharing summary card */}
            <div style={{
              background: theme.white,
              borderRadius: 10,
              padding: '12px 14px',
              boxShadow: '0 1px 4px rgba(0,0,0,0.06)',
            }}>
              <div style={{
                fontSize: 10,
                fontWeight: 700,
                textTransform: 'uppercase',
                letterSpacing: 1,
                color: theme.textSecondary,
                marginBottom: 10,
              }}>
                Sharing Summary
              </div>
              {[
                { key: 'all', count: 9, note: 'clan vocabulary' },
                { key: 'partial', count: 6, note: 'partial overlap' },
                { key: 'exclusive', count: 5, note: 'unit markers?' },
              ].map(s => (
                <div key={s.key} style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  marginBottom: 6,
                }}>
                  <div style={{
                    width: 11,
                    height: 11,
                    borderRadius: '50%',
                    background: SHARING_COLORS[s.key],
                    flexShrink: 0,
                  }} />
                  <span style={{ fontSize: 11.5, flex: 1 }}>{SHARING_LABELS[s.key]}</span>
                  <span style={{
                    fontFamily: theme.fontHeader,
                    fontWeight: 700,
                    fontSize: 20,
                    lineHeight: 1,
                    color: SHARING_TEXT_COLORS[s.key],
                  }}>
                    {s.count}
                  </span>
                </div>
              ))}
            </div>

            {/* Unit totals mini bar chart */}
            <div style={{
              background: theme.white,
              borderRadius: 10,
              padding: '10px 14px',
              boxShadow: '0 1px 4px rgba(0,0,0,0.06)',
            }}>
              <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1, color: theme.textSecondary, marginBottom: 8 }}>
                Codas per unit
              </div>
              {[
                { unit: 'A', count: totalA },
                { unit: 'D', count: totalD },
                { unit: 'F', count: totalF },
              ].map(u => (
                <div key={u.unit} style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 5 }}>
                  <span style={{ fontSize: 10.5, fontWeight: 700, color: UNIT_HEX[u.unit], width: 38 }}>Unit {u.unit}</span>
                  <div style={{ flex: 1, height: 10, background: 'rgba(0,0,0,0.04)', borderRadius: 5, overflow: 'hidden' }}>
                    <div style={{
                      width: `${(u.count / totalF) * 100}%`,
                      height: '100%',
                      background: UNIT_HEX[u.unit],
                      borderRadius: 5,
                    }} />
                  </div>
                  <span style={{ fontSize: 10.5, fontWeight: 600, fontVariantNumeric: 'tabular-nums', width: 28, textAlign: 'right' }}>{u.count}</span>
                </div>
              ))}
            </div>

            {/* Key patterns annotations */}
            <div style={{
              background: theme.bgLight,
              borderRadius: 10,
              padding: '10px 14px',
              display: 'flex',
              flexDirection: 'column',
              gap: 6,
              flex: 1,
            }}>
              <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1, color: theme.textSecondary, marginBottom: 2 }}>
                Key patterns
              </div>
              {[
                { label: '1+1+3', text: 'Clan signature — equal use across all units' },
                { label: '6D', text: 'Unit F only — potential unit marker (35 codas)' },
                { label: '4D', text: 'Shared but uneven — usage differs by unit' },
              ].map(a => (
                <div key={a.label} style={{ fontSize: 11, lineHeight: 1.4 }}>
                  <span style={{ fontWeight: 700, color: theme.text }}>{a.label}</span>
                  <span style={{ color: theme.textSecondary }}> — {a.text}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Sub-column B */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 8 }}>

            {/* KPI stats */}
            <div style={{
              background: theme.white,
              borderRadius: 10,
              padding: '12px 14px',
              boxShadow: '0 1px 4px rgba(0,0,0,0.06)',
              display: 'flex',
              flexDirection: 'column',
              gap: 10,
            }}>
              <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1, color: theme.textSecondary }}>
                At a glance
              </div>
              {[
                { value: codaTypeByUnit.length, label: 'Coda types analysed', color: theme.text },
                { value: `${top4Pct}%`, label: 'Covered by top 4 types', color: theme.unitA },
                { value: totalCodas.toLocaleString(), label: 'Total codas (top 15)', color: theme.text },
              ].map((kpi, i) => (
                <div key={i} style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
                  <span style={{
                    fontFamily: theme.fontHeader,
                    fontWeight: 700,
                    fontSize: 22,
                    lineHeight: 1,
                    color: kpi.color,
                  }}>
                    {kpi.value}
                  </span>
                  <span style={{ fontSize: 10, color: theme.textSecondary, lineHeight: 1.2 }}>{kpi.label}</span>
                </div>
              ))}
            </div>

            {/* Why this matters for modelling */}
            <div style={{
              background: theme.white,
              borderRadius: 10,
              padding: '10px 14px',
              boxShadow: '0 1px 4px rgba(0,0,0,0.06)',
              flex: 1,
            }}>
              <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1, color: theme.textSecondary, marginBottom: 8 }}>
                Why this matters
              </div>
              {[
                { icon: '→', text: 'Coda type alone cannot distinguish units — the vocabulary is shared' },
                { icon: '→', text: 'Exclusive types (6D, 9, 8) are too rare to train a classifier on' },
                { icon: '→', text: 'Even shared types have uneven usage — a frequency signal exists but is weak' },
                { icon: '→', text: 'Identity must come from how the coda is spoken, not which coda is spoken' },
              ].map((item, i) => (
                <div key={i} style={{ fontSize: 11, lineHeight: 1.4, marginBottom: 6, display: 'flex', gap: 6 }}>
                  <span style={{ color: theme.unitA, fontWeight: 700, flexShrink: 0 }}>{item.icon}</span>
                  <span style={{ color: theme.textSecondary }}>{item.text}</span>
                </div>
              ))}
            </div>

            {/* Insight box */}
            <InsightBox>
              9 of 20 coda types appear in all 3 units. A coda-type-only model sees the same vocabulary everywhere. Identity is not in the type — it is in the voice.
            </InsightBox>
          </div>
        </div>
      </div>
    </SlideLayout>
  );
};

export default Slide13;
