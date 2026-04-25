import { theme } from '../theme';

export const KPIBox = ({ value, label, color }) => (
  <div style={{
    background: theme.white,
    borderRadius: 10,
    padding: '16px 20px',
    textAlign: 'center',
    minWidth: 120,
    boxShadow: '0 1px 4px rgba(0,0,0,0.06)',
  }}>
    <div style={{
      fontFamily: theme.fontHeader,
      fontSize: 28,
      fontWeight: 700,
      color: color || theme.text,
      lineHeight: 1.1,
    }}>
      {value}
    </div>
    <div style={{
      fontSize: 11,
      color: theme.textSecondary,
      fontWeight: 600,
      marginTop: 4,
      textTransform: 'uppercase',
      letterSpacing: '0.5px',
    }}>
      {label}
    </div>
  </div>
);

export const InsightBox = ({ children, variant = 'dark' }) => {
  const isDark = variant === 'dark';
  return (
    <div style={{
      background: isDark ? theme.text : variant === 'green' ? '#e8f5e9' : variant === 'red' ? '#fce4ec' : theme.bgLight,
      color: isDark ? theme.white : theme.text,
      borderRadius: 8,
      padding: '12px 18px',
      fontSize: 13,
      lineHeight: 1.5,
      fontWeight: 500,
    }}>
      {children}
    </div>
  );
};

export const UnitBadge = ({ unit, size = 'md' }) => {
  const colors = { A: theme.unitA, D: theme.unitD, F: theme.unitF };
  const sizes = { sm: { px: 8, py: 3, fs: 10 }, md: { px: 12, py: 4, fs: 12 }, lg: { px: 16, py: 6, fs: 14 } };
  const s = sizes[size];
  return (
    <span style={{
      background: colors[unit],
      color: '#fff',
      borderRadius: 4,
      padding: `${s.py}px ${s.px}px`,
      fontSize: s.fs,
      fontWeight: 700,
      display: 'inline-block',
    }}>
      Unit {unit}
    </span>
  );
};

export const SectionLabel = ({ children }) => (
  <div style={{
    fontSize: 11,
    fontWeight: 700,
    textTransform: 'uppercase',
    letterSpacing: '1.2px',
    color: theme.textSecondary,
    marginBottom: 8,
  }}>
    {children}
  </div>
);
