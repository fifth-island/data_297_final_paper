import { theme } from '../theme';

const SlideLayout = ({ number, title, subtitle, children }) => (
  <div style={{
    width: '1280px',
    height: '720px',
    background: theme.bg,
    fontFamily: theme.fontBody,
    color: theme.text,
    position: 'relative',
    overflow: 'hidden',
    boxSizing: 'border-box',
    padding: '48px 56px 40px',
    display: 'flex',
    flexDirection: 'column',
  }}>
    {/* Slide number */}
    <div style={{
      position: 'absolute',
      top: 16,
      right: 24,
      fontSize: 13,
      color: theme.textSecondary,
      fontWeight: 500,
    }}>
      {number}
    </div>

    {/* Top divider */}
    <div style={{
      width: '100%',
      height: 2,
      background: theme.text,
      marginBottom: 20,
    }} />

    {/* Title */}
    <h1 style={{
      fontFamily: theme.fontHeader,
      fontSize: 32,
      fontWeight: 700,
      margin: '0 0 6px',
      lineHeight: 1.2,
      letterSpacing: '-0.5px',
    }}>
      {title}
    </h1>

    {/* Subtitle */}
    {subtitle && (
      <p style={{
        fontSize: 15,
        color: theme.textSecondary,
        fontWeight: 500,
        margin: '0 0 24px',
        lineHeight: 1.4,
      }}>
        {subtitle}
      </p>
    )}

    {/* Content */}
    <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
      {children}
    </div>
  </div>
);

export default SlideLayout;
