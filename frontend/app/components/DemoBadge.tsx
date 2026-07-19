type Props = { children?: React.ReactNode };

export default function DemoBadge({ children = "샘플 데이터" }: Props) {
  return (
    <span
      style={{
        display: "inline-block",
        marginLeft: 8,
        fontSize: 10,
        padding: "2px 6px",
        borderRadius: 4,
        border: "1px solid var(--border)",
        color: "var(--sub)",
        letterSpacing: "0.05em",
        textTransform: "uppercase",
        verticalAlign: "middle",
      }}
    >
      {children}
    </span>
  );
}
