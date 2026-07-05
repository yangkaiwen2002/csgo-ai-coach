interface Props {
  value: number; // 0–1
  showLabel?: boolean;
}

export default function ConfidenceBar({ value, showLabel = true }: Props) {
  const pct = Math.round(value * 100);

  const barColor =
    pct >= 80
      ? "bg-emerald-400"
      : pct >= 60
      ? "bg-yellow-400"
      : "bg-red-400";

  return (
    <div className="flex items-center gap-3">
      <div className="flex-1 h-1.5 bg-white/10 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ${barColor}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      {showLabel && (
        <span className="text-xs font-mono text-gray-400 w-9 text-right">
          {pct}%
        </span>
      )}
    </div>
  );
}
