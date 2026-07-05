interface Props {
  label: string;
  value: string | number;
  sub?: string;
  accent?: boolean;
}

export default function StatCard({ label, value, sub, accent }: Props) {
  return (
    <div className="rounded-lg border border-white/8 bg-white/3 px-4 py-3">
      <p className="text-xs text-gray-500 mb-1 uppercase tracking-wider">{label}</p>
      <p
        className={`text-2xl font-bold font-mono ${
          accent ? "text-orange-400" : "text-white"
        }`}
      >
        {value}
      </p>
      {sub && <p className="text-xs text-gray-600 mt-0.5">{sub}</p>}
    </div>
  );
}
