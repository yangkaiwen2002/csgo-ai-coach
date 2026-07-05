import { ACTION_META } from "@/lib/mock-data";

interface Props {
  action: string;
  size?: "sm" | "md" | "lg";
}

export default function ActionBadge({ action, size = "md" }: Props) {
  const meta = ACTION_META[action] ?? {
    label: action.toUpperCase().replace("_", " "),
    color: "text-gray-400",
    bg: "bg-gray-400/10 border-gray-400/30",
    description: "",
  };

  const sizeClass =
    size === "sm"
      ? "text-xs px-2 py-0.5"
      : size === "lg"
      ? "text-base px-4 py-1.5 font-bold tracking-widest"
      : "text-sm px-3 py-1 font-semibold tracking-wider";

  return (
    <span
      className={`inline-flex items-center border rounded font-mono ${sizeClass} ${meta.color} ${meta.bg}`}
    >
      {meta.label}
    </span>
  );
}
