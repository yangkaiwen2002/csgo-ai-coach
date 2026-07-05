import type { Round } from "@/lib/types";
import ActionBadge from "./ActionBadge";

interface Props {
  round: Round;
  selected: boolean;
  onClick: () => void;
}

export default function RoundRow({ round, selected, onClick }: Props) {
  const topAction = round.key_moments[0]?.recommended_action;

  return (
    <button
      onClick={onClick}
      className={`w-full text-left px-3 py-3 rounded-lg transition-all duration-150 group
        ${selected
          ? "bg-white/8 border border-white/12"
          : "border border-transparent hover:bg-white/4"
        }
      `}
    >
      <div className="flex items-center justify-between mb-1">
        <div className="flex items-center gap-2">
          <span
            className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${
              round.ct_won ? "bg-emerald-400" : "bg-red-400"
            }`}
          />
          <span className="text-xs font-mono text-gray-400">
            R{round.round_num}
          </span>
          <span
            className={`text-xs font-semibold ${
              round.ct_won ? "text-emerald-400" : "text-red-400"
            }`}
          >
            {round.ct_won ? "WIN" : "LOSS"}
          </span>
        </div>
        <span className="text-xs text-gray-600">
          {round.key_moments.length} moment{round.key_moments.length !== 1 ? "s" : ""}
        </span>
      </div>
      <p className="text-xs text-gray-500 pl-3.5 truncate">{round.scenario}</p>
      {topAction && (
        <div className="pl-3.5 mt-1.5">
          <ActionBadge action={topAction} size="sm" />
        </div>
      )}
    </button>
  );
}
