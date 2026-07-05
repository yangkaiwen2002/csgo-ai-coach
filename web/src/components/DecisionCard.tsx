import type { KeyMoment } from "@/lib/types";
import { ACTION_META } from "@/lib/mock-data";
import ActionBadge from "./ActionBadge";
import ConfidenceBar from "./ConfidenceBar";

interface Props {
  moment: KeyMoment;
  index: number;
}

export default function DecisionCard({ moment, index }: Props) {
  const meta = ACTION_META[moment.recommended_action];

  return (
    <div className="group relative rounded-lg border border-white/8 bg-white/3 hover:border-white/15 hover:bg-white/5 transition-all duration-200 p-5">
      {/* Subtle left accent */}
      <div
        className={`absolute left-0 top-4 bottom-4 w-0.5 rounded-full opacity-60 ${
          meta?.color.replace("text-", "bg-") ?? "bg-gray-400"
        }`}
      />

      <div className="flex items-start justify-between gap-4 mb-4">
        <div className="flex items-center gap-3">
          <span className="text-xs font-mono text-gray-600 w-5">
            #{index + 1}
          </span>
          <div>
            <p className="text-xs text-gray-500 font-mono mb-1">
              TICK {moment.tick} &nbsp;·&nbsp; {Math.round(moment.time_remaining)}s remaining
            </p>
            <p className="text-sm text-gray-400">{moment.situation_summary}</p>
          </div>
        </div>
        <ActionBadge action={moment.recommended_action} size="sm" />
      </div>

      {/* Confidence */}
      <div className="mb-4">
        <div className="flex items-center justify-between mb-1.5">
          <span className="text-xs text-gray-500">Model confidence</span>
          <span className="text-xs font-mono text-gray-300">
            {Math.round(moment.confidence * 100)}%
          </span>
        </div>
        <ConfidenceBar value={moment.confidence} showLabel={false} />
      </div>

      {/* Description */}
      {meta && (
        <p className="text-sm text-gray-500 mb-4">{meta.description}</p>
      )}

      {/* Alternatives */}
      <div>
        <p className="text-xs text-gray-600 mb-2">Other considered actions</p>
        <div className="flex flex-wrap gap-2">
          {moment.top_alternatives.map(([action, prob]) => (
            <div key={action} className="flex items-center gap-1.5">
              <ActionBadge action={action} size="sm" />
              <span className="text-xs font-mono text-gray-600">
                {Math.round(prob * 100)}%
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
