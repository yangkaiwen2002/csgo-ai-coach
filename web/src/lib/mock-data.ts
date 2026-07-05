import type { AnalysisResult } from "./types";

export const MOCK_ANALYSIS: AnalysisResult = {
  match_summary: {
    ct_rounds_won: 5,
    t_rounds_won: 3,
    total_rounds: 8,
    top_mistake: "Failed to rotate to B site when T pressure was detected (3 rounds)",
  },
  rounds: [
    {
      round_num: 1,
      scenario: "B Site Rush",
      ct_won: false,
      overall_assessment: "Round lost. 2 key moments detected. CTs failed to read B pressure — 3 opponents converged on B while only 2 CTs were defending.",
      key_moments: [
        {
          tick: 192,
          time_remaining: 71,
          recommended_action: "rotate_B",
          confidence: 0.87,
          top_alternatives: [["hold_B", 0.08], ["fall_back", 0.03]],
          situation_summary: "5v5 | 71s remaining",
        },
        {
          tick: 288,
          time_remaining: 47,
          recommended_action: "rotate_B",
          confidence: 0.91,
          top_alternatives: [["fall_back", 0.06], ["hold_A", 0.02]],
          situation_summary: "5v5 | 47s remaining",
        },
      ],
    },
    {
      round_num: 2,
      scenario: "A Split",
      ct_won: true,
      overall_assessment: "Round won. 1 key moment detected. Correct hold at A site prevented the split.",
      key_moments: [
        {
          tick: 240,
          time_remaining: 85,
          recommended_action: "hold_A",
          confidence: 0.79,
          top_alternatives: [["rotate_B", 0.14], ["push_A", 0.05]],
          situation_summary: "5v5 | 85s remaining",
        },
      ],
    },
    {
      round_num: 3,
      scenario: "CT Dominant Hold",
      ct_won: true,
      overall_assessment: "Round won. CT 5v3 advantage held both sites cleanly. No critical mistakes.",
      key_moments: [
        {
          tick: 160,
          time_remaining: 92,
          recommended_action: "hold_B",
          confidence: 0.72,
          top_alternatives: [["hold_A", 0.21], ["rotate_A", 0.05]],
          situation_summary: "5v3 | 92s remaining",
        },
      ],
    },
    {
      round_num: 4,
      scenario: "Eco Round",
      ct_won: true,
      overall_assessment: "Round won. T team full eco rush to B — CTs correctly held and cleaned up.",
      key_moments: [
        {
          tick: 208,
          time_remaining: 79,
          recommended_action: "hold_B",
          confidence: 0.94,
          top_alternatives: [["rotate_A", 0.04], ["push_B", 0.01]],
          situation_summary: "5v5 | 79s remaining",
        },
      ],
    },
    {
      round_num: 5,
      scenario: "B Site Rush",
      ct_won: false,
      overall_assessment: "Round lost. Same pattern as Round 1 — late B rotation cost the round. Model confidence was 88% on rotate_B at 74s remaining.",
      key_moments: [
        {
          tick: 176,
          time_remaining: 74,
          recommended_action: "rotate_B",
          confidence: 0.88,
          top_alternatives: [["hold_B", 0.07], ["smoke_B", 0.03]],
          situation_summary: "5v5 | 74s remaining",
        },
      ],
    },
    {
      round_num: 6,
      scenario: "A Split",
      ct_won: false,
      overall_assessment: "Round lost. 3-2 split at A was evenly matched but late rotation from a B player could have tipped it.",
      key_moments: [
        {
          tick: 256,
          time_remaining: 63,
          recommended_action: "rotate_A",
          confidence: 0.68,
          top_alternatives: [["hold_B", 0.22], ["push_A", 0.07]],
          situation_summary: "4v4 | 63s remaining",
        },
        {
          tick: 336,
          time_remaining: 38,
          recommended_action: "hold_A",
          confidence: 0.81,
          top_alternatives: [["fall_back", 0.12], ["rotate_B", 0.05]],
          situation_summary: "3v4 | 38s remaining",
        },
      ],
    },
    {
      round_num: 7,
      scenario: "CT Dominant Hold",
      ct_won: true,
      overall_assessment: "Round won. CT 5v3 advantage well managed.",
      key_moments: [
        {
          tick: 128,
          time_remaining: 97,
          recommended_action: "hold_A",
          confidence: 0.77,
          top_alternatives: [["hold_B", 0.18], ["push_A", 0.03]],
          situation_summary: "5v3 | 97s remaining",
        },
      ],
    },
    {
      round_num: 8,
      scenario: "Eco Round",
      ct_won: true,
      overall_assessment: "Round won. Eco read successful — CTs stacked B and held comfortably.",
      key_moments: [
        {
          tick: 144,
          time_remaining: 89,
          recommended_action: "hold_B",
          confidence: 0.92,
          top_alternatives: [["rotate_A", 0.05], ["smoke_B", 0.02]],
          situation_summary: "5v5 | 89s remaining",
        },
      ],
    },
  ],
};

export const ACTION_META: Record<string, { label: string; color: string; bg: string; description: string }> = {
  rotate_B: {
    label: "ROTATE B",
    color: "text-orange-400",
    bg: "bg-orange-400/10 border-orange-400/30",
    description: "Move CT players from A site to reinforce B",
  },
  rotate_A: {
    label: "ROTATE A",
    color: "text-orange-400",
    bg: "bg-orange-400/10 border-orange-400/30",
    description: "Move CT players from B site to reinforce A",
  },
  push_A: {
    label: "PUSH A",
    color: "text-red-400",
    bg: "bg-red-400/10 border-red-400/30",
    description: "Aggressive push toward A site",
  },
  push_B: {
    label: "PUSH B",
    color: "text-red-400",
    bg: "bg-red-400/10 border-red-400/30",
    description: "Aggressive push toward B site",
  },
  hold_A: {
    label: "HOLD A",
    color: "text-blue-400",
    bg: "bg-blue-400/10 border-blue-400/30",
    description: "Maintain defensive position at A site",
  },
  hold_B: {
    label: "HOLD B",
    color: "text-blue-400",
    bg: "bg-blue-400/10 border-blue-400/30",
    description: "Maintain defensive position at B site",
  },
  smoke_A: {
    label: "SMOKE A",
    color: "text-purple-400",
    bg: "bg-purple-400/10 border-purple-400/30",
    description: "Deploy smoke grenades at A site entry",
  },
  smoke_B: {
    label: "SMOKE B",
    color: "text-purple-400",
    bg: "bg-purple-400/10 border-purple-400/30",
    description: "Deploy smoke grenades at B site entry",
  },
  fall_back: {
    label: "FALL BACK",
    color: "text-gray-400",
    bg: "bg-gray-400/10 border-gray-400/30",
    description: "Retreat to safer position, trade space for time",
  },
};
