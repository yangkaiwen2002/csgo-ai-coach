export interface KeyMoment {
  tick: number;
  time_remaining: number;
  recommended_action: string;
  confidence: number;
  top_alternatives: [string, number][];
  situation_summary: string;
}

export interface Round {
  round_num: number;
  scenario: string;
  ct_won: boolean;
  overall_assessment: string;
  key_moments: KeyMoment[];
}

export interface AnalysisResult {
  rounds: Round[];
  match_summary: {
    ct_rounds_won: number;
    t_rounds_won: number;
    total_rounds: number;
    top_mistake: string;
  };
}

export type Action =
  | "push_A" | "push_B"
  | "hold_A" | "hold_B"
  | "rotate_A" | "rotate_B"
  | "smoke_A" | "smoke_B"
  | "fall_back";
