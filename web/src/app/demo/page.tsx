"use client";

import { useState } from "react";
import Link from "next/link";
import { MOCK_ANALYSIS } from "@/lib/mock-data";
import RoundRow from "@/components/RoundRow";
import DecisionCard from "@/components/DecisionCard";
import StatCard from "@/components/StatCard";
import ActionBadge from "@/components/ActionBadge";
import Navbar from "@/components/Navbar";

export default function DemoPage() {
  const [selectedRound, setSelectedRound] = useState(0);
  const { rounds, match_summary } = MOCK_ANALYSIS;
  const round = rounds[selectedRound];

  const winRate = Math.round(
    (match_summary.ct_rounds_won / match_summary.total_rounds) * 100
  );

  return (
    <div className="min-h-screen bg-[#06060a] text-gray-100">
      <Navbar />

      <div className="pt-14 h-screen flex flex-col">
        {/* Top bar */}
        <div className="border-b border-white/6 px-6 py-3 flex items-center justify-between bg-[#06060a]">
          <div className="flex items-center gap-4">
            <Link
              href="/"
              className="text-sm text-gray-500 hover:text-gray-300 transition-colors flex items-center gap-1.5"
            >
              <svg width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
              </svg>
              Back
            </Link>
            <div className="w-px h-4 bg-white/10" />
            <span className="text-sm text-gray-300 font-medium">
              de_dust2 — Demo Match
            </span>
            <span className="text-xs text-gray-600">CT perspective</span>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-xs text-gray-500">
              {match_summary.ct_rounds_won}W – {match_summary.t_rounds_won}L
            </span>
            <span
              className={`text-xs font-mono px-2 py-0.5 rounded ${
                winRate >= 50
                  ? "bg-emerald-400/10 text-emerald-400"
                  : "bg-red-400/10 text-red-400"
              }`}
            >
              {winRate}% win rate
            </span>
          </div>
        </div>

        {/* Main layout */}
        <div className="flex flex-1 overflow-hidden">
          {/* Sidebar */}
          <div className="w-56 border-r border-white/6 flex flex-col bg-[#07070c]">
            <div className="px-3 py-3 border-b border-white/6">
              <p className="text-xs text-gray-600 uppercase tracking-wider">Rounds</p>
            </div>
            <div className="flex-1 overflow-y-auto px-2 py-2 space-y-0.5">
              {rounds.map((r, i) => (
                <RoundRow
                  key={r.round_num}
                  round={r}
                  selected={i === selectedRound}
                  onClick={() => setSelectedRound(i)}
                />
              ))}
            </div>
            {/* Sidebar summary */}
            <div className="border-t border-white/6 px-3 py-3 space-y-1">
              <p className="text-xs text-gray-600">Top mistake pattern</p>
              <p className="text-xs text-gray-400 leading-relaxed">
                {match_summary.top_mistake}
              </p>
            </div>
          </div>

          {/* Main content */}
          <div className="flex-1 overflow-y-auto">
            <div className="max-w-3xl mx-auto px-8 py-8">
              {/* Round header */}
              <div className="flex items-start justify-between mb-6">
                <div>
                  <div className="flex items-center gap-3 mb-1">
                    <span className="text-xs font-mono text-gray-600">
                      ROUND {round.round_num}
                    </span>
                    <span
                      className={`text-xs font-semibold px-2 py-0.5 rounded ${
                        round.ct_won
                          ? "bg-emerald-400/10 text-emerald-400"
                          : "bg-red-400/10 text-red-400"
                      }`}
                    >
                      CT {round.ct_won ? "WIN" : "LOSS"}
                    </span>
                    <span className="text-xs text-gray-600 bg-white/5 px-2 py-0.5 rounded">
                      {round.scenario}
                    </span>
                  </div>
                  <p className="text-lg font-semibold text-white">
                    {round.key_moments.length} Key Decision Moment
                    {round.key_moments.length !== 1 ? "s" : ""} Detected
                  </p>
                </div>
              </div>

              {/* Assessment */}
              <div className="rounded-lg border border-white/8 bg-white/3 px-5 py-4 mb-6">
                <p className="text-xs text-gray-500 mb-1.5 uppercase tracking-wider">
                  AI Assessment
                </p>
                <p className="text-sm text-gray-300 leading-relaxed">
                  {round.overall_assessment}
                </p>
              </div>

              {/* Stats row */}
              <div className="grid grid-cols-3 gap-3 mb-6">
                <StatCard
                  label="Key Moments"
                  value={round.key_moments.length}
                  sub="decision points found"
                />
                <StatCard
                  label="Avg Confidence"
                  value={`${Math.round(
                    (round.key_moments.reduce((s, m) => s + m.confidence, 0) /
                      Math.max(round.key_moments.length, 1)) *
                      100
                  )}%`}
                  sub="model certainty"
                  accent
                />
                <StatCard
                  label="Round Outcome"
                  value={round.ct_won ? "WIN" : "LOSS"}
                  sub="CT perspective"
                />
              </div>

              {/* Decision moments */}
              <div>
                <p className="text-xs text-gray-500 uppercase tracking-wider mb-3">
                  Decision Timeline
                </p>
                <div className="space-y-3">
                  {round.key_moments.map((moment, i) => (
                    <DecisionCard key={i} moment={moment} index={i} />
                  ))}
                </div>
              </div>

              {/* Navigation between rounds */}
              <div className="flex items-center justify-between mt-8 pt-6 border-t border-white/6">
                <button
                  onClick={() => setSelectedRound(Math.max(0, selectedRound - 1))}
                  disabled={selectedRound === 0}
                  className="flex items-center gap-2 text-sm text-gray-500 hover:text-gray-300 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                >
                  <svg width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
                  </svg>
                  Previous Round
                </button>
                <span className="text-xs text-gray-600 font-mono">
                  {selectedRound + 1} / {rounds.length}
                </span>
                <button
                  onClick={() =>
                    setSelectedRound(Math.min(rounds.length - 1, selectedRound + 1))
                  }
                  disabled={selectedRound === rounds.length - 1}
                  className="flex items-center gap-2 text-sm text-gray-500 hover:text-gray-300 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                >
                  Next Round
                  <svg width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                  </svg>
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
