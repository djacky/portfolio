/* ------------------------------------------------------------------
   Shared types + Anthropic tool schema for the recruiter-match feature.
   Single source of truth used by both the API route and the React UI.
------------------------------------------------------------------ */

export type FitBand = "strong" | "partial" | "weak";
export type ReqStatus = "met" | "partial" | "gap";

export interface RubricRow {
  category: string;        // e.g. "Must-have skills", "Domain experience"
  weight: number;          // relative weight (1, 2, 3 ...)
  score: number;           // 0..100
  note: string;            // 1-line justification
}

export interface RequirementRow {
  requirement: string;     // verbatim or paraphrased from the JD
  status: ReqStatus;
  evidence: string;        // 1-line interpretation of the grounded match
  evidence_quote?: string; // exact verbatim snippet from the dossier (the source of truth)
  project_ref?: string;    // e.g. "CERN", "Eaton EV Fleet", "Disruptive Labs"
}

export interface MatchAnalysis {
  band: FitBand;
  score: number;           // headline 0..100, derived from the rubric
  summary: string;         // 2–3 sentence plain-English summary
  rubric: RubricRow[];
  requirements: RequirementRow[];
  gaps: string[];          // honest list of skill/experience gaps
  talking_points: string[];// 3 bullets to lead with in a first call
  unknowns?: string[];     // things the JD didn't specify that would change the score
}

/* Partial version for the streaming UI — every field optional while the
   structured output is being built up token-by-token. */
export type PartialMatchAnalysis = {
  band?: FitBand;
  score?: number;
  summary?: string;
  rubric?: Partial<RubricRow>[];
  requirements?: Partial<RequirementRow>[];
  gaps?: string[];
  talking_points?: string[];
  unknowns?: string[];
};

/* The Anthropic tool definition. Forcing tool_choice on this tool
   gives us guaranteed schema-conformant JSON output. */
export const SUBMIT_ANALYSIS_TOOL = {
  name: "submit_analysis",
  description:
    "Submit the structured fit analysis between the candidate's portfolio and the provided job description.",
  input_schema: {
    type: "object" as const,
    required: [
      "band",
      "score",
      "summary",
      "rubric",
      "requirements",
      "gaps",
      "talking_points",
    ],
    properties: {
      band: {
        type: "string",
        enum: ["strong", "partial", "weak"],
        description:
          "Overall fit band. strong = >=75, partial = 50-74, weak = <50.",
      },
      score: {
        type: "integer",
        minimum: 0,
        maximum: 100,
        description:
          "Headline score, computed as the weighted average of the rubric rows. Be conservative: a 90+ should be rare.",
      },
      summary: {
        type: "string",
        description:
          "2 to 3 sentence plain-English summary of the fit. No marketing language. Mention the strongest match and the biggest gap.",
      },
      rubric: {
        type: "array",
        minItems: 3,
        maxItems: 6,
        description:
          "Weighted rubric used to compute the headline score. Typical categories: Must-have skills, Nice-to-have skills, Domain experience, Seniority. Weight must-haves 2-3x nice-to-haves.",
        items: {
          type: "object",
          required: ["category", "weight", "score", "note"],
          properties: {
            category: { type: "string" },
            weight: { type: "number", minimum: 0.5, maximum: 5 },
            score: { type: "integer", minimum: 0, maximum: 100 },
            note: {
              type: "string",
              description: "One short sentence justifying the score.",
            },
          },
        },
      },
      requirements: {
        type: "array",
        minItems: 3,
        maxItems: 10,
        description:
          "The most important requirements extracted from the job description, scored individually.",
        items: {
          type: "object",
          required: ["requirement", "status", "evidence"],
          properties: {
            requirement: {
              type: "string",
              description: "The requirement, paraphrased from the JD.",
            },
            status: {
              type: "string",
              enum: ["met", "partial", "gap"],
            },
            evidence: {
              type: "string",
              description:
                "Short interpretation of how the candidate matches this requirement (1 sentence). If status is gap, briefly state what's missing.",
            },
            evidence_quote: {
              type: "string",
              description:
                "VERBATIM snippet (5-25 words) copied EXACTLY from the candidate dossier that grounds this match. Must appear character-for-character in the dossier. Omit only if status is 'gap' and there is genuinely nothing to quote.",
            },
            project_ref: {
              type: "string",
              description:
                "Optional: the project name from the dossier that supports this evidence (e.g. CERN, Eaton EV Fleet, Disruptive Labs).",
            },
          },
        },
      },
      gaps: {
        type: "array",
        minItems: 0,
        maxItems: 5,
        description:
          "Honest list of skill or experience gaps. Empty array if none worth flagging. This section builds trust — do not pad it, but do not hide real gaps either.",
        items: { type: "string" },
      },
      talking_points: {
        type: "array",
        minItems: 2,
        maxItems: 4,
        description:
          "2 to 4 short bullets the candidate could lead with in a first conversation about this role.",
        items: { type: "string" },
      },
      unknowns: {
        type: "array",
        minItems: 0,
        maxItems: 4,
        description:
          "Things the JD did NOT specify that would meaningfully change the score if known (e.g. 'team size', 'whether the role is research or applied', 'expected on-call rotation'). Empty array if the JD was complete.",
        items: { type: "string" },
      },
    },
  },
} as const;
