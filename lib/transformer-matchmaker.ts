/* ------------------------------------------------------------------
   Transformer Matchmaker — a tiny, hand-rolled, in-browser transformer
   that treats each player as a SEQUENCE OF MATCHES and outputs four
   heads simultaneously:

     • skill      (μ, logσ²)      — Gaussian posterior over true skill
     • playstyle  (4-D embedding) — trained with SimCLR-style contrast
     • anomaly    (P(smurf))      — binary head, BCE loss
     • form       ([-0.3, +0.3])  — short-term tilt / hot-streak modifier

   Architecture
     d_model = 32, n_heads = 2, n_layers = 2, d_ff = 64
     seq_len = 20, input_dim = 12
     Pre-norm transformer with RMSNorm (LLaMA-style)
     Rotary positional embeddings (time-ordered matches)
     Causal self-attention (a match can only attend to its past)
     ReLU FFN
     ~18K params, Adam optimizer with m/v state (~54K scalars total)

   Training signal
     Self-supervised next-match-placement prediction via Gaussian NLL
     on the skill head, plus contrastive + BCE auxiliary losses.

   All math is written out in plain TypeScript — no tfjs, no autograd.
   Designed to fit a ~30ms-per-player forward-pass budget so we can
   re-encode a 160-player queue at roughly 3–5 Hz.
------------------------------------------------------------------ */

export const D_MODEL = 32;
export const N_HEADS = 2;
export const D_HEAD = D_MODEL / N_HEADS; // 16
export const N_LAYERS = 2;
export const D_FFN = 64;
export const SEQ_LEN = 20;
// Input token fields: placement, K, D, dmg, acc, lobby μ mean, lobby μ spread,
// time-since-last, 4× patch one-hot, climbVelocity (MMR-gain-per-match proxy —
// Riot/Blizzard cite this as a primary smurf signal in their public writeups).
export const INPUT_DIM = 13;

export const SKILL_DIM = 2;       // μ, log σ²
export const PLAYSTYLE_DIM = 4;
export const ANOMALY_DIM = 1;
export const FORM_DIM = 1;

const EPS = 1e-5;

/* ---------------- tiny math helpers ---------------- */

function randn(std: number): number {
  const u = 1 - Math.random();
  const v = Math.random();
  return std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function zeros(n: number): Float32Array {
  return new Float32Array(n);
}
function randInit(n: number, fanIn: number): Float32Array {
  const std = Math.sqrt(2 / fanIn);
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = randn(std);
  return a;
}
function ones(n: number): Float32Array {
  const a = new Float32Array(n);
  a.fill(1);
  return a;
}

/* ---------------- RMSNorm (LLaMA-style) ----------------
   y[i] = g[i] · x[i] / rms(x),   rms(x) = sqrt(mean(x²) + ε)
   Simpler than LayerNorm (no mean subtraction), and the backward
   closed form is a one-liner.
------------------------------------------------------------ */

function rmsnormForward(
  x: Float32Array,
  g: Float32Array,
  out: Float32Array,
  T: number,
  D: number,
  rmsOut: Float32Array, // length T — cache for backward
) {
  for (let t = 0; t < T; t++) {
    let s = 0;
    const b = t * D;
    for (let i = 0; i < D; i++) s += x[b + i] * x[b + i];
    const rms = Math.sqrt(s / D + EPS);
    rmsOut[t] = rms;
    for (let i = 0; i < D; i++) out[b + i] = g[i] * x[b + i] / rms;
  }
}

function rmsnormBackward(
  dy: Float32Array,
  x: Float32Array,
  g: Float32Array,
  rmsCache: Float32Array,
  dx: Float32Array,
  dg: Float32Array,
  T: number,
  D: number,
) {
  // Given y = g·x/rms, with rms = sqrt(mean(x²)+ε):
  //   dg[i]  += sum_t dy[t,i] · x[t,i]/rms[t]
  //   dx[t,i] = g[i]·dy[t,i]/rms[t] − x[t,i]·sum_j(g[j]·dy[t,j]·x[t,j]) / (D·rms[t]³)
  for (let t = 0; t < T; t++) {
    const b = t * D;
    const rms = rmsCache[t];
    let dot = 0;
    for (let i = 0; i < D; i++) dot += g[i] * dy[b + i] * x[b + i];
    const scale = dot / (D * rms * rms * rms);
    for (let i = 0; i < D; i++) {
      dg[i] += dy[b + i] * x[b + i] / rms;
      dx[b + i] += g[i] * dy[b + i] / rms - x[b + i] * scale;
    }
  }
}

/* ---------------- rotary positional embeddings ----------------
   RoPE rotates pairs (2k, 2k+1) of each head's feature axis by an
   angle that depends on position t and pair index k.  The same
   rotation is applied to Q and K before the dot product; V is left
   alone.  RoPE is used in GPT-NeoX, LLaMA, Qwen — the modern default.
------------------------------------------------------------- */

interface Rope { cos: Float32Array; sin: Float32Array; }
function buildRope(): Rope {
  const cos = new Float32Array(SEQ_LEN * D_HEAD);
  const sin = new Float32Array(SEQ_LEN * D_HEAD);
  for (let t = 0; t < SEQ_LEN; t++) {
    for (let k = 0; k < D_HEAD / 2; k++) {
      const theta = t / Math.pow(10000, (2 * k) / D_HEAD);
      const c = Math.cos(theta);
      const s = Math.sin(theta);
      const p = t * D_HEAD + 2 * k;
      cos[p] = c; cos[p + 1] = c;
      sin[p] = s; sin[p + 1] = s;
    }
  }
  return { cos, sin };
}

// Rotate (a, b) → (a·cos − b·sin, a·sin + b·cos)
// Apply in-place to x of shape [T, D_HEAD].
function ropeApply(x: Float32Array, rope: Rope, T: number) {
  for (let t = 0; t < T; t++) {
    const b = t * D_HEAD;
    for (let k = 0; k < D_HEAD; k += 2) {
      const a = x[b + k];
      const bv = x[b + k + 1];
      const c = rope.cos[b + k];
      const s = rope.sin[b + k];
      x[b + k] = a * c - bv * s;
      x[b + k + 1] = a * s + bv * c;
    }
  }
}
// Inverse rotation: gradient w.r.t. pre-rope from gradient w.r.t. post-rope.
// If y = R·x, then dx = R^T·dy — since R is an orthogonal 2x2 rotation,
// R^T means negating the sine terms.
function ropeBackward(dx: Float32Array, rope: Rope, T: number) {
  for (let t = 0; t < T; t++) {
    const b = t * D_HEAD;
    for (let k = 0; k < D_HEAD; k += 2) {
      const da = dx[b + k];
      const db = dx[b + k + 1];
      const c = rope.cos[b + k];
      const s = rope.sin[b + k];
      dx[b + k]     =  da * c + db * s;
      dx[b + k + 1] = -da * s + db * c;
    }
  }
}

/* ---------------- parameter & gradient containers ---------------- */

interface BlockParams {
  // pre-attn norm
  ln1g: Float32Array; // [D]
  // QKV combined: [3D × D]
  Wqkv: Float32Array;
  bqkv: Float32Array; // [3D]
  // output projection
  Wo: Float32Array;   // [D × D]
  bo: Float32Array;   // [D]
  // pre-ffn norm
  ln2g: Float32Array; // [D]
  // FFN
  Wff1: Float32Array; // [d_ff × D]
  bff1: Float32Array; // [d_ff]
  Wff2: Float32Array; // [D × d_ff]
  bff2: Float32Array; // [D]
}

export interface TransformerParams {
  Wemb: Float32Array; // [D × input_dim]
  bemb: Float32Array; // [D]
  blocks: BlockParams[];
  lnFg: Float32Array; // [D]
  // heads
  Wsk: Float32Array; bsk: Float32Array;   // [2×D], [2]
  Wps: Float32Array; bps: Float32Array;   // [4×D], [4]
  Wan: Float32Array; ban: Float32Array;   // [1×D], [1]
  Wfm: Float32Array; bfm: Float32Array;   // [1×D], [1]
}

export type Grads = TransformerParams;

function makeBlockParams(): BlockParams {
  return {
    ln1g: ones(D_MODEL),
    Wqkv: randInit(3 * D_MODEL * D_MODEL, D_MODEL),
    bqkv: zeros(3 * D_MODEL),
    Wo:   randInit(D_MODEL * D_MODEL, D_MODEL),
    bo:   zeros(D_MODEL),
    ln2g: ones(D_MODEL),
    Wff1: randInit(D_FFN * D_MODEL, D_MODEL),
    bff1: zeros(D_FFN),
    Wff2: randInit(D_MODEL * D_FFN, D_FFN),
    bff2: zeros(D_MODEL),
  };
}
function makeBlockGrads(): BlockParams {
  return {
    ln1g: zeros(D_MODEL),
    Wqkv: zeros(3 * D_MODEL * D_MODEL),
    bqkv: zeros(3 * D_MODEL),
    Wo:   zeros(D_MODEL * D_MODEL),
    bo:   zeros(D_MODEL),
    ln2g: zeros(D_MODEL),
    Wff1: zeros(D_FFN * D_MODEL),
    bff1: zeros(D_FFN),
    Wff2: zeros(D_MODEL * D_FFN),
    bff2: zeros(D_MODEL),
  };
}

export function makeParams(): TransformerParams {
  // Anomaly bias initialized to logit(prior ≈ 0.12) ≈ −2.  Without this
  // prior-matching init, the untrained head outputs random logits around 0
  // and half the queue is flagged as smurfs on frame 1 — visually alarming
  // and pollutes the BCE training signal for the first ~200 steps.
  const banInit = new Float32Array(ANOMALY_DIM);
  banInit[0] = -2;
  return {
    Wemb: randInit(D_MODEL * INPUT_DIM, INPUT_DIM),
    bemb: zeros(D_MODEL),
    blocks: Array.from({ length: N_LAYERS }, makeBlockParams),
    lnFg: ones(D_MODEL),
    Wsk: randInit(SKILL_DIM * D_MODEL, D_MODEL),
    bsk: zeros(SKILL_DIM),
    Wps: randInit(PLAYSTYLE_DIM * D_MODEL, D_MODEL),
    bps: zeros(PLAYSTYLE_DIM),
    Wan: randInit(ANOMALY_DIM * D_MODEL, D_MODEL),
    ban: banInit,
    Wfm: randInit(FORM_DIM * D_MODEL, D_MODEL),
    bfm: zeros(FORM_DIM),
  };
}
export function makeGrads(): Grads {
  return {
    Wemb: zeros(D_MODEL * INPUT_DIM),
    bemb: zeros(D_MODEL),
    blocks: Array.from({ length: N_LAYERS }, makeBlockGrads),
    lnFg: zeros(D_MODEL),
    Wsk: zeros(SKILL_DIM * D_MODEL),
    bsk: zeros(SKILL_DIM),
    Wps: zeros(PLAYSTYLE_DIM * D_MODEL),
    bps: zeros(PLAYSTYLE_DIM),
    Wan: zeros(ANOMALY_DIM * D_MODEL),
    ban: zeros(ANOMALY_DIM),
    Wfm: zeros(FORM_DIM * D_MODEL),
    bfm: zeros(FORM_DIM),
  };
}

/** Deep-copy a TransformerParams bag into fresh Float32Arrays. */
export function cloneParams(p: TransformerParams): TransformerParams {
  const cp = (a: Float32Array) => new Float32Array(a);
  return {
    Wemb: cp(p.Wemb),
    bemb: cp(p.bemb),
    blocks: p.blocks.map((b) => ({
      ln1g: cp(b.ln1g),
      Wqkv: cp(b.Wqkv), bqkv: cp(b.bqkv),
      Wo:   cp(b.Wo),   bo:   cp(b.bo),
      ln2g: cp(b.ln2g),
      Wff1: cp(b.Wff1), bff1: cp(b.bff1),
      Wff2: cp(b.Wff2), bff2: cp(b.bff2),
    })),
    lnFg: cp(p.lnFg),
    Wsk: cp(p.Wsk), bsk: cp(p.bsk),
    Wps: cp(p.Wps), bps: cp(p.bps),
    Wan: cp(p.Wan), ban: cp(p.ban),
    Wfm: cp(p.Wfm), bfm: cp(p.bfm),
  };
}

/** Polyak-average online params into shadow: ema ← decay · ema + (1-decay) · online. */
export function emaUpdate(ema: TransformerParams, online: TransformerParams, decay: number) {
  const blend = (e: Float32Array, o: Float32Array) => {
    for (let i = 0; i < e.length; i++) e[i] = decay * e[i] + (1 - decay) * o[i];
  };
  blend(ema.Wemb, online.Wemb); blend(ema.bemb, online.bemb);
  blend(ema.lnFg, online.lnFg);
  blend(ema.Wsk, online.Wsk); blend(ema.bsk, online.bsk);
  blend(ema.Wps, online.Wps); blend(ema.bps, online.bps);
  blend(ema.Wan, online.Wan); blend(ema.ban, online.ban);
  blend(ema.Wfm, online.Wfm); blend(ema.bfm, online.bfm);
  for (let l = 0; l < ema.blocks.length; l++) {
    const be = ema.blocks[l], bo = online.blocks[l];
    blend(be.ln1g, bo.ln1g);
    blend(be.Wqkv, bo.Wqkv); blend(be.bqkv, bo.bqkv);
    blend(be.Wo,   bo.Wo);   blend(be.bo,   bo.bo);
    blend(be.ln2g, bo.ln2g);
    blend(be.Wff1, bo.Wff1); blend(be.bff1, bo.bff1);
    blend(be.Wff2, bo.Wff2); blend(be.bff2, bo.bff2);
  }
}

/* ---------------- forward-pass activation cache ----------------
   Everything needed to backprop one forward pass of one player.
---------------------------------------------------------------- */

interface BlockCache {
  // inputs and norm stats
  xIn: Float32Array;   // [T × D] — input to block
  ln1Out: Float32Array;// [T × D] — post-RMSNorm
  ln1Rms: Float32Array;// [T]
  // attention
  q: Float32Array;     // [T × D]  (post-rope, reshaped by head)
  k: Float32Array;     // [T × D]  (post-rope)
  v: Float32Array;     // [T × D]
  attn: Float32Array;  // [N_HEADS × T × T] — softmax weights
  attnOut: Float32Array; // [T × D] pre-Wo
  // after attention residual
  xPostAttn: Float32Array; // [T × D]
  ln2Out: Float32Array;    // [T × D]
  ln2Rms: Float32Array;    // [T]
  // FFN
  ffnHidden: Float32Array; // [T × d_ff] post-ReLU
  ffnPre: Float32Array;    // [T × d_ff] pre-ReLU (for ReLU mask on backward)
  // block output (post-FFN-residual) — consumed by next block or final LN
  xOut: Float32Array;  // [T × D]
}

interface ForwardCache {
  T: number; // actual sequence length used
  tokens: Float32Array; // [T × input_dim]
  xEmb: Float32Array;   // [T × D]
  blocks: BlockCache[];
  lnFOut: Float32Array;  // [T × D]
  lnFRms: Float32Array;  // [T]
  zLast: Float32Array;   // [D] — last-token representation for heads
  // head outputs
  skill: Float32Array;     // [2]
  playstyle: Float32Array; // [4]
  anomaly: Float32Array;   // [1] pre-sigmoid logit
  form: Float32Array;      // [1] pre-tanh
}

/* ---------------- forward pass ---------------- */

export class TransformerMatchmaker {
  params: TransformerParams;
  rope: Rope;

  // Adam state — same shape as params
  step = 0;
  m: TransformerParams;
  v: TransformerParams;

  // Polyak-averaged shadow of `params`.  forwardEma() runs inference through
  // these — used for ρ/evaluation so the metric isn't rattled by every
  // single Adam step.  Tani & Liwicki 2019 and Izmailov 2018 (SWA) both
  // report ~2-4× lower eval variance at essentially zero runtime cost.
  ema: TransformerParams;
  emaDecay = 0.995;

  constructor(params?: TransformerParams) {
    this.params = params ?? makeParams();
    this.rope = buildRope();
    this.m = makeGrads();
    this.v = makeGrads();
    this.ema = cloneParams(this.params);
  }

  reset() {
    this.params = makeParams();
    this.m = makeGrads();
    this.v = makeGrads();
    this.step = 0;
    this.ema = cloneParams(this.params);
  }

  /** Update the EMA shadow using the current online params. */
  stepEma(decay = this.emaDecay) {
    emaUpdate(this.ema, this.params, decay);
  }

  /**
   * Forward using the EMA shadow weights.  Swap-trick: restore online
   * params in finally so an exception mid-forward never leaves the model
   * in a corrupt state.
   */
  forwardEma(tokens: Float32Array, T: number): ForwardCache {
    const online = this.params;
    this.params = this.ema;
    try {
      return this.forward(tokens, T);
    } finally {
      this.params = online;
    }
  }

  /**
   * Forward a single player's history.
   *   tokens: shape [T × INPUT_DIM], T ≤ SEQ_LEN
   * Returns a cache with all intermediate activations + head outputs.
   */
  forward(tokens: Float32Array, T: number): ForwardCache {
    const P = this.params;
    const D = D_MODEL;

    // --- token embedding: xEmb[t] = Wemb · tokens[t] + bemb
    const xEmb = new Float32Array(T * D);
    for (let t = 0; t < T; t++) {
      const tb = t * INPUT_DIM;
      const eb = t * D;
      for (let i = 0; i < D; i++) {
        let s = P.bemb[i];
        const row = i * INPUT_DIM;
        for (let j = 0; j < INPUT_DIM; j++) s += P.Wemb[row + j] * tokens[tb + j];
        xEmb[eb + i] = s;
      }
    }

    // --- stack of blocks
    const blockCaches: BlockCache[] = [];
    let xCur = xEmb;
    for (let l = 0; l < N_LAYERS; l++) {
      const bc = this.blockForward(xCur, T, P.blocks[l]);
      blockCaches.push(bc);
      xCur = bc.xOut;
    }

    // --- final RMSNorm
    const lnFOut = new Float32Array(T * D);
    const lnFRms = new Float32Array(T);
    rmsnormForward(xCur, P.lnFg, lnFOut, T, D, lnFRms);

    // --- pull last-token representation
    const zLast = new Float32Array(D);
    for (let i = 0; i < D; i++) zLast[i] = lnFOut[(T - 1) * D + i];

    // --- heads
    const skill = matvec(P.Wsk, P.bsk, zLast, SKILL_DIM, D);
    const playstyle = matvec(P.Wps, P.bps, zLast, PLAYSTYLE_DIM, D);
    const anomaly = matvec(P.Wan, P.ban, zLast, ANOMALY_DIM, D);
    const form = matvec(P.Wfm, P.bfm, zLast, FORM_DIM, D);

    return {
      T, tokens,
      xEmb, blocks: blockCaches,
      lnFOut, lnFRms, zLast,
      skill, playstyle, anomaly, form,
    };
  }

  private blockForward(xIn: Float32Array, T: number, B: BlockParams): BlockCache {
    const D = D_MODEL;

    // pre-attn RMSNorm
    const ln1Out = new Float32Array(T * D);
    const ln1Rms = new Float32Array(T);
    rmsnormForward(xIn, B.ln1g, ln1Out, T, D, ln1Rms);

    // Q, K, V = ln1Out @ Wqkv^T + bqkv  (split into three D-chunks)
    const q = new Float32Array(T * D);
    const k = new Float32Array(T * D);
    const v = new Float32Array(T * D);
    for (let t = 0; t < T; t++) {
      const xb = t * D;
      for (let i = 0; i < D; i++) {
        // Q row i  (Wqkv rows 0..D-1)
        let sq = B.bqkv[i];
        let sk = B.bqkv[D + i];
        let sv = B.bqkv[2 * D + i];
        const rowQ = i * D;
        const rowK = (D + i) * D;
        const rowV = (2 * D + i) * D;
        for (let j = 0; j < D; j++) {
          sq += B.Wqkv[rowQ + j] * ln1Out[xb + j];
          sk += B.Wqkv[rowK + j] * ln1Out[xb + j];
          sv += B.Wqkv[rowV + j] * ln1Out[xb + j];
        }
        q[xb + i] = sq;
        k[xb + i] = sk;
        v[xb + i] = sv;
      }
    }

    // Apply RoPE to Q, K (per-head slice)
    for (let h = 0; h < N_HEADS; h++) {
      const qh = subheadSlice(q, h, T);
      const kh = subheadSlice(k, h, T);
      ropeApply(qh, this.rope, T);
      ropeApply(kh, this.rope, T);
      writeSubheadSlice(q, qh, h, T);
      writeSubheadSlice(k, kh, h, T);
    }

    // Causal-masked scaled dot-product attention per head
    const attn = new Float32Array(N_HEADS * T * T);
    const attnOut = new Float32Array(T * D);
    const invSqrt = 1 / Math.sqrt(D_HEAD);
    for (let h = 0; h < N_HEADS; h++) {
      const headOff = h * D_HEAD;
      for (let t = 0; t < T; t++) {
        // scores[t, s] = q_h[t]·k_h[s]·invSqrt for s ≤ t, else −∞
        const scores = new Float32Array(t + 1);
        let m = -Infinity;
        for (let s = 0; s <= t; s++) {
          let dot = 0;
          const qb = t * D + headOff;
          const kb = s * D + headOff;
          for (let d = 0; d < D_HEAD; d++) dot += q[qb + d] * k[kb + d];
          const sc = dot * invSqrt;
          scores[s] = sc;
          if (sc > m) m = sc;
        }
        let sumE = 0;
        for (let s = 0; s <= t; s++) {
          scores[s] = Math.exp(scores[s] - m);
          sumE += scores[s];
        }
        for (let s = 0; s <= t; s++) {
          const a = scores[s] / sumE;
          attn[(h * T + t) * T + s] = a;
        }
        // attnOut[t, headOff:headOff+D_HEAD] = sum_s attn[h,t,s] * v_h[s]
        for (let d = 0; d < D_HEAD; d++) {
          let acc = 0;
          for (let s = 0; s <= t; s++) {
            acc += attn[(h * T + t) * T + s] * v[s * D + headOff + d];
          }
          attnOut[t * D + headOff + d] = acc;
        }
      }
    }

    // y = attnOut @ Wo^T + bo
    const y = new Float32Array(T * D);
    for (let t = 0; t < T; t++) {
      const ob = t * D;
      for (let i = 0; i < D; i++) {
        let s = B.bo[i];
        const row = i * D;
        for (let j = 0; j < D; j++) s += B.Wo[row + j] * attnOut[ob + j];
        y[ob + i] = s;
      }
    }
    // residual
    const xPostAttn = new Float32Array(T * D);
    for (let i = 0; i < T * D; i++) xPostAttn[i] = xIn[i] + y[i];

    // pre-FFN RMSNorm
    const ln2Out = new Float32Array(T * D);
    const ln2Rms = new Float32Array(T);
    rmsnormForward(xPostAttn, B.ln2g, ln2Out, T, D, ln2Rms);

    // FFN: h = ReLU(ln2Out @ Wff1^T + bff1); out = h @ Wff2^T + bff2
    const ffnPre = new Float32Array(T * D_FFN);
    const ffnHidden = new Float32Array(T * D_FFN);
    for (let t = 0; t < T; t++) {
      const xb = t * D;
      const hb = t * D_FFN;
      for (let i = 0; i < D_FFN; i++) {
        let s = B.bff1[i];
        const row = i * D;
        for (let j = 0; j < D; j++) s += B.Wff1[row + j] * ln2Out[xb + j];
        ffnPre[hb + i] = s;
        ffnHidden[hb + i] = s > 0 ? s : 0;
      }
    }
    const ffnOut = new Float32Array(T * D);
    for (let t = 0; t < T; t++) {
      const xb = t * D;
      const hb = t * D_FFN;
      for (let i = 0; i < D; i++) {
        let s = B.bff2[i];
        const row = i * D_FFN;
        for (let j = 0; j < D_FFN; j++) s += B.Wff2[row + j] * ffnHidden[hb + j];
        ffnOut[xb + i] = s;
      }
    }
    // residual
    const xOut = new Float32Array(T * D);
    for (let i = 0; i < T * D; i++) xOut[i] = xPostAttn[i] + ffnOut[i];

    const cache: BlockCache = {
      xIn, ln1Out, ln1Rms,
      q, k, v,
      attn, attnOut,
      xPostAttn, ln2Out, ln2Rms,
      ffnHidden, ffnPre,
      xOut,
    };
    return cache;
  }

  /**
   * Backprop given gradients on each head output. Accumulates into `grads`.
   *   dSkill, dPlaystyle, dAnomaly, dForm: gradient tensors
   *   of shapes [SKILL_DIM], [PLAYSTYLE_DIM], [ANOMALY_DIM], [FORM_DIM]
   */
  backward(
    fc: ForwardCache,
    dSkill: Float32Array,
    dPlaystyle: Float32Array,
    dAnomaly: Float32Array,
    dForm: Float32Array,
    grads: Grads,
  ) {
    const P = this.params;
    const D = D_MODEL;
    const T = fc.T;

    // --- Head backwards: all four heads are linear, contribute dZ = sum W^T dy
    const dzLast = new Float32Array(D);
    matvecBackward(dSkill, P.Wsk, fc.zLast, grads.Wsk, grads.bsk, dzLast, SKILL_DIM, D);
    matvecBackward(dPlaystyle, P.Wps, fc.zLast, grads.Wps, grads.bps, dzLast, PLAYSTYLE_DIM, D);
    matvecBackward(dAnomaly, P.Wan, fc.zLast, grads.Wan, grads.ban, dzLast, ANOMALY_DIM, D);
    matvecBackward(dForm, P.Wfm, fc.zLast, grads.Wfm, grads.bfm, dzLast, FORM_DIM, D);

    // --- Spread dzLast into a full [T × D] grad at the last position
    const dLnFOut = new Float32Array(T * D);
    for (let i = 0; i < D; i++) dLnFOut[(T - 1) * D + i] = dzLast[i];

    // --- Final RMSNorm backward (input = last block's xOut)
    const lastBlock = fc.blocks[N_LAYERS - 1];
    const dXafter = new Float32Array(T * D);
    rmsnormBackward(dLnFOut, lastBlock.xOut, P.lnFg, fc.lnFRms, dXafter, grads.lnFg, T, D);

    // --- Backward through each block (reverse order)
    let dX: Float32Array = dXafter;
    for (let l = N_LAYERS - 1; l >= 0; l--) {
      dX = this.blockBackward(fc.blocks[l], dX, T, P.blocks[l], grads.blocks[l]);
    }

    // --- Embedding backward: xEmb[t,i] = bemb[i] + sum_j Wemb[i,j] · tokens[t,j]
    for (let t = 0; t < T; t++) {
      const tb = t * INPUT_DIM;
      const eb = t * D;
      for (let i = 0; i < D; i++) {
        const dOut = dX[eb + i];
        grads.bemb[i] += dOut;
        const row = i * INPUT_DIM;
        for (let j = 0; j < INPUT_DIM; j++) {
          grads.Wemb[row + j] += dOut * fc.tokens[tb + j];
        }
      }
    }
  }

  private blockBackward(
    bc: BlockCache,
    dxOut: Float32Array,
    T: number,
    B: BlockParams,
    gB: BlockParams,
  ): Float32Array {
    const D = D_MODEL;

    // --- residual: xOut = xPostAttn + ffnOut
    const dFfnOut = dxOut; // gradient through residual to ffn branch
    const dxPostAttnResid = new Float32Array(T * D);
    for (let i = 0; i < T * D; i++) dxPostAttnResid[i] = dxOut[i];

    // --- FFN backward
    // ffnOut = ffnHidden @ Wff2^T + bff2
    const dFfnHidden = new Float32Array(T * D_FFN);
    for (let t = 0; t < T; t++) {
      const xb = t * D;
      const hb = t * D_FFN;
      for (let i = 0; i < D; i++) {
        const dOut = dFfnOut[xb + i];
        gB.bff2[i] += dOut;
        const row = i * D_FFN;
        for (let j = 0; j < D_FFN; j++) {
          gB.Wff2[row + j] += dOut * bc.ffnHidden[hb + j];
          dFfnHidden[hb + j] += dOut * B.Wff2[row + j];
        }
      }
    }
    // ReLU backward + Wff1/bff1 backward
    const dLn2Out = new Float32Array(T * D);
    for (let t = 0; t < T; t++) {
      const xb = t * D;
      const hb = t * D_FFN;
      for (let i = 0; i < D_FFN; i++) {
        let dh = dFfnHidden[hb + i];
        if (bc.ffnPre[hb + i] <= 0) dh = 0;
        gB.bff1[i] += dh;
        const row = i * D;
        for (let j = 0; j < D; j++) {
          gB.Wff1[row + j] += dh * bc.ln2Out[xb + j];
          dLn2Out[xb + j] += dh * B.Wff1[row + j];
        }
      }
    }
    // pre-FFN RMSNorm backward → contributes to dxPostAttn
    const dxPostAttnFromFfn = new Float32Array(T * D);
    rmsnormBackward(dLn2Out, bc.xPostAttn, B.ln2g, bc.ln2Rms, dxPostAttnFromFfn, gB.ln2g, T, D);
    const dxPostAttn = new Float32Array(T * D);
    for (let i = 0; i < T * D; i++) dxPostAttn[i] = dxPostAttnResid[i] + dxPostAttnFromFfn[i];

    // --- Attention residual: xPostAttn = xIn + y_attn
    const dxInResid = new Float32Array(T * D);
    const dYAttn = new Float32Array(T * D);
    for (let i = 0; i < T * D; i++) {
      dxInResid[i] = dxPostAttn[i];
      dYAttn[i] = dxPostAttn[i];
    }

    // --- Wo backward: y = attnOut @ Wo^T + bo
    const dAttnOut = new Float32Array(T * D);
    for (let t = 0; t < T; t++) {
      const ob = t * D;
      for (let i = 0; i < D; i++) {
        const dOut = dYAttn[ob + i];
        gB.bo[i] += dOut;
        const row = i * D;
        for (let j = 0; j < D; j++) {
          gB.Wo[row + j] += dOut * bc.attnOut[ob + j];
          dAttnOut[ob + j] += dOut * B.Wo[row + j];
        }
      }
    }

    // --- Attention backward (per head)
    const dQ = new Float32Array(T * D);
    const dK = new Float32Array(T * D);
    const dV = new Float32Array(T * D);
    const invSqrt = 1 / Math.sqrt(D_HEAD);
    for (let h = 0; h < N_HEADS; h++) {
      const headOff = h * D_HEAD;
      // for each query-position t
      for (let t = 0; t < T; t++) {
        // 1) dV from attnOut = sum_s attn[t,s] · v[s]
        //    dv[s] += attn[t,s] · dAttnOut[t]
        //    dAttn[t,s] = dAttnOut[t] · v[s]
        const dAttnRow = new Float32Array(t + 1);
        for (let s = 0; s <= t; s++) {
          const a = bc.attn[(h * T + t) * T + s];
          let dot = 0;
          for (let d = 0; d < D_HEAD; d++) {
            const grad = dAttnOut[t * D + headOff + d];
            dV[s * D + headOff + d] += a * grad;
            dot += grad * bc.v[s * D + headOff + d];
          }
          dAttnRow[s] = dot;
        }
        // 2) softmax backward: dScore[s] = attn[t,s]·(dAttn[s] − sum_s' attn[t,s']·dAttn[s'])
        let dotAsum = 0;
        for (let s = 0; s <= t; s++) {
          dotAsum += bc.attn[(h * T + t) * T + s] * dAttnRow[s];
        }
        const dScores = new Float32Array(t + 1);
        for (let s = 0; s <= t; s++) {
          dScores[s] = bc.attn[(h * T + t) * T + s] * (dAttnRow[s] - dotAsum);
        }
        // 3) dQ[t], dK[s] from scores[t,s] = q[t]·k[s]·invSqrt
        for (let s = 0; s <= t; s++) {
          const g = dScores[s] * invSqrt;
          for (let d = 0; d < D_HEAD; d++) {
            dQ[t * D + headOff + d] += g * bc.k[s * D + headOff + d];
            dK[s * D + headOff + d] += g * bc.q[t * D + headOff + d];
          }
        }
      }
    }

    // --- RoPE backward (inverse rotation applied to dQ and dK per head)
    for (let h = 0; h < N_HEADS; h++) {
      const dQh = subheadSlice(dQ, h, T);
      const dKh = subheadSlice(dK, h, T);
      ropeBackward(dQh, this.rope, T);
      ropeBackward(dKh, this.rope, T);
      writeSubheadSlice(dQ, dQh, h, T);
      writeSubheadSlice(dK, dKh, h, T);
    }

    // --- Wqkv backward: Q = ln1Out @ W_q^T + b_q  (same for K, V; concat into Wqkv)
    const dLn1Out = new Float32Array(T * D);
    for (let t = 0; t < T; t++) {
      const xb = t * D;
      for (let i = 0; i < D; i++) {
        const dq = dQ[xb + i], dk = dK[xb + i], dv = dV[xb + i];
        gB.bqkv[i]         += dq;
        gB.bqkv[D + i]     += dk;
        gB.bqkv[2 * D + i] += dv;
        const rowQ = i * D;
        const rowK = (D + i) * D;
        const rowV = (2 * D + i) * D;
        for (let j = 0; j < D; j++) {
          gB.Wqkv[rowQ + j] += dq * bc.ln1Out[xb + j];
          gB.Wqkv[rowK + j] += dk * bc.ln1Out[xb + j];
          gB.Wqkv[rowV + j] += dv * bc.ln1Out[xb + j];
          dLn1Out[xb + j] += dq * B.Wqkv[rowQ + j]
                           + dk * B.Wqkv[rowK + j]
                           + dv * B.Wqkv[rowV + j];
        }
      }
    }

    // --- pre-attn RMSNorm backward → contributes to dxIn
    const dxInFromAttn = new Float32Array(T * D);
    rmsnormBackward(dLn1Out, bc.xIn, B.ln1g, bc.ln1Rms, dxInFromAttn, gB.ln1g, T, D);

    // combine both residual paths
    const dxIn = new Float32Array(T * D);
    for (let i = 0; i < T * D; i++) dxIn[i] = dxInResid[i] + dxInFromAttn[i];
    return dxIn;
  }

  /**
   * Adam update. Call AFTER accumulating gradients into `grads`.
   * Mutates this.params and this.m, this.v. Zeros grads as a convenience.
   *
   * Applies global-norm gradient clipping (clipNorm) BEFORE the Adam step
   * — without this, unlucky contrastive or Plackett-Luce batches can spike
   * a few gradients by 50–100×, which Adam's denominator can't normalize
   * away fast enough and the ρ metric falls off a cliff for ~20 ticks.
   * Clip at 1.0 is the standard GPT/BERT recipe.
   */
  adamStep(grads: Grads, lr: number, beta1 = 0.9, beta2 = 0.999, wd = 1e-4, clipNorm = 1.0) {
    this.step += 1;
    const bc1 = 1 - Math.pow(beta1, this.step);
    const bc2 = 1 - Math.pow(beta2, this.step);

    const zipParams = (p: TransformerParams, m: TransformerParams, v: TransformerParams, g: Grads) => {
      const flat: [Float32Array, Float32Array, Float32Array, Float32Array, boolean][] = [
        [p.Wemb, m.Wemb, v.Wemb, g.Wemb, true],
        [p.bemb, m.bemb, v.bemb, g.bemb, false],
        [p.lnFg, m.lnFg, v.lnFg, g.lnFg, false],
        [p.Wsk, m.Wsk, v.Wsk, g.Wsk, true],
        [p.bsk, m.bsk, v.bsk, g.bsk, false],
        [p.Wps, m.Wps, v.Wps, g.Wps, true],
        [p.bps, m.bps, v.bps, g.bps, false],
        [p.Wan, m.Wan, v.Wan, g.Wan, true],
        [p.ban, m.ban, v.ban, g.ban, false],
        [p.Wfm, m.Wfm, v.Wfm, g.Wfm, true],
        [p.bfm, m.bfm, v.bfm, g.bfm, false],
      ];
      for (let l = 0; l < p.blocks.length; l++) {
        const bp = p.blocks[l], bm = m.blocks[l], bv = v.blocks[l], bg = g.blocks[l];
        flat.push(
          [bp.ln1g, bm.ln1g, bv.ln1g, bg.ln1g, false],
          [bp.Wqkv, bm.Wqkv, bv.Wqkv, bg.Wqkv, true],
          [bp.bqkv, bm.bqkv, bv.bqkv, bg.bqkv, false],
          [bp.Wo,   bm.Wo,   bv.Wo,   bg.Wo,   true],
          [bp.bo,   bm.bo,   bv.bo,   bg.bo,   false],
          [bp.ln2g, bm.ln2g, bv.ln2g, bg.ln2g, false],
          [bp.Wff1, bm.Wff1, bv.Wff1, bg.Wff1, true],
          [bp.bff1, bm.bff1, bv.bff1, bg.bff1, false],
          [bp.Wff2, bm.Wff2, bv.Wff2, bg.Wff2, true],
          [bp.bff2, bm.bff2, bv.bff2, bg.bff2, false],
        );
      }
      return flat;
    };

    const flat = zipParams(this.params, this.m, this.v, grads);

    // Global L2 norm across ALL grad tensors, then compute clip scale.
    let sqSum = 0;
    for (const [, , , gv] of flat) {
      for (let i = 0; i < gv.length; i++) sqSum += gv[i] * gv[i];
    }
    const gNorm = Math.sqrt(sqSum);

    // NaN/inf safety valve: if a bad forward pass leaked a non-finite
    // gradient (e.g. 0/0 from a zero-norm embedding early in training),
    // zero the grads and skip this update instead of poisoning params.
    // Without this, a single NaN grad propagates through Adam and every
    // subsequent forward returns NaN — the visible symptom is dots
    // disappearing from the scene as playstyleEmb becomes NaN.
    if (!Number.isFinite(gNorm)) {
      for (const [, , , gv] of flat) {
        for (let i = 0; i < gv.length; i++) gv[i] = 0;
      }
      this.step -= 1; // don't advance the step counter on a no-op
      return Number.POSITIVE_INFINITY;
    }

    const scale = gNorm > clipNorm ? clipNorm / (gNorm + 1e-8) : 1;

    for (const [param, mv, vv, gv, isWeight] of flat) {
      for (let i = 0; i < param.length; i++) {
        let g = gv[i] * scale;
        if (isWeight) g += wd * param[i];
        mv[i] = beta1 * mv[i] + (1 - beta1) * g;
        vv[i] = beta2 * vv[i] + (1 - beta2) * g * g;
        const mhat = mv[i] / bc1;
        const vhat = vv[i] / bc2;
        param[i] -= lr * mhat / (Math.sqrt(vhat) + 1e-8);
        gv[i] = 0;
      }
    }
    return gNorm;
  }
}

/* ---------------- helpers ---------------- */

function matvec(
  W: Float32Array, b: Float32Array, x: Float32Array,
  outDim: number, inDim: number,
): Float32Array {
  const y = new Float32Array(outDim);
  for (let i = 0; i < outDim; i++) {
    let s = b[i];
    const row = i * inDim;
    for (let j = 0; j < inDim; j++) s += W[row + j] * x[j];
    y[i] = s;
  }
  return y;
}

// Backward of y = W·x + b given dy.  Accumulates dW, db; ADDS to dxAcc.
function matvecBackward(
  dy: Float32Array, W: Float32Array, x: Float32Array,
  dW: Float32Array, db: Float32Array, dxAcc: Float32Array,
  outDim: number, inDim: number,
) {
  for (let i = 0; i < outDim; i++) {
    const g = dy[i];
    db[i] += g;
    const row = i * inDim;
    for (let j = 0; j < inDim; j++) {
      dW[row + j] += g * x[j];
      dxAcc[j]   += g * W[row + j];
    }
  }
}

// Extract head-slice of x of shape [T × D] → new Float32Array of shape [T × D_HEAD]
function subheadSlice(x: Float32Array, h: number, T: number): Float32Array {
  const out = new Float32Array(T * D_HEAD);
  const headOff = h * D_HEAD;
  for (let t = 0; t < T; t++) {
    for (let d = 0; d < D_HEAD; d++) out[t * D_HEAD + d] = x[t * D_MODEL + headOff + d];
  }
  return out;
}
// Write a head-slice back into x of shape [T × D].
function writeSubheadSlice(x: Float32Array, slice: Float32Array, h: number, T: number) {
  const headOff = h * D_HEAD;
  for (let t = 0; t < T; t++) {
    for (let d = 0; d < D_HEAD; d++) x[t * D_MODEL + headOff + d] = slice[t * D_HEAD + d];
  }
}

/* ---------------- loss functions ----------------
   These compute loss + gradients to feed into backward().
------------------------------------------------------ */

/**
 * Gaussian NLL on the skill head.
 *   skill = [μ, log σ²], target = scalar y
 *   L = 0.5·log(2π) + 0.5·logσ² + 0.5·(y − μ)² / σ²
 *   Clamp logσ² to [−3, +3] to prevent blow-up early in training.
 */
export function skillNllLoss(
  skill: Float32Array, target: number,
): { loss: number; grad: Float32Array } {
  const mu = skill[0];
  let logvar = skill[1];
  if (logvar > 3) logvar = 3;
  if (logvar < -3) logvar = -3;
  const varv = Math.exp(logvar);
  const diff = target - mu;
  const loss = 0.5 * (logvar + (diff * diff) / varv);
  const grad = new Float32Array(2);
  grad[0] = -diff / varv;               // ∂L/∂μ
  grad[1] = 0.5 * (1 - (diff * diff) / varv); // ∂L/∂logσ²
  // clamp-carrier: if clipping fired, zero the grad component in that direction.
  if (skill[1] > 3 && grad[1] < 0) grad[1] = 0;
  if (skill[1] < -3 && grad[1] > 0) grad[1] = 0;
  return { loss, grad };
}

/**
 * Binary cross-entropy on the anomaly head.
 *   logit → σ(logit) = p;  L = −[y·log p + (1−y)·log(1−p)]
 *   dL/d_logit = p − y.
 */
export function anomalyBceLoss(
  logit: Float32Array, target: number,
): { loss: number; grad: Float32Array } {
  const x = logit[0];
  // numerically stable log-sum-exp form
  const maxX = Math.max(x, 0);
  const loss = maxX - x * target + Math.log(1 + Math.exp(-Math.abs(x)));
  const p = 1 / (1 + Math.exp(-x));
  const grad = new Float32Array(1);
  grad[0] = p - target;
  return { loss, grad };
}

/**
 * SimCLR-style contrastive loss on playstyle embeddings.
 *   Given an anchor emb a and N candidate embs c[i], where positive is c[0]
 *   and negatives are c[1..N−1], with temperature τ:
 *     L = −log(exp(sim(a,c⁺)/τ) / Σ_i exp(sim(a,c_i)/τ))
 *   Cosine similarity.  Returns gradients only w.r.t. the anchor; negatives
 *   are detached (stop-grad style) to keep the compute cheap.
 */
export function contrastivePlaystyleLoss(
  anchor: Float32Array,
  candidates: Float32Array[],    // length N; index 0 = positive
  tau = 0.2,
): { loss: number; grad: Float32Array } {
  const N = candidates.length;
  const aLen = norm(anchor);
  const lens = candidates.map(norm);
  const sims = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    sims[i] = dot(anchor, candidates[i]) / (aLen * lens[i] + 1e-9) / tau;
  }
  let m = -Infinity;
  for (let i = 0; i < N; i++) if (sims[i] > m) m = sims[i];
  let Z = 0;
  const probs = new Float32Array(N);
  for (let i = 0; i < N; i++) { probs[i] = Math.exp(sims[i] - m); Z += probs[i]; }
  for (let i = 0; i < N; i++) probs[i] /= Z;
  const loss = -Math.log(Math.max(probs[0], 1e-9));
  // dL/d_sims[i] = probs[i] − 1{i=0}
  // dL/d_anchor = (1/τ) · Σ_i (probs[i] − 1{i=0}) · d(cos)/d_anchor
  // d(cos(a,c))/d_a = c/(|a||c|) − a·cos(a,c)/|a|²
  const grad = new Float32Array(anchor.length);
  for (let i = 0; i < N; i++) {
    const w = (probs[i] - (i === 0 ? 1 : 0)) / tau;
    const c = candidates[i];
    const cl = lens[i];
    const cosAC = sims[i] * tau; // invert the 1/τ
    for (let d = 0; d < anchor.length; d++) {
      grad[d] += w * (c[d] / (aLen * cl + 1e-9) - anchor[d] * cosAC / (aLen * aLen + 1e-9));
    }
  }
  return { loss, grad };
}

function norm(x: Float32Array): number {
  let s = 0; for (let i = 0; i < x.length; i++) s += x[i] * x[i];
  return Math.sqrt(s);
}
function dot(a: Float32Array, b: Float32Array): number {
  let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/**
 * Plackett–Luce negative-log-likelihood on a completed lobby ordering.
 *   Input: thetas[k] = predicted skill score of the player who finished in
 *   rank k+1 (k=0 is the winner).  The Luce choice axiom gives the
 *   probability of the ordering as
 *       P(ordering) = Π_k  exp(θ_k) / Σ_{j≥k} exp(θ_j)
 *   and the NLL is
 *       L = Σ_k  (−θ_k + logsumexp(θ_{k..N−1}))
 *
 *   This is the CORRECT likelihood for "observed finishing order given
 *   latent skills", matching how the simulation (plackettLuce(lobby, β=6))
 *   was generated — so the training signal is aligned with the true
 *   generative model.  Regressing placement/30 with Gaussian NLL (the
 *   old signal) throws away the ordinal structure and gets confused by
 *   strong upsets.  Cao et al. 2007 "ListNet" established PL as the
 *   right objective for learning-to-rank; Guiver & Snelson 2009 extended
 *   it to partial orderings.
 *
 *   Complexity: O(N²) gradient pass (trivial for N=10 or 30).
 */
export function plackettLuceLoss(
  thetas: Float32Array,
): { loss: number; grad: Float32Array } {
  const n = thetas.length;
  // logZ[k] = log Σ_{j≥k} exp(θ_j), computed right-to-left, numerically stable.
  const logZ = new Float32Array(n);
  let runMax = -Infinity;
  let runSum = 0;
  for (let k = n - 1; k >= 0; k--) {
    const t = thetas[k];
    if (k === n - 1) {
      runMax = t; runSum = 1;
    } else if (t > runMax) {
      runSum = runSum * Math.exp(runMax - t) + 1;
      runMax = t;
    } else {
      runSum += Math.exp(t - runMax);
    }
    logZ[k] = runMax + Math.log(runSum);
  }
  let loss = 0;
  for (let k = 0; k < n; k++) loss += logZ[k] - thetas[k];

  // ∂L/∂θ_i = −1 + Σ_{k=0..i} exp(θ_i − logZ[k])
  const grad = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let g = -1;
    const ti = thetas[i];
    for (let k = 0; k <= i; k++) g += Math.exp(ti - logZ[k]);
    grad[i] = g;
  }
  return { loss, grad };
}

