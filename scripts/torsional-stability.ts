/* Verifies G_vel(s) is stable for the torsional-chain parameters.
   Builds den(s) = α₁(α₂α₃ − β₂²) − β₁²α₃, factors out the integrator
   (root at s=0 from the rigid-body mode), and runs Routh-Hurwitz on
   the residual quintic. */

const TORS = {
  J1: 5e-3, J2: 3e-3, J3: 4e-3,
  k1: 800,  k2: 300,
  d1: 0.08, d2: 0.04,
  b:  0.05,
};

type Poly = number[]; // coefficients in ascending powers: p[0] + p[1]·s + …

function polyAdd(a: Poly, b: Poly): Poly {
  const n = Math.max(a.length, b.length);
  const r = new Array(n).fill(0);
  for (let i = 0; i < n; i++) r[i] = (a[i] ?? 0) + (b[i] ?? 0);
  return r;
}
function polySub(a: Poly, b: Poly): Poly {
  return polyAdd(a, b.map((x) => -x));
}
function polyMul(a: Poly, b: Poly): Poly {
  const r = new Array(a.length + b.length - 1).fill(0);
  for (let i = 0; i < a.length; i++)
    for (let j = 0; j < b.length; j++) r[i + j] += a[i] * b[j];
  return r;
}
function polyScale(a: Poly, s: number): Poly {
  return a.map((x) => x * s);
}
function polyTrim(a: Poly, tol = 1e-10): Poly {
  const r = [...a];
  while (r.length > 1 && Math.abs(r[r.length - 1]) < tol) r.pop();
  return r;
}

// Descending coefficients helper for Routh-Hurwitz (highest power first).
function descending(p: Poly): Poly {
  return [...p].reverse();
}

function routhHurwitz(pDesc: Poly): { stable: boolean; table: number[][] } {
  const n = pDesc.length - 1; // polynomial degree
  const rows: number[][] = [];
  // Row 0: even-index coefs of descending poly (s^n, s^(n-2), …)
  // Row 1: odd-index coefs (s^(n-1), s^(n-3), …)
  const r0: number[] = [];
  const r1: number[] = [];
  for (let i = 0; i <= n; i += 2) r0.push(pDesc[i]);
  for (let i = 1; i <= n; i += 2) r1.push(pDesc[i]);
  // Pad to equal length
  while (r1.length < r0.length) r1.push(0);
  rows.push(r0, r1);
  for (let k = 2; k <= n; k++) {
    const prev1 = rows[k - 2];
    const prev0 = rows[k - 1];
    const row: number[] = [];
    const leadPrev0 = prev0[0];
    if (Math.abs(leadPrev0) < 1e-18) {
      return { stable: false, table: rows };
    }
    for (let j = 0; j < prev1.length - 1; j++) {
      const val = (leadPrev0 * prev1[j + 1] - prev1[0] * prev0[j + 1]) / leadPrev0;
      row.push(val);
    }
    while (row.length < prev0.length) row.push(0);
    rows.push(row);
  }
  const firstCol = rows.map((r) => r[0]);
  const stable = firstCol.every((v) => v > 0);
  return { stable, table: rows };
}

function main() {
  const { J1, J2, J3, k1, k2, d1, d2, b } = TORS;

  // α_i(s) as ascending polynomials [c0, c1, c2]
  const a1: Poly = [k1, d1 + b, J1];
  const a2: Poly = [k1 + k2, d1 + d2 + b, J2];
  const a3: Poly = [k2, d2 + b, J3];
  const be1: Poly = [k1, d1];
  const be2: Poly = [k2, d2];

  const a2a3 = polyMul(a2, a3);
  const be2sq = polyMul(be2, be2);
  const inner = polySub(a2a3, be2sq);
  const be1sq = polyMul(be1, be1);
  const term1 = polyMul(a1, inner);
  const term2 = polyMul(be1sq, a3);
  const den = polyTrim(polySub(term1, term2));

  console.log("den(s) coefficients (ascending):");
  den.forEach((c, i) => console.log(`  s^${i}:  ${c.toExponential(4)}`));

  // Factor out s (rigid-body integrator)
  if (Math.abs(den[0]) > 1e-9) {
    console.log("\n❌ den(0) ≠ 0 — unexpected");
    return;
  }
  const quintic = den.slice(1); // divide by s
  console.log("\nG_vel denominator (quintic, ascending):");
  quintic.forEach((c, i) => console.log(`  s^${i}:  ${c.toExponential(4)}`));

  const negCount = quintic.filter((v) => v <= 0).length;
  if (negCount > 0) {
    console.log(`\n❌ ${negCount} non-positive coefficient(s) — necessary stability condition violated`);
    return;
  }
  console.log("\n✅ All coefficients positive (necessary condition passes)");

  const { stable, table } = routhHurwitz(descending(quintic));
  console.log("\nRouth-Hurwitz table (first column):");
  table.forEach((row, i) => {
    const pow = table.length - 1 - i;
    console.log(`  s^${pow}:  ${row[0].toExponential(4)}`);
  });
  console.log(`\n${stable ? "✅ STABLE" : "❌ UNSTABLE"} — all first-column entries ${stable ? "positive" : "NOT all positive"}`);
}

main();
