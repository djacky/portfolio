import createSCS from 'scs-solver';

const SCS = await createSCS();
const data = {
  m: 3, n: 2,
  A_x: [-1.0, 1.0, 1.0, 1.0],
  A_i: [0, 1, 0, 2],
  A_p: [0, 2, 4],
  P_x: [3.0, -1.0, 2.0],
  P_i: [0, 0, 1],
  P_p: [0, 1, 3],
  b: [-1.0, 0.3, -0.5],
  c: [-1.0, -1.0]
};
const cone = { z: 1, l: 2 };
const settings = new SCS.ScsSettings();
SCS.setDefaultSettings(settings);
settings.verbose = 0;
const sol = SCS.solve(data, cone, settings);
console.log("x:", sol.x, "status:", sol.status, "pobj:", sol.info.pobj);
