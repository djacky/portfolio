declare module "scs-solver" {
  interface ScsInfo {
    iter: number;
    pobj: number;
    dobj: number;
    resPri: number;
    resDual: number;
    resInfeas: number;
    resUnbdd: number;
    solveTime: number;
    setupTime: number;
  }

  interface ScsSolution {
    x: number[] | Float64Array;
    y: number[] | Float64Array;
    s: number[] | Float64Array;
    info: ScsInfo;
    status: string;
  }

  interface ScsData {
    m: number;
    n: number;
    A_x: number[];
    A_i: number[];
    A_p: number[];
    P_x?: number[];
    P_i?: number[];
    P_p?: number[];
    b: number[];
    c: number[];
  }

  interface ScsCone {
    z?: number;
    l?: number;
    bu?: number[];
    bl?: number[];
    bsize?: number;
    q?: number[];
    qsize?: number;
    ep?: number;
    ed?: number;
    p?: number[];
    psize?: number;
  }

  interface ScsSettingsShape {
    normalize?: boolean;
    scale?: number;
    adaptiveScale?: boolean;
    rhoX?: number;
    maxIters?: number;
    epsAbs?: number;
    epsRel?: number;
    epsInfeas?: number;
    alpha?: number;
    timeLimitSecs?: number;
    verbose?: number;
    warmStart?: boolean;
  }

  interface ScsModule {
    solve(
      data: ScsData,
      cone: ScsCone,
      settings: ScsSettingsShape,
      warmStart?: ScsSolution,
    ): ScsSolution;
    ScsSettings: new () => ScsSettingsShape;
    setDefaultSettings(s: ScsSettingsShape): void;
  }

  export default function createSCS(): Promise<ScsModule>;
}
