/* ------------------------------------------------------------------
   Web Worker entry — runs the H∞ RST synthesis off the UI thread.
   Loaded from the client via:
     new Worker(new URL("../lib/hinf-worker.ts", import.meta.url))
   ------------------------------------------------------------------ */

import { plantById } from "./hinf-plants";
import {
  SynthesisSpecs,
  synthesizeController,
} from "./hinf-synthesis";

export type WorkerRequest =
  | {
      id: number;
      type?: "solve";
      plantId: string;
      specs: SynthesisSpecs;
    }
  | { id: number; type: "cancel" };

export type WorkerResponse =
  | {
      id: number;
      type: "progress";
      iter: number;
      gamma: number;
      bw: number | null;
      feasible: boolean;
    }
  | {
      id: number;
      type: "done";
      result: Awaited<ReturnType<typeof synthesizeController>>;
    }
  | { id: number; type: "error"; message: string };

const ctx = self as unknown as Worker;

// IDs of in-flight solves that have been asked to bail. The synthesis
// loop polls this set after every bisection step and returns its best
// feasible iterate so far (or the standard infeasible result if it
// has none yet).
const cancelled = new Set<number>();

ctx.addEventListener("message", async (ev: MessageEvent<WorkerRequest>) => {
  const req = ev.data;
  if (req.type === "cancel") {
    cancelled.add(req.id);
    return;
  }
  const { id, plantId, specs } = req;
  try {
    const plant = plantById(plantId);
    const grid = plant.buildGrid(specs.Ts, specs.desBw);
    const result = await synthesizeController(
      grid,
      specs,
      (p) => {
        const msg: WorkerResponse = {
          id,
          type: "progress",
          iter: p.iter,
          gamma: p.gamma,
          bw: p.bw,
          feasible: p.feasible,
        };
        ctx.postMessage(msg);
      },
      () => cancelled.has(id),
    );
    const done: WorkerResponse = { id, type: "done", result };
    ctx.postMessage(done);
  } catch (err) {
    const msg: WorkerResponse = {
      id,
      type: "error",
      message: err instanceof Error ? err.message : String(err),
    };
    ctx.postMessage(msg);
  } finally {
    cancelled.delete(req.id);
  }
});
