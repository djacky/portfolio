/* ------------------------------------------------------------------
   Client helper — spins up the H∞ Web Worker and exposes a clean
   Promise-based API with progress streaming.
   ------------------------------------------------------------------ */

import type { SynthesisResult, SynthesisSpecs } from "./hinf-synthesis";
import type { WorkerRequest, WorkerResponse } from "./hinf-worker";

export interface ProgressEvent {
  iter: number;
  gamma: number;
  bw: number | null;
  feasible: boolean;
}

// One worker per page lifetime — `createSCS()` compiles the WASM blob
// only on first solve, then reuses it.
let sharedWorker: Worker | null = null;
let nextId = 1;

function getWorker(): Worker {
  if (!sharedWorker) {
    sharedWorker = new Worker(new URL("./hinf-worker.ts", import.meta.url));
  }
  return sharedWorker;
}

export interface SynthesizeHandle {
  promise: Promise<SynthesisResult>;
  cancel: () => void;
}

export function synthesizeInWorker(
  plantId: string,
  specs: SynthesisSpecs,
  onProgress?: (p: ProgressEvent) => void,
): SynthesizeHandle {
  const worker = getWorker();
  const id = nextId++;
  let listener: ((ev: MessageEvent<WorkerResponse>) => void) | null = null;

  const promise = new Promise<SynthesisResult>((resolve, reject) => {
    listener = (ev: MessageEvent<WorkerResponse>) => {
      const msg = ev.data;
      if (msg.id !== id) return;
      switch (msg.type) {
        case "progress":
          onProgress?.({
            iter: msg.iter,
            gamma: msg.gamma,
            bw: msg.bw,
            feasible: msg.feasible,
          });
          break;
        case "done":
          worker.removeEventListener("message", listener!);
          resolve(msg.result);
          break;
        case "error":
          worker.removeEventListener("message", listener!);
          reject(new Error(msg.message));
          break;
      }
    };
    worker.addEventListener("message", listener);
    const req: WorkerRequest = { id, type: "solve", plantId, specs };
    worker.postMessage(req);
  });

  return {
    promise,
    // Soft cancel: ask the worker to stop after the current bisection
    // iteration and return the best feasible iterate it has so far
    // (or the standard infeasible result if none was found yet). The
    // listener stays attached so the promise still resolves with the
    // worker's response — the caller's .then() handles both branches.
    cancel: () => {
      const req: WorkerRequest = { id, type: "cancel" };
      worker.postMessage(req);
    },
  };
}

// Explicit kill for dev / HMR / unmount-heavy scenarios.
export function disposeHinfWorker() {
  if (sharedWorker) {
    sharedWorker.terminate();
    sharedWorker = null;
  }
}
