/* ------------------------------------------------------------------ *
 *  Redis client (node-redis v5, standard TCP connection).             *
 *                                                                      *
 *  Used by the global pendulum training counter. Connects lazily       *
 *  on the first call and caches the client for reuse across warm       *
 *  serverless invocations, so we don't eat a connection round-trip     *
 *  on every request. Returns null if REDIS_URL is unset (local dev     *
 *  without the integration), so callers can degrade gracefully.       *
 * ------------------------------------------------------------------ */

import { createClient, type RedisClientType } from "redis";

type LazyClient = Promise<RedisClientType> | null;

// Stash the client on globalThis so hot-module reloads in dev don't
// leak connections every time a route handler is re-imported.
const globalForRedis = globalThis as unknown as { __redisClient?: LazyClient };

export async function getRedis(): Promise<RedisClientType | null> {
  if (!process.env.REDIS_URL) return null;

  if (!globalForRedis.__redisClient) {
    const c = createClient({ url: process.env.REDIS_URL });
    c.on("error", (err) => {
      console.error("redis client error:", err);
      // Drop the cached promise so the next call tries a fresh
      // connection. node-redis also auto-reconnects, but this
      // protects us if `connect()` itself rejected.
      globalForRedis.__redisClient = null;
    });
    globalForRedis.__redisClient = c.connect() as Promise<RedisClientType>;
  }

  try {
    return await globalForRedis.__redisClient;
  } catch (err) {
    console.error("redis connect failed:", err);
    globalForRedis.__redisClient = null;
    return null;
  }
}
