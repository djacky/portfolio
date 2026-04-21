/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  webpack: (config, { dev, isServer }) => {
    if (dev) {
      config.cache = { type: "memory" };
    }
    // scs-solver's WASM loader references Node builtins (module, fs, path).
    // Stub them out in the client bundle so the dynamic import doesn't
    // fail to resolve during webpack's static analysis.
    if (!isServer) {
      config.resolve.fallback = {
        ...(config.resolve.fallback || {}),
        module: false,
        fs: false,
        path: false,
        crypto: false,
      };
    }
    return config;
  },
};
export default nextConfig;
