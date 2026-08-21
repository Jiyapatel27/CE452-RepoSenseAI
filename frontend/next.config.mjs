import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/** @type {import('next').NextConfig} */
const nextConfig = {
  turbopack: {
    // Next.js infers the project root by walking up for a lockfile. There is a
    // stray package-lock.json in C:\Users\jiyak, so it walked all the way to
    // the home directory and warned that it was ignoring it. Pinning the root
    // to this folder removes the warning and keeps file watching scoped to the
    // frontend.
    root: __dirname,
  },
};

export default nextConfig;
