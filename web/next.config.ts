import type { NextConfig } from "next";

const FASTAPI_ORIGIN = process.env.FASTAPI_ORIGIN ?? "http://localhost:8000";

const nextConfig: NextConfig = {
  // Rewrite /api/* requests to the FastAPI backend so the browser sees a
  // same-origin request and we sidestep CORS entirely during dev.
  // CORS is also configured in api/main.py as belt-and-suspenders for
  // any direct fetches.
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${FASTAPI_ORIGIN}/:path*`,
      },
    ];
  },
};

export default nextConfig;
