import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Allow cross-origin requests from development environments
  allowedDevOrigins: [
    '192.168.4.226',
    'localhost',
    '127.0.0.1',
    '0.0.0.0'
  ],
  
  // Optimize for development performance
  devIndicators: {
    position: 'bottom-right'
  },

  // Enable source maps for better debugging
  productionBrowserSourceMaps: false,
  
  // Configure headers for better development experience
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'X-Frame-Options',
            value: 'DENY'
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff'
          },
          {
            key: 'Referrer-Policy',
            value: 'origin-when-cross-origin'
          }
        ]
      }
    ]
  }
};

export default nextConfig;
