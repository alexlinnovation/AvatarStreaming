// const createNextPluginPreval = require("next-plugin-preval/config");
// const withNextPluginPreval = createNextPluginPreval();

// /** @type {import('next').NextConfig} */
// const nextConfig = {
//   reactStrictMode: false,
// };

// module.exports = withNextPluginPreval(nextConfig);

const createNextPluginPreval = require('next-plugin-preval/config');
const withNextPluginPreval   = createNextPluginPreval();

/** @type {import('next').NextConfig} */
module.exports = withNextPluginPreval({
  reactStrictMode: false,

  // ⬇️  tells Next.js to produce the static `out/` folder on `npm run build`
  output: 'export',
});