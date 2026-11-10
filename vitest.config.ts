import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    environment: 'node',
    unstubEnvs: true,
    clearMocks: false,
    include: [
      'packages/*/src/**/__tests__/**/*.test.ts',
      'packages/*/test/**/*.test.ts',
    ],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      include: ['packages/*/src/**/*.{ts,js}'],
      exclude: ['packages/*/src/**/__tests__/**'],
    },
  },
})
