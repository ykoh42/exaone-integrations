import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    environment: 'node',
    unstubEnvs: true,
    clearMocks: false,
    include: [
      'typescript/*/src/**/__tests__/**/*.test.ts',
      'typescript/*/test/**/*.test.ts',
    ],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      include: ['typescript/*/src/**/*.{ts,js}'],
      exclude: ['typescript/*/src/**/__tests__/**'],
    },
  },
})
