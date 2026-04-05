module.exports = {
  preset: 'ts-jest/presets/js-with-ts',
  testEnvironment: 'jsdom',
  roots: ['<rootDir>/tests'],
  setupFilesAfterEnv: ['<rootDir>/tests/setup/jest.setup.js'],
  transform: {
    '^.+\\.[tj]sx?$': ['ts-jest', { tsconfig: '<rootDir>/tsconfig.json' }]
  },
  moduleFileExtensions: ['js', 'jsx', 'ts', 'tsx', 'json'],
  testPathIgnorePatterns: ['/node_modules/', '/tests/css/.*\\.spec\\.js$']
};

