FROM node:20-slim

WORKDIR /benchmark

ENV CI=1

COPY package.json package-lock.json ./

RUN npm ci --ignore-scripts

RUN npx playwright install --with-deps chromium

COPY jest.config.js playwright.config.js tsconfig.json README.md ./
COPY tests/ tests/
COPY tasks/ tasks/
COPY solutions/ solutions/
COPY fixtures/ fixtures/
COPY scripts/ scripts/

RUN chmod +x scripts/*.sh

CMD ["bash", "scripts/validate-all.sh"]
