# NEXT_PUBLIC_* vars are inlined into the client bundle at build time, not
# read at container runtime -- so they're build ARGs here, and must be the
# browser-reachable addresses (published host ports), not compose service
# names. The browser runs outside the compose network entirely.

FROM node:20-slim AS deps
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci

FROM node:20-slim AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

ARG NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
ARG NEXT_PUBLIC_WHISPER_WS_URL=ws://localhost:8002/transcribe/ws
ARG NEXT_PUBLIC_TTS_API_BASE_URL=http://localhost:8003
ENV NEXT_PUBLIC_API_BASE_URL=$NEXT_PUBLIC_API_BASE_URL
ENV NEXT_PUBLIC_WHISPER_WS_URL=$NEXT_PUBLIC_WHISPER_WS_URL
ENV NEXT_PUBLIC_TTS_API_BASE_URL=$NEXT_PUBLIC_TTS_API_BASE_URL

RUN npm run build

FROM node:20-slim AS runner
WORKDIR /app
ENV NODE_ENV=production

COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json

EXPOSE 3000
CMD ["npm", "start"]
