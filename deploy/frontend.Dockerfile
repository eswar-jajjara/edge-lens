FROM node:22-alpine AS build
WORKDIR /app
COPY package.json ./
COPY frontend ./frontend
COPY scripts/build.mjs ./scripts/build.mjs
RUN node scripts/build.mjs

FROM nginx:stable-alpine
COPY deploy/nginx/default.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
HEALTHCHECK --interval=15s --timeout=5s --retries=3 CMD wget -q -O /dev/null http://127.0.0.1/healthz || exit 1
