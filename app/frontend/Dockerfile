# --- build the static site ---
FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
# Same-origin API: the browser calls /api/... on this site and nginx forwards it to the backend
ENV VITE_API_BASE_URL=/
RUN npm run build

# --- serve it ---
FROM nginxinc/nginx-unprivileged:stable-alpine
COPY nginx.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html

# Cloud Run overrides PORT. BACKEND_HOST (hostname only, no https://) must be set at runtime,
# e.g. backend-xxxxx-uc.a.run.app
ENV PORT=8080
# Makes the nginx image fill in ${NGINX_LOCAL_RESOLVERS} (the container's own DNS servers) for nginx.conf.template
ENV NGINX_ENTRYPOINT_LOCAL_RESOLVERS=1
