// Copyright (C) 2026 Nicholas Perez
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

package languages

import (
	"fmt"
	"math/rand/v2"
	"strings"
)

type YAML struct{}

func (YAML) Name() string { return "yaml" }

func (YAML) Generate(rng *rand.Rand) string {
	switch rng.IntN(4) {
	case 0:
		return genYAMLConfig(rng)
	case 1:
		return genYAMLDocker(rng)
	case 2:
		return genYAMLCI(rng)
	default:
		return genYAMLK8s(rng)
	}
}

func genYAMLConfig(rng *rand.Rand) string {
	lines := []string{
		fmt.Sprintf("app_name: %s", pick(rng, []string{"myapp", "api-server", "worker", "gateway", "scheduler"})),
		fmt.Sprintf("version: %d.%d.%d", rng.IntN(3), rng.IntN(10), rng.IntN(20)),
		fmt.Sprintf("environment: %s", pick(rng, []string{"production", "staging", "development", "testing"})),
		"",
		"database:",
		fmt.Sprintf("  host: %s", pick(rng, []string{"localhost", "db.internal", "postgres-primary", "10.0.1.5"})),
		"  port: " + pick(rng, []string{"5432", "3306", "27017"}),
		fmt.Sprintf("  name: %s", pick(rng, []string{"app_db", "main", "analytics", "users"})),
		"  pool_size: " + pick(rng, []string{"10", "25", "50", "100"}),
		"",
		"logging:",
		fmt.Sprintf("  level: %s", pick(rng, []string{"debug", "info", "warn", "error"})),
		fmt.Sprintf("  format: %s", pick(rng, []string{"json", "text", "logfmt"})),
	}
	if rng.Float64() < 0.5 {
		lines = append(lines, "", "features:", "  - "+pick(rng, []string{"auth", "caching", "rate_limiting", "metrics"}),
			"  - "+pick(rng, []string{"websockets", "grpc", "graphql", "webhooks"}))
	}
	return strings.Join(lines, "\n")
}

func genYAMLDocker(rng *rand.Rand) string {
	service := pick(rng, []string{"web", "api", "worker", "db", "redis", "nginx"})
	return fmt.Sprintf(`version: "3.8"

services:
  %s:
    image: %s
    ports:
      - "%d:%d"
    environment:
      - NODE_ENV=%s
      - DATABASE_URL=%s
    volumes:
      - ./%s:/app
      - /app/node_modules
    depends_on:
      - %s
    restart: %s`,
		service,
		pick(rng, []string{"node:20-alpine", "python:3.12-slim", "golang:1.22", "nginx:alpine", "postgres:16"}),
		3000+rng.IntN(5000), 3000+rng.IntN(5000),
		pick(rng, []string{"production", "development"}),
		pick(rng, []string{"postgres://user:pass@db:5432/app", "mysql://root:pass@db:3306/app"}),
		pick(rng, []string{"src", "app", ".", "backend"}),
		pick(rng, []string{"db", "redis", "rabbitmq"}),
		pick(rng, []string{"always", "unless-stopped", "on-failure"}))
}

func genYAMLCI(rng *rand.Rand) string {
	return fmt.Sprintf(`name: %s

on:
  push:
    branches: [%s]
  pull_request:
    branches: [%s]

jobs:
  build:
    runs-on: %s
    steps:
      - uses: actions/checkout@v4
      - name: Setup
        uses: %s
      - name: Install dependencies
        run: %s
      - name: Run tests
        run: %s`,
		pick(rng, []string{"CI", "Build & Test", "Deploy", "Lint & Test"}),
		pick(rng, []string{"main", "master", "develop"}),
		pick(rng, []string{"main", "master", "develop"}),
		pick(rng, []string{"ubuntu-latest", "ubuntu-22.04", "macos-latest"}),
		pick(rng, []string{"actions/setup-node@v4", "actions/setup-go@v5", "actions/setup-python@v5"}),
		pick(rng, []string{"npm ci", "go mod download", "pip install -r requirements.txt", "yarn install --frozen-lockfile"}),
		pick(rng, []string{"npm test", "go test ./...", "pytest", "yarn test --coverage"}))
}

func genYAMLK8s(rng *rand.Rand) string {
	name := pick(rng, []string{"myapp", "api-server", "web-frontend", "worker", "cron-job"})
	return fmt.Sprintf(`apiVersion: apps/v1
kind: Deployment
metadata:
  name: %s
  labels:
    app: %s
spec:
  replicas: %d
  selector:
    matchLabels:
      app: %s
  template:
    metadata:
      labels:
        app: %s
    spec:
      containers:
        - name: %s
          image: %s/%s:%s
          ports:
            - containerPort: %d
          resources:
            requests:
              memory: "%s"
              cpu: "%s"
            limits:
              memory: "%s"
              cpu: "%s"`,
		name, name, 1+rng.IntN(5), name, name, name,
		pick(rng, []string{"gcr.io/myproject", "docker.io/myorg", "registry.example.com"}),
		name,
		pick(rng, []string{"latest", "v1.0.0", "sha-abc123"}),
		3000+rng.IntN(5000),
		pick(rng, []string{"64Mi", "128Mi", "256Mi"}),
		pick(rng, []string{"100m", "250m", "500m"}),
		pick(rng, []string{"128Mi", "256Mi", "512Mi"}),
		pick(rng, []string{"250m", "500m", "1000m"}))
}
