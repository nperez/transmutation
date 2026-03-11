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

// Package randtext provides compositional random text generation from large
// word pools. Designed to produce effectively infinite unique sentences,
// paragraphs, and structured text for training data generation.
package randtext

import "math/rand/v2"

func pick(rng *rand.Rand, items []string) string {
	return items[rng.IntN(len(items))]
}

// Nouns — infrastructure, software, data, general.
var nouns = []string{
	// Infrastructure
	"server", "database", "cache", "queue", "pipeline", "cluster", "node",
	"instance", "container", "volume", "network", "socket", "port", "gateway",
	"proxy", "router", "switch", "bridge", "endpoint", "firewall", "tunnel",
	"subnet", "interface", "balancer", "replica", "partition", "shard",
	// Software components
	"service", "handler", "controller", "middleware", "plugin", "module",
	"extension", "driver", "adapter", "connector", "resolver", "validator",
	"formatter", "serializer", "parser", "compiler", "loader", "runner",
	"scheduler", "dispatcher", "listener", "watcher", "monitor", "tracker",
	"logger", "profiler", "scanner", "analyzer", "optimizer", "allocator",
	"collector", "generator", "builder", "factory", "provider", "manager",
	"broker", "agent", "worker", "executor", "daemon", "timer",
	// Data structures
	"buffer", "stream", "channel", "signal", "event", "trigger", "hook",
	"callback", "filter", "transformer", "reducer", "mapper", "iterator",
	"cursor", "pointer", "reference", "handle", "context", "session",
	"transaction", "record", "entry", "field", "column", "row", "table",
	"index", "key", "value", "token", "credential", "certificate", "secret",
	// Messages & protocol
	"header", "payload", "body", "response", "request", "query", "command",
	"message", "packet", "frame", "segment", "block", "chunk", "batch",
	// State & control
	"snapshot", "checkpoint", "backup", "archive", "log", "metric", "trace",
	"span", "alert", "notification", "warning", "error", "exception", "fault",
	"timeout", "retry", "fallback", "circuit", "limiter", "throttle", "gate",
	"lock", "mutex", "semaphore", "barrier", "fence", "latch",
	// Config & metadata
	"flag", "toggle", "config", "setting", "option", "parameter", "argument",
	"variable", "constant", "expression", "statement", "function", "method",
	"procedure", "operation", "task", "job", "step", "phase", "stage",
	// General
	"cycle", "iteration", "epoch", "attempt", "sample", "template", "pattern",
	"schema", "model", "layout", "format", "encoding", "protocol", "standard",
	"specification", "contract", "boundary", "layer", "tier", "level",
	"zone", "region", "namespace", "domain", "scope", "environment",
	"runtime", "framework", "platform", "infrastructure", "architecture",
	"topology", "graph", "tree", "heap", "stack", "pool", "store",
	"registry", "repository", "catalog", "inventory", "ledger", "journal",
	// People & roles
	"user", "admin", "developer", "operator", "reviewer", "owner", "author",
	"consumer", "producer", "subscriber", "publisher", "sender", "receiver",
	"client", "tenant", "member", "contributor", "maintainer",
	// Concepts
	"policy", "permission", "role", "claim", "rule", "constraint", "threshold",
	"budget", "quota", "limit", "capacity", "bandwidth", "latency", "throughput",
	"availability", "durability", "consistency", "integrity", "compliance",
	"redundancy", "resilience", "scalability", "observability", "idempotency",
}

// Verbs — actions related to software/infrastructure.
var verbs = []string{
	"configure", "deploy", "initialize", "restart", "validate", "process",
	"schedule", "monitor", "update", "migrate", "provision", "terminate",
	"replicate", "synchronize", "compress", "encrypt", "decrypt", "serialize",
	"deserialize", "parse", "compile", "execute", "invoke", "dispatch",
	"route", "forward", "relay", "buffer", "cache", "evict", "flush",
	"drain", "allocate", "release", "acquire", "revoke", "rotate", "refresh",
	"retry", "throttle", "limit", "scale", "balance", "distribute", "partition",
	"shard", "index", "query", "filter", "transform", "aggregate", "reduce",
	"merge", "split", "truncate", "archive", "restore", "rollback", "commit",
	"checkpoint", "snapshot", "export", "import", "publish", "subscribe",
	"broadcast", "notify", "alert", "acknowledge", "resolve", "escalate",
	"delegate", "authorize", "authenticate", "verify", "audit", "inspect",
	"scan", "probe", "benchmark", "profile", "trace", "instrument",
	"register", "deregister", "bind", "unbind", "mount", "unmount",
	"attach", "detach", "connect", "disconnect", "enable", "disable",
	"activate", "deactivate", "suspend", "resume", "pause", "abort",
	"cancel", "timeout", "expire", "renew", "extend", "override",
	"patch", "upgrade", "downgrade", "deprecate", "remove", "prune",
	"clean", "purge", "reset", "reboot", "failover", "recover",
	"heal", "remediate", "isolate", "quarantine", "redact", "sanitize",
	"normalize", "denormalize", "interpolate", "extrapolate", "correlate",
	"annotate", "label", "tag", "classify", "prioritize", "enqueue",
	"dequeue", "push", "pop", "peek", "yield", "await", "poll",
	"fetch", "load", "store", "read", "write", "append", "prepend",
	"insert", "delete", "upsert", "truncate",
}

// Adjectives — descriptors for nouns.
var adjectives = []string{
	"active", "idle", "pending", "failed", "stale", "healthy", "degraded",
	"primary", "secondary", "tertiary", "internal", "external", "upstream",
	"downstream", "inbound", "outbound", "synchronous", "asynchronous",
	"transient", "persistent", "volatile", "immutable", "mutable", "readonly",
	"ephemeral", "durable", "distributed", "centralized", "federated",
	"replicated", "partitioned", "sharded", "cached", "indexed", "compressed",
	"encrypted", "authenticated", "authorized", "throttled", "rate-limited",
	"deprecated", "experimental", "stable", "unstable", "critical", "optional",
	"mandatory", "conditional", "recursive", "iterative", "concurrent",
	"sequential", "parallel", "blocking", "non-blocking", "lazy", "eager",
	"greedy", "conservative", "aggressive", "graceful", "abrupt", "partial",
	"complete", "incremental", "cumulative", "atomic", "eventual", "strict",
	"relaxed", "optimistic", "pessimistic", "stateful", "stateless",
	"deterministic", "idempotent", "reentrant", "thread-safe", "lock-free",
	"wait-free", "bounded", "unbounded", "weighted", "unweighted",
	"normalized", "denormalized", "validated", "sanitized", "raw",
	"cooked", "buffered", "unbuffered", "batched", "streaming",
	"default", "custom", "legacy", "modern", "provisional", "canonical",
}

// Adverbs — modify verbs.
var adverbs = []string{
	"automatically", "manually", "periodically", "continuously", "immediately",
	"eventually", "gradually", "intermittently", "consistently", "reliably",
	"securely", "efficiently", "optimally", "transparently", "silently",
	"explicitly", "implicitly", "conditionally", "unconditionally",
	"sequentially", "concurrently", "atomically", "idempotently",
	"incrementally", "recursively", "lazily", "eagerly", "gracefully",
	"aggressively", "conservatively", "temporarily", "permanently",
	"partially", "completely", "successfully", "incorrectly",
}

// Prepositions & connectors.
var prepositions = []string{
	"in", "on", "at", "from", "to", "with", "without", "before", "after",
	"during", "between", "across", "through", "above", "below", "within",
	"beyond", "against", "alongside", "underneath", "behind", "around",
}

var conjunctions = []string{
	"and", "but", "or", "so", "because", "although", "while", "when",
	"if", "unless", "until", "since", "whereas", "however", "therefore",
	"moreover", "furthermore", "nevertheless", "consequently", "otherwise",
}

// Technical proper nouns — product/service names.
var techNames = []string{
	"PostgreSQL", "MySQL", "Redis", "MongoDB", "Cassandra", "CockroachDB",
	"ClickHouse", "TimescaleDB", "SQLite", "DynamoDB", "BigQuery",
	"Elasticsearch", "Solr", "Memcached", "etcd", "ZooKeeper", "Consul",
	"Kafka", "RabbitMQ", "NATS", "Pulsar", "SQS", "Kinesis",
	"Kubernetes", "Docker", "Podman", "Terraform", "Ansible", "Puppet",
	"Helm", "ArgoCD", "Istio", "Envoy", "Nginx", "HAProxy", "Caddy",
	"Prometheus", "Grafana", "Datadog", "Splunk", "Jaeger", "Zipkin",
	"Sentry", "PagerDuty", "OpsGenie", "Vault", "Keycloak",
	"Jenkins", "GitLab", "GitHub", "CircleCI", "TravisCI", "Buildkite",
	"React", "Vue", "Angular", "Svelte", "Next.js", "Remix", "Nuxt",
	"Django", "Flask", "FastAPI", "Rails", "Spring", "Express", "Gin",
	"Actix", "Rocket", "Laravel", "Phoenix", "Fiber",
	"Python", "JavaScript", "TypeScript", "Go", "Rust", "Java", "Kotlin",
	"Swift", "C#", "Ruby", "PHP", "Scala", "Elixir", "Haskell", "Lua",
	"gRPC", "GraphQL", "REST", "WebSocket", "MQTT", "AMQP",
	"AWS", "GCP", "Azure", "Cloudflare", "Vercel", "Fly.io", "Heroku",
	"S3", "Lambda", "ECS", "EKS", "CloudFront", "Route53",
	"Cloud Run", "Cloud Functions", "App Engine", "Spanner",
	"Linux", "Ubuntu", "Debian", "Alpine", "CentOS", "RHEL",
	"macOS", "Windows", "FreeBSD",
}

// Quantities — numbers with units.
var quantities = []string{
	"100", "200", "500", "1000", "2048", "4096", "8192", "16384",
	"10ms", "50ms", "100ms", "250ms", "500ms", "1s", "5s", "10s", "30s", "60s",
	"1MB", "10MB", "64MB", "128MB", "256MB", "512MB", "1GB", "4GB", "16GB",
	"80%", "90%", "95%", "99%", "99.9%", "99.99%",
	"3", "5", "7", "10", "12", "15", "20", "25", "30", "50", "100",
	"1024", "2000", "5000", "10000", "50000", "100000",
	"8080", "8443", "3000", "3306", "5432", "6379", "9090", "27017",
}
