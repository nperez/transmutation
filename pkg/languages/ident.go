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

// Compositional identifier generation for high-entropy code snippets.
// Instead of picking from flat pools of 10 items, we combine parts
// to produce 1000s of unique identifiers.

var (
	// Function name parts — prefix × suffix = 20×20 = 400 combinations.
	funcPrefixes = []string{
		"get", "set", "create", "delete", "update", "find", "parse", "build",
		"check", "load", "save", "init", "run", "start", "stop", "send",
		"fetch", "validate", "compute", "process",
	}
	funcSuffixes = []string{
		"User", "Data", "Config", "Item", "Record", "Entry", "Field", "Value",
		"Result", "Response", "Request", "Event", "Task", "Job", "Message",
		"Token", "Session", "Cache", "Buffer", "Stream",
	}

	// Variable name parts.
	varBases = []string{
		"tmp", "val", "key", "idx", "cnt", "sum", "cur", "prev", "next",
		"src", "dst", "buf", "msg", "err", "res", "req", "ctx", "cfg",
		"out", "ret", "acc", "len", "pos", "ptr", "ref", "obj", "num",
		"str", "flag", "tag",
	}
	varQualifiers = []string{
		"user", "order", "item", "product", "account", "payment", "event",
		"task", "message", "session", "token", "config", "request", "response",
		"query", "record", "entry", "field", "cache", "batch",
	}
	varProperties = []string{
		"id", "name", "type", "status", "count", "total", "size", "index",
		"key", "value", "code", "path", "url", "data", "body", "hash",
		"time", "date", "level", "score",
	}

	// Table/collection names — qualifier × entity = 20×15 = 300.
	tableEntities = []string{
		"users", "orders", "products", "sessions", "logs", "accounts",
		"payments", "events", "tasks", "messages", "tokens", "configs",
		"records", "entries", "metrics",
	}
	tablePrefixes = []string{
		"", "app_", "sys_", "tmp_", "audit_", "user_", "admin_",
	}

	// Column name parts — reuse varQualifiers × varProperties.

	// String literal building blocks.
	stringAdjectives = []string{
		"invalid", "missing", "expired", "active", "pending", "failed",
		"completed", "blocked", "archived", "updated", "new", "old",
		"primary", "secondary", "default", "custom", "internal", "external",
	}
	stringNouns = []string{
		"request", "response", "connection", "session", "token", "credential",
		"permission", "configuration", "parameter", "resource", "endpoint",
		"transaction", "operation", "process", "service", "handler",
	}

	// Type name parts.
	typePrefixes = []string{
		"Base", "Abstract", "Default", "Custom", "Remote", "Local",
		"Async", "Cached", "Pooled", "Lazy",
	}
	typeRoots = []string{
		"Handler", "Processor", "Manager", "Service", "Client", "Factory",
		"Provider", "Controller", "Adapter", "Resolver", "Validator",
		"Formatter", "Converter", "Builder", "Runner",
	}
)

// FuncName generates a random function name (camelCase).
func FuncName(rng *rand.Rand) string {
	return pick(rng, funcPrefixes) + pick(rng, funcSuffixes)
}

// SnakeFuncName generates a random function name (snake_case).
func SnakeFuncName(rng *rand.Rand) string {
	return pick(rng, funcPrefixes) + "_" + strings.ToLower(pick(rng, funcSuffixes))
}

// VarName generates a random variable name.
func VarName(rng *rand.Rand) string {
	switch rng.IntN(3) {
	case 0:
		return pick(rng, varBases)
	case 1:
		return pick(rng, varQualifiers) + "_" + pick(rng, varProperties)
	default:
		return pick(rng, varBases) + fmt.Sprintf("%d", rng.IntN(100))
	}
}

// TableName generates a random SQL table name.
func TableName(rng *rand.Rand) string {
	return pick(rng, tablePrefixes) + pick(rng, tableEntities)
}

// ColumnName generates a random SQL column name.
func ColumnName(rng *rand.Rand) string {
	if rng.Float64() < 0.4 {
		return pick(rng, varProperties)
	}
	return pick(rng, varQualifiers) + "_" + pick(rng, varProperties)
}

// TypeName generates a random type/class name (PascalCase).
func TypeName(rng *rand.Rand) string {
	if rng.Float64() < 0.4 {
		return pick(rng, typePrefixes) + pick(rng, typeRoots)
	}
	return pick(rng, funcSuffixes) + pick(rng, typeRoots)
}

// RandInt generates a random integer literal as a string.
func RandInt(rng *rand.Rand) string {
	switch rng.IntN(5) {
	case 0:
		return fmt.Sprintf("%d", rng.IntN(10)) // 0-9
	case 1:
		return fmt.Sprintf("%d", rng.IntN(100)) // 0-99
	case 2:
		return fmt.Sprintf("%d", rng.IntN(10000)) // 0-9999
	case 3:
		return fmt.Sprintf("%d", 1000+rng.IntN(99000)) // 1000-99999
	default:
		return fmt.Sprintf("%d", -rng.IntN(1000)) // negative
	}
}

// RandFloat generates a random float literal as a string.
func RandFloat(rng *rand.Rand) string {
	return fmt.Sprintf("%.2f", rng.Float64()*1000)
}

// RandPort generates a random port number as a string.
func RandPort(rng *rand.Rand) string {
	return fmt.Sprintf("%d", 1024+rng.IntN(64000))
}

// RandStringLiteral generates a random quoted string literal.
func RandStringLiteral(rng *rand.Rand) string {
	return "'" + pick(rng, stringAdjectives) + " " + pick(rng, stringNouns) + "'"
}

// RandURL generates a random URL.
func RandURL(rng *rand.Rand) string {
	paths := []string{"api", "v1", "v2", "auth", "data", "admin", "public", "internal"}
	resources := []string{"users", "items", "orders", "config", "status", "health", "metrics", "events"}
	n := 1 + rng.IntN(3)
	parts := make([]string, n)
	for i := range parts {
		if i == n-1 {
			parts[i] = pick(rng, resources)
		} else {
			parts[i] = pick(rng, paths)
		}
	}
	return "https://example.com/" + strings.Join(parts, "/")
}

// RandPath generates a random file path.
func RandPath(rng *rand.Rand) string {
	dirs := []string{"src", "lib", "config", "data", "tmp", "logs", "output", "input", "assets", "public"}
	exts := []string{".json", ".yaml", ".txt", ".csv", ".log", ".py", ".js", ".go", ".sql", ".xml"}
	depth := 1 + rng.IntN(3)
	parts := make([]string, depth)
	for i := range parts {
		parts[i] = pick(rng, dirs)
	}
	return "/" + strings.Join(parts, "/") + "/" + pick(rng, varQualifiers) + pick(rng, exts)
}

// RandSQLValue generates a random SQL value (number, string, or keyword).
func RandSQLValue(rng *rand.Rand) string {
	switch rng.IntN(5) {
	case 0:
		return RandInt(rng)
	case 1:
		return RandStringLiteral(rng)
	case 2:
		return "NULL"
	case 3:
		return "TRUE"
	default:
		return "NOW()"
	}
}
