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

type JSON struct{}

func (JSON) Name() string { return "json" }

// Generate produces stringified JSON — JSON-as-a-string, common in agent APIs
// where a response field contains serialized JSON.
func (JSON) Generate(rng *rand.Rand) string {
	switch rng.IntN(4) {
	case 0:
		return genJSONFlat(rng)
	case 1:
		return genJSONNested(rng)
	case 2:
		return genJSONArray(rng)
	default:
		return genJSONConfig(rng)
	}
}

var (
	jsonKeys = []string{"id", "name", "type", "status", "value", "data", "message", "code", "result", "error", "url", "path", "count", "enabled", "version"}
	jsonVals = []string{`"active"`, `"pending"`, `"error"`, `42`, `true`, `false`, `null`, `"hello"`, `3.14`, `""`, `"2024-01-15T10:30:00Z"`, `"admin"`}
)

func genJSONFlat(rng *rand.Rand) string {
	pairs := 2 + rng.IntN(5)
	keys := pickN(rng, jsonKeys, pairs)
	parts := make([]string, len(keys))
	for i, k := range keys {
		parts[i] = fmt.Sprintf(`"%s": %s`, k, pick(rng, jsonVals))
	}
	return "{" + strings.Join(parts, ", ") + "}"
}

func genJSONNested(rng *rand.Rand) string {
	inner := genJSONFlat(rng)
	outerKey := pick(rng, jsonKeys)
	wrapperKey := pick(rng, []string{"data", "result", "payload", "response", "body"})
	return fmt.Sprintf(`{"%s": "%s", "%s": %s}`,
		pick(rng, jsonKeys), pick(rng, jsonVals),
		wrapperKey, fmt.Sprintf(`{"%s": %s}`, outerKey, inner))
}

func genJSONArray(rng *rand.Rand) string {
	items := 2 + rng.IntN(5)
	elements := make([]string, items)
	for i := range elements {
		elements[i] = genJSONFlat(rng)
	}
	return "[" + strings.Join(elements, ", ") + "]"
}

func genJSONConfig(rng *rand.Rand) string {
	return fmt.Sprintf(`{"host": "%s", "port": %d, "database": "%s", "ssl": %s, "pool_size": %d, "timeout_ms": %d}`,
		pick(rng, []string{"localhost", "db.internal", "10.0.1.5", "postgres.example.com"}),
		1000+rng.IntN(60000),
		pick(rng, []string{"myapp", "production", "analytics", "users"}),
		pick(rng, []string{"true", "false"}),
		5+rng.IntN(95),
		100+rng.IntN(9900))
}
