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

package jsongen

import (
	"encoding/json"
	"math/rand/v2"
	"strconv"
	"strings"
)

// ValuePopulator fills in string nodes with realistic content.
type ValuePopulator struct {
	rng        *rand.Rand
	generators []LanguageGenerator
}

// LanguageGenerator produces snippets of a specific language.
type LanguageGenerator interface {
	// Name returns the language name (for diagnostics).
	Name() string
	// Generate produces a random code snippet.
	Generate(rng *rand.Rand) string
}

// NewValuePopulator creates a populator with the given language generators.
func NewValuePopulator(rng *rand.Rand, generators []LanguageGenerator) *ValuePopulator {
	return &ValuePopulator{rng: rng, generators: generators}
}

// Populate fills in all empty string values in the tree.
func (vp *ValuePopulator) Populate(node *Node) {
	vp.walk(node)
}

func (vp *ValuePopulator) walk(node *Node) {
	if node.Kind == KindString && node.Value == "" {
		node.Value = vp.randomStringValue()
	}
	for _, child := range node.Children {
		vp.walk(child)
	}
}

// Simple string value pools.
var (
	simpleStrings = []string{
		"hello", "world", "foo", "bar", "baz", "test", "example",
		"Alice", "Bob", "Charlie", "admin", "user", "guest",
		"success", "failure", "pending", "active", "inactive",
		"2024-01-15", "2024-12-31T23:59:59Z", "127.0.0.1",
		"https://example.com/api/v1/users", "/home/user/.config",
		"application/json", "text/html", "Bearer eyJhbGciOiJIUzI1NiJ9",
		"The quick brown fox jumps over the lazy dog.",
		"Error: connection refused", "Operation completed successfully",
		"TODO: fix this later", "FIXME: handle edge case",
		"v2.1.0", "1.0.0-beta.3", "sha256:abc123def456",
	}

	sentenceStarts = []string{
		"The system", "This function", "An error occurred",
		"The user", "Processing", "The request",
		"A new", "The database", "The API",
		"Authentication", "The server", "Validation",
	}

	sentenceEnds = []string{
		"was successful.", "failed unexpectedly.",
		"is currently unavailable.", "has been updated.",
		"returned an error.", "completed in 3.2 seconds.",
		"requires authentication.", "was not found.",
		"exceeded the rate limit.", "is being processed.",
		"needs to be reviewed.", "contains invalid data.",
	}
)

func (vp *ValuePopulator) randomStringValue() string {
	// Decide what kind of string to generate.
	r := vp.rng.Float64()
	switch {
	case r < 0.30:
		// Simple string
		return simpleStrings[vp.rng.IntN(len(simpleStrings))]
	case r < 0.45:
		// Random sentence
		return sentenceStarts[vp.rng.IntN(len(sentenceStarts))] + " " +
			sentenceEnds[vp.rng.IntN(len(sentenceEnds))]
	case r < 0.55:
		// Multi-sentence
		n := 2 + vp.rng.IntN(3)
		parts := make([]string, n)
		for i := range parts {
			parts[i] = sentenceStarts[vp.rng.IntN(len(sentenceStarts))] + " " +
				sentenceEnds[vp.rng.IntN(len(sentenceEnds))]
		}
		return strings.Join(parts, " ")
	default:
		// Embedded language snippet (if generators available)
		if len(vp.generators) > 0 {
			gen := vp.generators[vp.rng.IntN(len(vp.generators))]
			return gen.Generate(vp.rng)
		}
		return simpleStrings[vp.rng.IntN(len(simpleStrings))]
	}
}

// Serialize converts a Node tree to a JSON string.
func Serialize(node *Node) string {
	var b strings.Builder
	serializeNode(&b, node, 0)
	return b.String()
}

func serializeNode(b *strings.Builder, node *Node, indent int) {
	prefix := strings.Repeat("  ", indent)
	switch node.Kind {
	case KindObject:
		b.WriteString("{\n")
		for i, child := range node.Children {
			b.WriteString(prefix)
			b.WriteString("  ")
			keyEncoded, _ := json.Marshal(child.Key)
		b.Write(keyEncoded)
			b.WriteString(": ")
			serializeNode(b, child, indent+1)
			if i < len(node.Children)-1 {
				b.WriteByte(',')
			}
			b.WriteByte('\n')
		}
		b.WriteString(prefix)
		b.WriteByte('}')
	case KindArray:
		b.WriteString("[\n")
		for i, child := range node.Children {
			b.WriteString(prefix)
			b.WriteString("  ")
			serializeNode(b, child, indent+1)
			if i < len(node.Children)-1 {
				b.WriteByte(',')
			}
			b.WriteByte('\n')
		}
		b.WriteString(prefix)
		b.WriteByte(']')
	case KindString:
		s, _ := node.Value.(string)
		encoded, _ := json.Marshal(s)
		b.Write(encoded)
	case KindNumber:
		f, _ := node.Value.(float64)
		if f == float64(int64(f)) && f >= -1e15 && f <= 1e15 {
			b.WriteString(strconv.FormatInt(int64(f), 10))
		} else {
			b.WriteString(strconv.FormatFloat(f, 'f', -1, 64))
		}
	case KindBool:
		v, _ := node.Value.(bool)
		if v {
			b.WriteString("true")
		} else {
			b.WriteString("false")
		}
	case KindNull:
		b.WriteString("null")
	}
}
