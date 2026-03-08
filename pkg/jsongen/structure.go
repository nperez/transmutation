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
	"math/rand/v2"
)

// NodeKind represents the type of a JSON node.
type NodeKind int

const (
	KindObject NodeKind = iota
	KindArray
	KindString
	KindNumber
	KindBool
	KindNull
)

// Node represents a node in a JSON tree.
type Node struct {
	Kind     NodeKind
	Key      string // only set for object entries
	Value    any    // string, float64, bool, nil for scalars
	Children []*Node
}

// Config controls the shape of generated JSON structures.
type Config struct {
	MaxDepth   int     // maximum nesting depth (1-8)
	MaxBreadth int     // maximum children per object/array (1-20)
	MinBreadth int     // minimum children per object/array
	SizeBudget int     // approximate total node count budget
	ArrayProb  float64 // probability of choosing array over object for containers
	ScalarDist ScalarDistribution
}

// ScalarDistribution controls how often each scalar type appears.
type ScalarDistribution struct {
	String float64
	Number float64
	Bool   float64
	Null   float64
}

// DefaultConfig returns a reasonable default configuration.
func DefaultConfig() Config {
	return Config{
		MaxDepth:   5,
		MaxBreadth: 8,
		MinBreadth: 1,
		SizeBudget: 50,
		ArrayProb:  0.3,
		ScalarDist: ScalarDistribution{
			String: 0.5,
			Number: 0.25,
			Bool:   0.15,
			Null:   0.10,
		},
	}
}

// Generator builds random JSON trees.
type Generator struct {
	cfg       Config
	rng       *rand.Rand
	nodeCount int
}

// NewGenerator creates a new JSON structure generator.
func NewGenerator(cfg Config, rng *rand.Rand) *Generator {
	return &Generator{cfg: cfg, rng: rng}
}

// Generate produces a random JSON tree. The root is always an object or array.
func (g *Generator) Generate() *Node {
	g.nodeCount = 0
	if g.rng.Float64() < g.cfg.ArrayProb {
		return g.genArray(0)
	}
	return g.genObject(0)
}

func (g *Generator) genObject(depth int) *Node {
	g.nodeCount++
	n := &Node{Kind: KindObject}
	breadth := g.randBreadth()
	usedKeys := make(map[string]bool)
	for i := 0; i < breadth && g.nodeCount < g.cfg.SizeBudget; i++ {
		key := g.uniqueKey(usedKeys)
		child := g.genValue(depth + 1)
		child.Key = key
		n.Children = append(n.Children, child)
	}
	return n
}

func (g *Generator) genArray(depth int) *Node {
	g.nodeCount++
	n := &Node{Kind: KindArray}
	breadth := g.randBreadth()
	for i := 0; i < breadth && g.nodeCount < g.cfg.SizeBudget; i++ {
		child := g.genValue(depth + 1)
		n.Children = append(n.Children, child)
	}
	return n
}

func (g *Generator) genValue(depth int) *Node {
	// If we're at max depth or near budget, emit a scalar.
	if depth >= g.cfg.MaxDepth || g.nodeCount >= g.cfg.SizeBudget-1 {
		return g.genScalar()
	}

	// Scale container probability based on how much budget remains.
	// When we've used very little of the budget, strongly prefer containers.
	budgetUsed := float64(g.nodeCount) / float64(g.cfg.SizeBudget)
	depthRatio := float64(depth) / float64(g.cfg.MaxDepth)

	// Start at 70% container probability, taper off with depth and budget usage.
	containerProb := 0.7 * (1.0 - depthRatio) * (1.0 - budgetUsed*0.5)
	if g.rng.Float64() < containerProb {
		if g.rng.Float64() < g.cfg.ArrayProb {
			return g.genArray(depth)
		}
		return g.genObject(depth)
	}
	return g.genScalar()
}

func (g *Generator) genScalar() *Node {
	g.nodeCount++
	r := g.rng.Float64()
	dist := g.cfg.ScalarDist
	switch {
	case r < dist.String:
		return &Node{Kind: KindString, Value: ""}
	case r < dist.String+dist.Number:
		return &Node{Kind: KindNumber, Value: g.randNumber()}
	case r < dist.String+dist.Number+dist.Bool:
		return &Node{Kind: KindBool, Value: g.rng.Float64() < 0.5}
	default:
		return &Node{Kind: KindNull}
	}
}

func (g *Generator) randBreadth() int {
	lo := max(g.cfg.MinBreadth, 1)
	hi := max(g.cfg.MaxBreadth, lo)
	return lo + g.rng.IntN(hi-lo+1)
}

func (g *Generator) randNumber() float64 {
	// Mix of integers and floats, small and large.
	switch g.rng.IntN(4) {
	case 0:
		return float64(g.rng.IntN(100))
	case 1:
		return float64(g.rng.IntN(1000000))
	case 2:
		return g.rng.Float64() * 100.0
	default:
		return -float64(g.rng.IntN(1000))
	}
}

func (g *Generator) uniqueKey(used map[string]bool) string {
	for range 100 {
		key := randomKey(g.rng)
		if !used[key] {
			used[key] = true
			return key
		}
	}
	// Fallback: append a number.
	key := randomKey(g.rng)
	for i := 0; used[key]; i++ {
		key = randomKey(g.rng) + "_" + itoa(i)
	}
	used[key] = true
	return key
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	s := ""
	for n > 0 {
		s = string(rune('0'+n%10)) + s
		n /= 10
	}
	return s
}

// Key pools for realistic JSON field names.
var (
	camelKeys = []string{
		"id", "name", "type", "status", "message", "data", "result",
		"error", "code", "value", "count", "total", "items", "list",
		"userId", "userName", "firstName", "lastName", "email",
		"createdAt", "updatedAt", "deletedAt", "startTime", "endTime",
		"isActive", "isEnabled", "isValid", "hasPermission",
		"requestId", "responseCode", "statusCode", "errorMessage",
		"pageSize", "pageNumber", "totalPages", "totalCount",
		"contentType", "authorization", "apiKey", "accessToken",
		"description", "title", "summary", "content", "body",
		"url", "path", "method", "headers", "params", "query",
		"action", "thought", "observation", "actionInput",
		"toolName", "toolInput", "toolOutput", "reasoning",
		"prompt", "response", "model", "temperature", "maxTokens",
	}

	snakeKeys = []string{
		"id", "name", "type", "status", "message", "data", "result",
		"error", "code", "value", "count", "total", "items", "list",
		"user_id", "user_name", "first_name", "last_name", "email",
		"created_at", "updated_at", "deleted_at", "start_time", "end_time",
		"is_active", "is_enabled", "is_valid", "has_permission",
		"request_id", "response_code", "status_code", "error_message",
		"page_size", "page_number", "total_pages", "total_count",
		"content_type", "authorization", "api_key", "access_token",
		"description", "title", "summary", "content", "body",
		"url", "path", "method", "headers", "params", "query",
		"action", "thought", "observation", "action_input",
		"tool_name", "tool_input", "tool_output", "reasoning",
		"prompt", "response", "model", "temperature", "max_tokens",
	}
)

func randomKey(rng *rand.Rand) string {
	if rng.Float64() < 0.5 {
		return camelKeys[rng.IntN(len(camelKeys))]
	}
	return snakeKeys[rng.IntN(len(snakeKeys))]
}
