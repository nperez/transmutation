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
	"strings"
	"testing"
)

func deterministicRNG(seed uint64) *rand.Rand {
	return rand.New(rand.NewPCG(seed, seed+1))
}

func TestGenerateProducesValidJSON(t *testing.T) {
	for seed := range uint64(100) {
		rng := deterministicRNG(seed)
		cfg := DefaultConfig()
		gen := NewGenerator(cfg, rng)
		node := gen.Generate()
		jsonStr := Serialize(node)

		var parsed any
		if err := json.Unmarshal([]byte(jsonStr), &parsed); err != nil {
			t.Errorf("seed %d: invalid JSON: %v\nJSON:\n%s", seed, err, jsonStr)
		}
	}
}

func TestGenerateRootIsContainerType(t *testing.T) {
	for seed := range uint64(50) {
		rng := deterministicRNG(seed)
		gen := NewGenerator(DefaultConfig(), rng)
		node := gen.Generate()
		if node.Kind != KindObject && node.Kind != KindArray {
			t.Errorf("seed %d: root kind is %d, want Object or Array", seed, node.Kind)
		}
	}
}

func TestGenerateRespectsMaxDepth(t *testing.T) {
	for _, maxDepth := range []int{1, 2, 3, 5, 8} {
		rng := deterministicRNG(42)
		cfg := DefaultConfig()
		cfg.MaxDepth = maxDepth
		cfg.SizeBudget = 500
		gen := NewGenerator(cfg, rng)
		node := gen.Generate()

		actual := measureDepth(node)
		if actual > maxDepth+1 { // +1 because root is depth 0 but counts as 1 level
			t.Errorf("maxDepth=%d but measured depth=%d", maxDepth, actual)
		}
	}
}

func measureDepth(node *Node) int {
	if len(node.Children) == 0 {
		return 1
	}
	maxChild := 0
	for _, child := range node.Children {
		d := measureDepth(child)
		if d > maxChild {
			maxChild = d
		}
	}
	return maxChild + 1
}

func TestGenerateRespectsSizeBudget(t *testing.T) {
	for _, budget := range []int{5, 10, 50, 200} {
		rng := deterministicRNG(99)
		cfg := DefaultConfig()
		cfg.SizeBudget = budget
		gen := NewGenerator(cfg, rng)
		node := gen.Generate()

		count := countNodes(node)
		// Allow some overshoot since we check budget at loop boundaries.
		if count > budget+cfg.MaxBreadth {
			t.Errorf("budget=%d but got %d nodes", budget, count)
		}
	}
}

func countNodes(node *Node) int {
	count := 1
	for _, child := range node.Children {
		count += countNodes(child)
	}
	return count
}

func TestGenerateObjectKeysAreUnique(t *testing.T) {
	for seed := range uint64(50) {
		rng := deterministicRNG(seed)
		cfg := DefaultConfig()
		cfg.MaxBreadth = 20
		gen := NewGenerator(cfg, rng)
		node := gen.Generate()
		checkUniqueKeys(t, node, seed)
	}
}

func checkUniqueKeys(t *testing.T, node *Node, seed uint64) {
	t.Helper()
	if node.Kind == KindObject {
		seen := make(map[string]bool)
		for _, child := range node.Children {
			if seen[child.Key] {
				t.Errorf("seed %d: duplicate key %q in object", seed, child.Key)
			}
			seen[child.Key] = true
		}
	}
	for _, child := range node.Children {
		checkUniqueKeys(t, child, seed)
	}
}

func TestGenerateAllNodeKindsAppear(t *testing.T) {
	// Over many seeds, all node kinds should appear.
	seen := make(map[NodeKind]bool)
	for seed := range uint64(200) {
		rng := deterministicRNG(seed)
		cfg := DefaultConfig()
		cfg.SizeBudget = 100
		gen := NewGenerator(cfg, rng)
		node := gen.Generate()
		collectKinds(node, seen)
	}
	for _, kind := range []NodeKind{KindObject, KindArray, KindString, KindNumber, KindBool, KindNull} {
		if !seen[kind] {
			t.Errorf("node kind %d never appeared across 200 seeds", kind)
		}
	}
}

func collectKinds(node *Node, seen map[NodeKind]bool) {
	seen[node.Kind] = true
	for _, child := range node.Children {
		collectKinds(child, seen)
	}
}

func TestSerializeEmptyObject(t *testing.T) {
	node := &Node{Kind: KindObject}
	s := Serialize(node)
	var parsed any
	if err := json.Unmarshal([]byte(s), &parsed); err != nil {
		t.Errorf("empty object not valid JSON: %v\nJSON: %s", err, s)
	}
}

func TestSerializeEmptyArray(t *testing.T) {
	node := &Node{Kind: KindArray}
	s := Serialize(node)
	var parsed any
	if err := json.Unmarshal([]byte(s), &parsed); err != nil {
		t.Errorf("empty array not valid JSON: %v\nJSON: %s", err, s)
	}
}

func TestSerializeScalars(t *testing.T) {
	tests := []struct {
		name string
		node *Node
		want string
	}{
		{"string", &Node{Kind: KindString, Value: "hello"}, `"hello"`},
		{"string with quotes", &Node{Kind: KindString, Value: `say "hi"`}, `"say \"hi\""`},
		{"integer", &Node{Kind: KindNumber, Value: 42.0}, "42"},
		{"float", &Node{Kind: KindNumber, Value: 3.14}, "3.14"},
		{"negative", &Node{Kind: KindNumber, Value: -7.0}, "-7"},
		{"true", &Node{Kind: KindBool, Value: true}, "true"},
		{"false", &Node{Kind: KindBool, Value: false}, "false"},
		{"null", &Node{Kind: KindNull}, "null"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Serialize(tt.node)
			if strings.TrimSpace(got) != tt.want {
				t.Errorf("got %q, want %q", got, tt.want)
			}
		})
	}
}

func TestSerializeNestedStructure(t *testing.T) {
	// Build: {"items": [1, "two", null]}
	node := &Node{
		Kind: KindObject,
		Children: []*Node{
			{
				Kind: KindArray,
				Key:  "items",
				Children: []*Node{
					{Kind: KindNumber, Value: 1.0},
					{Kind: KindString, Value: "two"},
					{Kind: KindNull},
				},
			},
		},
	}
	s := Serialize(node)
	var parsed map[string]any
	if err := json.Unmarshal([]byte(s), &parsed); err != nil {
		t.Fatalf("invalid JSON: %v\nJSON:\n%s", err, s)
	}
	items, ok := parsed["items"].([]any)
	if !ok || len(items) != 3 {
		t.Fatalf("unexpected structure: %v", parsed)
	}
}

func TestSerializeSpecialCharactersInStrings(t *testing.T) {
	specials := []string{
		"line1\nline2",
		"tab\there",
		`back\slash`,
		"null\x00byte",
		"emoji 🎉",
		"<html>&amp;</html>",
		`{"nested": "json"}`,
		"SELECT * FROM users WHERE name = 'alice'",
	}
	for _, s := range specials {
		node := &Node{Kind: KindString, Value: s}
		jsonStr := Serialize(node)
		var parsed string
		if err := json.Unmarshal([]byte(jsonStr), &parsed); err != nil {
			t.Errorf("special string %q: invalid JSON: %v\nJSON: %s", s, err, jsonStr)
			continue
		}
		if parsed != s {
			t.Errorf("round-trip mismatch: got %q, want %q", parsed, s)
		}
	}
}

func TestValuePopulatorFillsAllStrings(t *testing.T) {
	rng := deterministicRNG(42)
	cfg := DefaultConfig()
	cfg.SizeBudget = 100
	gen := NewGenerator(cfg, rng)
	node := gen.Generate()

	pop := NewValuePopulator(deterministicRNG(99), nil)
	pop.Populate(node)

	// Verify no empty strings remain.
	checkNoEmptyStrings(t, node)

	// Verify still valid JSON.
	jsonStr := Serialize(node)
	var parsed any
	if err := json.Unmarshal([]byte(jsonStr), &parsed); err != nil {
		t.Errorf("populated JSON invalid: %v\nJSON:\n%s", err, jsonStr)
	}
}

func checkNoEmptyStrings(t *testing.T, node *Node) {
	t.Helper()
	if node.Kind == KindString {
		s, _ := node.Value.(string)
		if s == "" {
			t.Error("found empty string value after population")
		}
	}
	for _, child := range node.Children {
		checkNoEmptyStrings(t, child)
	}
}

func TestGenerateDeterministic(t *testing.T) {
	// Same seed should produce same output.
	json1 := generateWithSeed(42)
	json2 := generateWithSeed(42)
	if json1 != json2 {
		t.Error("same seed produced different output")
	}

	// Different seeds should produce different output.
	json3 := generateWithSeed(43)
	if json1 == json3 {
		t.Error("different seeds produced same output")
	}
}

func generateWithSeed(seed uint64) string {
	rng := deterministicRNG(seed)
	cfg := DefaultConfig()
	gen := NewGenerator(cfg, rng)
	node := gen.Generate()
	pop := NewValuePopulator(rng, nil)
	pop.Populate(node)
	return Serialize(node)
}

func TestMinBreadthRespected(t *testing.T) {
	rng := deterministicRNG(42)
	cfg := DefaultConfig()
	cfg.MinBreadth = 3
	cfg.MaxBreadth = 3
	cfg.SizeBudget = 500
	gen := NewGenerator(cfg, rng)
	node := gen.Generate()

	// Root should have exactly 3 children (if budget allows).
	if len(node.Children) != 3 {
		t.Errorf("root has %d children, want 3", len(node.Children))
	}
}

func TestLargeSizeBudget(t *testing.T) {
	rng := deterministicRNG(42)
	cfg := DefaultConfig()
	cfg.SizeBudget = 1000
	cfg.MaxDepth = 8
	cfg.MaxBreadth = 15
	gen := NewGenerator(cfg, rng)
	node := gen.Generate()

	jsonStr := Serialize(node)
	var parsed any
	if err := json.Unmarshal([]byte(jsonStr), &parsed); err != nil {
		t.Errorf("large JSON invalid: %v", err)
	}

	count := countNodes(node)
	if count < 100 {
		t.Errorf("expected at least 100 nodes for budget 1000, got %d", count)
	}
}
