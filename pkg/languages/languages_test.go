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
	"math/rand/v2"
	"slices"
	"testing"
)

func deterministicRNG(seed uint64) *rand.Rand {
	return rand.New(rand.NewPCG(seed, seed+1))
}

func TestAllGeneratorsRegistered(t *testing.T) {
	all := All()
	if len(all) != 10 {
		t.Errorf("expected 10 generators, got %d", len(all))
	}
	names := make(map[string]bool)
	for _, g := range all {
		if names[g.Name()] {
			t.Errorf("duplicate generator name: %s", g.Name())
		}
		names[g.Name()] = true
	}
}

func TestEachGeneratorProducesNonEmptyOutput(t *testing.T) {
	for _, gen := range All() {
		t.Run(gen.Name(), func(t *testing.T) {
			for seed := range uint64(50) {
				rng := deterministicRNG(seed)
				output := gen.Generate(rng)
				if output == "" {
					t.Errorf("seed %d: generator %s produced empty output", seed, gen.Name())
				}
			}
		})
	}
}

func TestEachGeneratorProducesVariedOutput(t *testing.T) {
	for _, gen := range All() {
		t.Run(gen.Name(), func(t *testing.T) {
			outputs := make(map[string]bool)
			for seed := range uint64(20) {
				rng := deterministicRNG(seed)
				output := gen.Generate(rng)
				outputs[output] = true
			}
			// With 20 seeds we should get at least a few distinct outputs.
			if len(outputs) < 3 {
				t.Errorf("generator %s produced only %d distinct outputs across 20 seeds", gen.Name(), len(outputs))
			}
		})
	}
}

func TestEachGeneratorIsDeterministic(t *testing.T) {
	for _, gen := range All() {
		t.Run(gen.Name(), func(t *testing.T) {
			rng1 := deterministicRNG(42)
			rng2 := deterministicRNG(42)
			out1 := gen.Generate(rng1)
			out2 := gen.Generate(rng2)
			if out1 != out2 {
				t.Errorf("generator %s is not deterministic", gen.Name())
			}
		})
	}
}

func TestGeneratorsProduceReasonableLength(t *testing.T) {
	for _, gen := range All() {
		t.Run(gen.Name(), func(t *testing.T) {
			for seed := range uint64(20) {
				rng := deterministicRNG(seed)
				output := gen.Generate(rng)
				if len(output) < 10 {
					t.Errorf("seed %d: generator %s produced suspiciously short output (%d chars): %q",
						seed, gen.Name(), len(output), output)
				}
				if len(output) > 10000 {
					t.Errorf("seed %d: generator %s produced suspiciously long output (%d chars)",
						seed, gen.Name(), len(output))
				}
			}
		})
	}
}

func TestPickAndPickN(t *testing.T) {
	rng := deterministicRNG(42)
	items := []string{"a", "b", "c", "d", "e"}

	// pick should return an item from the list.
	for range 100 {
		p := pick(rng, items)
		if !slices.Contains(items, p) {
			t.Errorf("pick returned %q which is not in items", p)
		}
	}

	// pickN should return n unique items.
	result := pickN(rng, items, 3)
	if len(result) != 3 {
		t.Errorf("pickN(3) returned %d items", len(result))
	}
	seen := make(map[string]bool)
	for _, r := range result {
		if seen[r] {
			t.Errorf("pickN returned duplicate: %s", r)
		}
		seen[r] = true
	}

	// pickN with n >= len should return all.
	all := pickN(rng, items, 10)
	if len(all) != len(items) {
		t.Errorf("pickN(10) returned %d items, want %d", len(all), len(items))
	}
}

// TestSQLGeneratorCoversAllStatementTypes verifies SELECT, INSERT, UPDATE, DELETE, and JOIN queries.
func TestSQLGeneratorCoversAllStatementTypes(t *testing.T) {
	gen := SQL{}
	types := map[string]bool{"SELECT": false, "INSERT": false, "UPDATE": false, "DELETE": false, "JOIN": false}
	for seed := range uint64(200) {
		rng := deterministicRNG(seed)
		out := gen.Generate(rng)
		for keyword := range types {
			if contains(out, keyword) {
				types[keyword] = true
			}
		}
	}
	for keyword, found := range types {
		if !found {
			t.Errorf("SQL generator never produced a %s statement across 200 seeds", keyword)
		}
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && searchString(s, substr)
}

func searchString(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

// TestPythonGeneratorCoversAllVariants covers function, class, list comp, imports, f-string, async.
func TestPythonGeneratorCoversAllVariants(t *testing.T) {
	gen := Python{}
	patterns := map[string]bool{"def ": false, "class ": false, "[": false, "import ": false, "f\"": false, "async def": false}
	for seed := range uint64(200) {
		rng := deterministicRNG(seed)
		out := gen.Generate(rng)
		for p := range patterns {
			if contains(out, p) {
				patterns[p] = true
			}
		}
	}
	for p, found := range patterns {
		if !found {
			t.Errorf("Python generator never produced output matching %q across 200 seeds", p)
		}
	}
}

// TestIntegrationWithValuePopulator ensures language generators work as LanguageGenerators for jsongen.
func TestIntegrationWithValuePopulator(t *testing.T) {
	// Verify all generators satisfy the Generator interface.
	var generators []Generator
	generators = append(generators, All()...)
	if len(generators) != 10 {
		t.Fatalf("expected 10 generators, got %d", len(generators))
	}
	for _, g := range generators {
		rng := deterministicRNG(42)
		out := g.Generate(rng)
		if out == "" {
			t.Errorf("generator %s returned empty string", g.Name())
		}
	}
}
