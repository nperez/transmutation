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

package corrupt

import (
	"encoding/json"
	"math/rand/v2"
	"strings"
	"testing"
)

func deterministicRNG(seed uint64) *rand.Rand {
	return rand.New(rand.NewPCG(seed, seed+1))
}

const sampleJSON = `{
  "thought": "I need to query the database",
  "action": "execute_sql",
  "action_input": "SELECT * FROM users WHERE name = 'alice'",
  "items": [1, 2, 3],
  "nested": {
    "key": "value",
    "enabled": true,
    "count": 42
  }
}`

func TestStripQuotesModifiesOutput(t *testing.T) {
	rng := deterministicRNG(42)
	result := StripQuotes(sampleJSON, rng)
	if result == sampleJSON {
		t.Error("StripQuotes did not modify the JSON")
	}
	// Should have fewer quotes than original.
	origQuotes := strings.Count(sampleJSON, `"`)
	resultQuotes := strings.Count(result, `"`)
	if resultQuotes >= origQuotes {
		t.Errorf("expected fewer quotes: orig=%d, result=%d", origQuotes, resultQuotes)
	}
}

func TestStripQuotesPreservesNonIdentifiers(t *testing.T) {
	// Strings with spaces/special chars should NOT be stripped.
	input := `{"message": "hello world", "url": "https://example.com"}`
	for seed := range uint64(50) {
		rng := deterministicRNG(seed)
		result := StripQuotes(input, rng)
		// "hello world" and URL should remain quoted (they're not identifier-like).
		if !strings.Contains(result, "hello world") {
			t.Errorf("seed %d: lost 'hello world' content", seed)
		}
	}
}

func TestDropCommasModifiesOutput(t *testing.T) {
	modified := false
	for seed := range uint64(20) {
		rng := deterministicRNG(seed)
		result := DropCommas(sampleJSON, rng)
		if result != sampleJSON {
			modified = true
			break
		}
	}
	if !modified {
		t.Error("DropCommas never modified the JSON across 20 seeds")
	}
}

func TestDropCommasDoesNotRemoveCommasInStrings(t *testing.T) {
	input := `{"data": "a, b, c", "items": [1, 2]}`
	for seed := range uint64(50) {
		rng := deterministicRNG(seed)
		result := DropCommas(input, rng)
		// The commas inside "a, b, c" should never be touched.
		if !strings.Contains(result, "a, b, c") {
			t.Errorf("seed %d: commas inside string were modified: %s", seed, result)
		}
	}
}

func TestDropColonsModifiesOutput(t *testing.T) {
	modified := false
	for seed := range uint64(20) {
		rng := deterministicRNG(seed)
		result := DropColons(sampleJSON, rng)
		if result != sampleJSON {
			modified = true
			break
		}
	}
	if !modified {
		t.Error("DropColons never modified the JSON across 20 seeds")
	}
}

func TestInsertCommentsAddsComments(t *testing.T) {
	hasLineComment := false
	hasBlockComment := false
	for seed := range uint64(50) {
		rng := deterministicRNG(seed)
		result := InsertComments(sampleJSON, rng)
		if strings.Contains(result, "//") {
			hasLineComment = true
		}
		if strings.Contains(result, "/*") {
			hasBlockComment = true
		}
	}
	if !hasLineComment {
		t.Error("InsertComments never added a line comment across 50 seeds")
	}
	if !hasBlockComment {
		t.Error("InsertComments never added a block comment across 50 seeds")
	}
}

func TestAddWrapperAddsPreamble(t *testing.T) {
	hasPreamble := false
	hasPostamble := false
	for seed := range uint64(50) {
		rng := deterministicRNG(seed)
		result := AddWrapper(sampleJSON, rng)
		if !strings.HasPrefix(result, "{") {
			hasPreamble = true
		}
		if !strings.HasSuffix(result, "}") && !strings.HasSuffix(result, "}\n") {
			hasPostamble = true
		}
	}
	if !hasPreamble {
		t.Error("AddWrapper never added a preamble across 50 seeds")
	}
	if !hasPostamble {
		t.Error("AddWrapper never added a postamble across 50 seeds")
	}
}

func TestAddTrailingCommasModifiesOutput(t *testing.T) {
	modified := false
	for seed := range uint64(20) {
		rng := deterministicRNG(seed)
		result := AddTrailingCommas(sampleJSON, rng)
		if result != sampleJSON {
			modified = true
			// The result should have more commas.
			if strings.Count(result, ",") <= strings.Count(sampleJSON, ",") {
				t.Errorf("seed %d: trailing commas didn't add commas", seed)
			}
			break
		}
	}
	if !modified {
		t.Error("AddTrailingCommas never modified the JSON across 20 seeds")
	}
}

func TestMangleBracketsProducesMutation(t *testing.T) {
	mutated := false
	for seed := range uint64(100) {
		rng := deterministicRNG(seed)
		result := MangleBrackets(sampleJSON, rng)
		if result != sampleJSON {
			mutated = true
			// Result should still contain at least some brackets.
			closers := strings.Count(result, "}") + strings.Count(result, "]")
			if closers == 0 {
				t.Errorf("seed %d: all brackets removed", seed)
			}
			break
		}
	}
	if !mutated {
		t.Error("MangleBrackets never mutated across 100 seeds")
	}
}

func TestMangleBracketsCanDropFinalBracket(t *testing.T) {
	input := `{"key": "value"}`
	droppedFinal := false
	for seed := range uint64(500) {
		rng := deterministicRNG(seed)
		result := MangleBrackets(input, rng)
		if !strings.Contains(result, "}") {
			droppedFinal = true
			break
		}
	}
	if !droppedFinal {
		t.Error("MangleBrackets never dropped the final bracket across 500 seeds")
	}
}

func TestMangleWhitespaceModifiesOutput(t *testing.T) {
	modified := false
	for seed := range uint64(20) {
		rng := deterministicRNG(seed)
		result := MangleWhitespace(sampleJSON, rng)
		if result != sampleJSON {
			modified = true
			break
		}
	}
	if !modified {
		t.Error("MangleWhitespace never modified the JSON across 20 seeds")
	}
}

func TestApplyWithNoneConfigPassesThrough(t *testing.T) {
	rng := deterministicRNG(42)
	result := Apply(sampleJSON, NoneConfig(), rng)
	if result != sampleJSON {
		t.Error("NoneConfig should produce no corruption")
	}
}

func TestApplyProducesCorruptedOutput(t *testing.T) {
	for _, cfg := range []struct {
		name string
		cfg  Config
	}{
		{"light", LightConfig()},
		{"medium", MediumConfig()},
		{"heavy", HeavyConfig()},
	} {
		t.Run(cfg.name, func(t *testing.T) {
			modified := false
			for seed := range uint64(20) {
				rng := deterministicRNG(seed)
				result := Apply(sampleJSON, cfg.cfg, rng)
				if result != sampleJSON {
					modified = true
					break
				}
			}
			if !modified {
				t.Errorf("%s config never modified the JSON", cfg.name)
			}
		})
	}
}

func TestApplyIsDeterministic(t *testing.T) {
	cfg := MediumConfig()
	rng1 := deterministicRNG(42)
	rng2 := deterministicRNG(42)
	result1 := Apply(sampleJSON, cfg, rng1)
	result2 := Apply(sampleJSON, cfg, rng2)
	if result1 != result2 {
		t.Error("Apply is not deterministic with same seed")
	}
}

func TestCorruptionPreservesOriginalData(t *testing.T) {
	// After corruption, the key data should still be findable in the output.
	cfg := MediumConfig()
	cfg.WrapperProb = 0 // Exclude wrapper to simplify check.
	cfg.BracketProb = 0

	for seed := range uint64(50) {
		rng := deterministicRNG(seed)
		result := Apply(sampleJSON, cfg, rng)
		// The SQL query value should survive corruption (it's in a string).
		if !strings.Contains(result, "SELECT") {
			t.Errorf("seed %d: lost 'SELECT' content after corruption", seed)
		}
	}
}

func TestIsEscaped(t *testing.T) {
	tests := []struct {
		input string
		pos   int
		want  bool
	}{
		{`"hello"`, 6, false},
		{`"he\"llo"`, 4, true},
		{`"he\\"`, 5, false},     // \\ is escaped backslash, closing quote at pos 5 is not escaped
		{`"he\\\"llo"`, 6, true}, // \\\" — the quote at pos 6 is escaped (odd backslashes)
	}
	for _, tt := range tests {
		got := isEscaped(tt.input, tt.pos)
		if got != tt.want {
			t.Errorf("isEscaped(%q, %d) = %v, want %v", tt.input, tt.pos, got, tt.want)
		}
	}
}

func TestStripQuotesHandlesEmptyInput(t *testing.T) {
	rng := deterministicRNG(42)
	result := StripQuotes("", rng)
	if result != "" {
		t.Errorf("expected empty string, got %q", result)
	}
}

func TestCorruptionStatisticalDistribution(t *testing.T) {
	// Verify each corruption type fires at roughly expected frequency.
	quotesModified := 0
	commasModified := 0
	commentsAdded := 0

	for seed := range uint64(1000) {
		rng1 := deterministicRNG(seed)
		if StripQuotes(sampleJSON, rng1) != sampleJSON {
			quotesModified++
		}
		rng2 := deterministicRNG(seed + 10000)
		if DropCommas(sampleJSON, rng2) != sampleJSON {
			commasModified++
		}
		rng3 := deterministicRNG(seed + 20000)
		if InsertComments(sampleJSON, rng3) != sampleJSON {
			commentsAdded++
		}
	}

	// With the medium config probabilities and internal randomness,
	// each should fire at least occasionally.
	if quotesModified < 100 {
		t.Errorf("quotes modified only %d/1000 times, expected more", quotesModified)
	}
	if commasModified < 100 {
		t.Errorf("commas modified only %d/1000 times, expected more", commasModified)
	}
	if commentsAdded < 100 {
		t.Errorf("comments added only %d/1000 times, expected more", commentsAdded)
	}
}

func TestFullPipelineRoundTrip(t *testing.T) {
	// Generate clean JSON, corrupt it, verify the original data is still
	// represented in the corrupted version (at least the values survive).
	clean := `{"name": "Alice", "age": 30, "active": true, "tags": ["admin", "user"]}`
	var original map[string]any
	if err := json.Unmarshal([]byte(clean), &original); err != nil {
		t.Fatalf("bad test data: %v", err)
	}

	cfg := HeavyConfig()
	cfg.BracketProb = 0 // Don't break structure for this test.

	for seed := range uint64(50) {
		rng := deterministicRNG(seed)
		corrupted := Apply(clean, cfg, rng)
		// At minimum, the value "Alice" should survive.
		if !strings.Contains(corrupted, "Alice") {
			t.Errorf("seed %d: lost 'Alice' in corrupted output: %s", seed, corrupted)
		}
	}
}
