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

package agent

import (
	"encoding/json"
	"encoding/xml"
	"math/rand/v2"
	"strings"
	"testing"
)

func TestGenerateAllStages(t *testing.T) {
	for stage := StageSimple; stage <= StageWrapper; stage++ {
		for seed := range uint64(50) {
			rng := rand.New(rand.NewPCG(seed, seed+1))
			gen := NewGenerator(rng, stage)
			cleanJSON, xmlOut := gen.Generate()

			// JSON must be valid.
			var raw any
			if err := json.Unmarshal([]byte(cleanJSON), &raw); err != nil {
				t.Errorf("stage %d seed %d: invalid JSON: %v\nJSON:\n%s", stage, seed, err, cleanJSON)
				continue
			}

			// XML must be valid.
			wrapped := "<root>" + xmlOut + "</root>"
			decoder := xml.NewDecoder(strings.NewReader(wrapped))
			for {
				_, err := decoder.Token()
				if err != nil {
					if err.Error() == "EOF" {
						break
					}
					t.Errorf("stage %d seed %d: invalid XML: %v\nXML:\n%s", stage, seed, err, xmlOut)
					break
				}
			}

			// Agent schema: must have thought and memory.
			obj, ok := raw.(map[string]any)
			if !ok {
				t.Errorf("stage %d seed %d: top-level is not an object", stage, seed)
				continue
			}
			if _, ok := obj["thought"]; !ok {
				t.Errorf("stage %d seed %d: missing 'thought' field", stage, seed)
			}
			if _, ok := obj["memory"]; !ok {
				t.Errorf("stage %d seed %d: missing 'memory' field", stage, seed)
			}

			// Exactly one of answer/tool should be non-null.
			answerNull := obj["answer"] == nil
			toolNull := obj["tool"] == nil
			if answerNull == toolNull {
				t.Errorf("stage %d seed %d: expected exactly one of answer/tool to be non-null (answer=%v, tool=%v)",
					stage, seed, obj["answer"] != nil, obj["tool"] != nil)
			}
		}
	}
}

func TestStageContentDistribution(t *testing.T) {
	// Stage 1 should only produce answers (no tools).
	for seed := range uint64(100) {
		rng := rand.New(rand.NewPCG(seed, seed+1))
		gen := NewGenerator(rng, StageSimple)
		cleanJSON, _ := gen.Generate()

		var obj map[string]any
		json.Unmarshal([]byte(cleanJSON), &obj)

		if obj["tool"] != nil {
			t.Errorf("stage 1 seed %d: produced a tool call (should only produce answers)", seed)
		}
	}

	// Stage 2+ should produce some tool calls.
	toolCount := 0
	for seed := range uint64(200) {
		rng := rand.New(rand.NewPCG(seed, seed+1))
		gen := NewGenerator(rng, StageTools)
		cleanJSON, _ := gen.Generate()

		var obj map[string]any
		json.Unmarshal([]byte(cleanJSON), &obj)

		if obj["tool"] != nil {
			toolCount++
		}
	}
	if toolCount == 0 {
		t.Error("stage 2: no tool calls produced across 200 seeds")
	}
	if toolCount < 50 {
		t.Errorf("stage 2: only %d tool calls across 200 seeds (expected ~120)", toolCount)
	}
}
