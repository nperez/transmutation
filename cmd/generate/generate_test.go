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

package main

import (
	"encoding/json"
	"encoding/xml"
	"math/rand/v2"
	"strings"
	"testing"

	"nickandperla.net/transmutation/pkg/languages"
)

func TestGeneratePairProducesValidOutput(t *testing.T) {
	langGens := languages.AsJsongenGenerators()

	for seed := range uint64(200) {
		rng := rand.New(rand.NewPCG(seed, seed+1))
		pair := generateLegacyPair(rng, langGens)

		// Input should be non-empty.
		if pair.Input == "" {
			t.Errorf("seed %d: empty input", seed)
		}

		// Target should be valid XML.
		wrapped := "<root>" + pair.Target + "</root>"
		decoder := xml.NewDecoder(strings.NewReader(wrapped))
		for {
			_, err := decoder.Token()
			if err != nil {
				if err.Error() == "EOF" {
					break
				}
				t.Errorf("seed %d: invalid target XML: %v\nXML:\n%s", seed, err, pair.Target)
				break
			}
		}

		// The pair should be JSON-serializable.
		bs, err := json.Marshal(pair)
		if err != nil {
			t.Errorf("seed %d: failed to marshal pair: %v", seed, err)
			continue
		}

		// And round-trip back.
		var decoded TrainingPair
		if err := json.Unmarshal(bs, &decoded); err != nil {
			t.Errorf("seed %d: failed to unmarshal pair: %v", seed, err)
		}
		if decoded.Input != pair.Input || decoded.Target != pair.Target {
			t.Errorf("seed %d: round-trip mismatch", seed)
		}
	}
}

func TestGeneratePairWithAllCorruptionLevels(t *testing.T) {
	langGens := languages.AsJsongenGenerators()
	// Run enough seeds that all corruption levels get exercised.
	levels := map[string]bool{"none": false, "light": false, "medium": false, "heavy": false}

	for seed := range uint64(500) {
		rng := rand.New(rand.NewPCG(seed, seed+1))
		_ = randomConfig(rng) // consume the config RNG draws

		corrCfg := randomCorruptionConfig(rng)
		switch {
		case corrCfg.QuoteStripProb == 0 && corrCfg.CommaDropProb == 0:
			levels["none"] = true
		case corrCfg.QuoteStripProb <= 0.3:
			levels["light"] = true
		case corrCfg.QuoteStripProb <= 0.5:
			levels["medium"] = true
		default:
			levels["heavy"] = true
		}

		// Also verify the full pair generation works.
		rng2 := rand.New(rand.NewPCG(seed, seed+1))
		pair := generateLegacyPair(rng2, langGens)
		if pair.Target == "" {
			t.Errorf("seed %d: empty target", seed)
		}
	}

	for level, seen := range levels {
		if !seen {
			t.Errorf("corruption level %q never seen across 500 seeds", level)
		}
	}
}

func TestRandomConfigProducesValidConfigs(t *testing.T) {
	for seed := range uint64(100) {
		rng := rand.New(rand.NewPCG(seed, seed+1))
		cfg := randomConfig(rng)
		if cfg.MaxDepth < 2 || cfg.MaxDepth > 7 {
			t.Errorf("seed %d: MaxDepth %d out of range", seed, cfg.MaxDepth)
		}
		if cfg.MaxBreadth < 2 || cfg.MaxBreadth > 13 {
			t.Errorf("seed %d: MaxBreadth %d out of range", seed, cfg.MaxBreadth)
		}
		if cfg.SizeBudget < 5 {
			t.Errorf("seed %d: SizeBudget %d too small", seed, cfg.SizeBudget)
		}
	}
}

