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
	"fmt"
	"math/rand/v2"
	"strings"

	"nickandperla.net/transmutation/pkg/corrupt"
	"nickandperla.net/transmutation/pkg/jsongen"
	"nickandperla.net/transmutation/pkg/languages"
	"nickandperla.net/transmutation/pkg/xmlconv"
)

func main() {
	adapted := languages.AsJsongenGenerators()

	configs := []struct {
		name     string
		cfg      jsongen.Config
		corr     corrupt.Config
		corrName string
	}{
		{"Small flat object", jsongen.Config{MaxDepth: 2, MaxBreadth: 4, MinBreadth: 3, SizeBudget: 15, ArrayProb: 0.1, ScalarDist: jsongen.DefaultConfig().ScalarDist}, corrupt.LightConfig(), "light"},
		{"Nested with arrays", jsongen.Config{MaxDepth: 4, MaxBreadth: 5, MinBreadth: 2, SizeBudget: 30, ArrayProb: 0.4, ScalarDist: jsongen.DefaultConfig().ScalarDist}, corrupt.MediumConfig(), "medium"},
		{"Deep structure", jsongen.Config{MaxDepth: 6, MaxBreadth: 3, MinBreadth: 2, SizeBudget: 40, ArrayProb: 0.3, ScalarDist: jsongen.DefaultConfig().ScalarDist}, corrupt.HeavyConfig(), "heavy"},
		{"Array root", jsongen.Config{MaxDepth: 3, MaxBreadth: 4, MinBreadth: 2, SizeBudget: 25, ArrayProb: 1.0, ScalarDist: jsongen.DefaultConfig().ScalarDist}, corrupt.MediumConfig(), "medium"},
		{"ReAct-like agent response", jsongen.Config{MaxDepth: 2, MaxBreadth: 5, MinBreadth: 4, SizeBudget: 20, ArrayProb: 0.1, ScalarDist: jsongen.ScalarDistribution{String: 0.8, Number: 0.1, Bool: 0.05, Null: 0.05}}, corrupt.HeavyConfig(), "heavy"},
	}

	for i, c := range configs {
		seed := uint64(i*7 + 42)
		rng := rand.New(rand.NewPCG(seed, seed+1))

		gen := jsongen.NewGenerator(c.cfg, rng)
		node := gen.Generate()

		pop := jsongen.NewValuePopulator(rng, adapted)
		pop.Populate(node)

		cleanJSON := jsongen.Serialize(node)
		corrupted := corrupt.Apply(cleanJSON, c.corr, rng)
		xmlOut, _ := xmlconv.Convert([]byte(cleanJSON))

		fmt.Println(strings.Repeat("=", 80))
		fmt.Printf("EXAMPLE %d: %s (corruption: %s)\n", i+1, c.name, c.corrName)
		fmt.Println(strings.Repeat("=", 80))
		fmt.Println()
		fmt.Println("--- CLEAN JSON ---")
		fmt.Println(cleanJSON)
		fmt.Println()
		fmt.Println("--- CORRUPTED INPUT ---")
		fmt.Println(corrupted)
		fmt.Println()
		fmt.Println("--- TARGET XML ---")
		fmt.Println(xmlOut)
		fmt.Println()
	}
}

