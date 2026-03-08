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

// Package languages provides generators for syntactically plausible code
// snippets across multiple programming languages. These are used to populate
// string values in synthetic JSON training data.
package languages

import "math/rand/v2"

// Generator produces random code snippets for a given language.
type Generator interface {
	Name() string
	Generate(rng *rand.Rand) string
}

// All returns all available language generators.
func All() []Generator {
	return []Generator{
		&SQL{},
		&Python{},
		&JavaScript{},
		&Shell{},
		&HTML{},
		&Markdown{},
		&CSS{},
		&YAML{},
		&Go{},
		&JSON{},
	}
}

func pick(rng *rand.Rand, items []string) string {
	return items[rng.IntN(len(items))]
}

func pickN(rng *rand.Rand, items []string, n int) []string {
	if n >= len(items) {
		return items
	}
	// Fisher-Yates on a copy, take first n.
	cp := make([]string, len(items))
	copy(cp, items)
	for i := len(cp) - 1; i > 0; i-- {
		j := rng.IntN(i + 1)
		cp[i], cp[j] = cp[j], cp[i]
	}
	return cp[:n]
}
