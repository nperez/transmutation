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
	"math/rand/v2"
	"strings"
)

// MangleBrackets occasionally drops a closing bracket or brace.
// This is applied with very low probability since it creates the most
// severe corruption.
func MangleBrackets(json string, rng *rand.Rand) string {
	var b strings.Builder
	b.Grow(len(json))

	inString := false
	// Only drop one bracket per application to keep corruption recoverable.
	dropped := false

	for i := 0; i < len(json); i++ {
		ch := json[i]

		if ch == '"' && !isEscaped(json, i) {
			inString = !inString
		}

		// Only consider dropping closing brackets, never opening ones.
		// This simulates LLMs that stop generating mid-structure or
		// truncate output before the final bracket.
		if !inString && !dropped && (ch == '}' || ch == ']') {
			remaining := json[i+1:]
			isFinal := !strings.Contains(remaining, "}") && !strings.Contains(remaining, "]")
			if isFinal {
				// Final bracket — drop with lower probability (simulates truncation).
				if rng.Float64() < 0.2 {
					dropped = true
					continue
				}
			} else if rng.Float64() < 0.3 {
				dropped = true
				continue
			}
		}

		b.WriteByte(ch)
	}
	return b.String()
}
