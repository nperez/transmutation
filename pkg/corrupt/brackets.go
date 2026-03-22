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

// MangleBrackets corrupts closing brackets/braces to simulate real LLM failures.
// Mutation types (weighted by probability):
//   - 30% swap: } ↔ ] (most common real error)
//   - 20% drop: remove a closing bracket (truncation)
//   - 20% extra: duplicate a closing bracket
//   - 30% run: produce a run of 2-5 mismatched closers at a nesting boundary
//     (e.g. "]}}}" or "}}}}") — matches the most common real reject pattern
func MangleBrackets(json string, rng *rand.Rand) string {
	var b strings.Builder
	b.Grow(len(json) + 4)

	inString := false
	mutated := false

	// Decide mutation type.
	roll := rng.Float64()
	doSwap := roll < 0.3
	doDrop := roll >= 0.3 && roll < 0.5
	doExtra := roll >= 0.5 && roll < 0.7
	doRun := roll >= 0.7

	for i := 0; i < len(json); i++ {
		ch := json[i]

		if ch == '"' && !isEscaped(json, i) {
			inString = !inString
		}

		if !inString && !mutated && (ch == '}' || ch == ']') {
			if doSwap && rng.Float64() < 0.4 {
				mutated = true
				if ch == '}' {
					b.WriteByte(']')
				} else {
					b.WriteByte('}')
				}
				continue
			}
			if doDrop && rng.Float64() < 0.3 {
				mutated = true
				continue
			}
			if doExtra && rng.Float64() < 0.3 {
				mutated = true
				b.WriteByte(ch)
				b.WriteByte(ch)
				continue
			}
			// Run: at a nesting boundary (consecutive closers), replace with
			// a run of 2-5 mismatched brackets. Targets the end of nested
			// tool arguments where real LLMs produce "]}}}}".
			if doRun && rng.Float64() < 0.4 {
				// Check if we're at a run of consecutive closers.
				runEnd := i
				for runEnd+1 < len(json) {
					switch json[runEnd+1] {
					case '}', ']', '\n', ' ', '\t', '\r':
						runEnd++
					default:
						goto endRun
					}
				}
			endRun:
				origLen := 0
				for j := i; j <= runEnd; j++ {
					if json[j] == '}' || json[j] == ']' {
						origLen++
					}
				}
				// Only mangle runs of 2+ closers (nesting boundaries).
				if origLen >= 2 {
					mutated = true
					runLen := origLen + rng.IntN(3) // same length or up to 2 extra
					for k := range runLen {
						if rng.Float64() < 0.4 {
							b.WriteByte(']')
						} else {
							b.WriteByte('}')
						}
						_ = k
					}
					i = runEnd // skip past original run
					continue
				}
			}
		}

		b.WriteByte(ch)
	}
	return b.String()
}
