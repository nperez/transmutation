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

// StripQuotes removes quotes from some JSON keys and identifier-like string values.
func StripQuotes(json string, rng *rand.Rand) string {
	var b strings.Builder
	b.Grow(len(json))

	i := 0
	for i < len(json) {
		if json[i] != '"' {
			b.WriteByte(json[i])
			i++
			continue
		}

		// Found a quote — find the matching close quote.
		end := findClosingQuote(json, i)
		if end == -1 {
			// Malformed — just write the rest.
			b.WriteString(json[i:])
			break
		}

		content := json[i+1 : end]

		// Decide whether to strip these quotes.
		// Strip with higher probability for keys (followed by ':')
		// and identifier-like values.
		shouldStrip := false
		if isIdentifierLike(content) && rng.Float64() < 0.6 {
			shouldStrip = true
		}

		if shouldStrip {
			b.WriteString(content)
		} else {
			b.WriteString(json[i : end+1])
		}
		i = end + 1
	}
	return b.String()
}

// findClosingQuote finds the index of the closing quote starting from an opening quote at pos.
func findClosingQuote(json string, pos int) int {
	for i := pos + 1; i < len(json); i++ {
		if json[i] == '\\' {
			i++ // skip escaped character
			continue
		}
		if json[i] == '"' {
			return i
		}
	}
	return -1
}

// isIdentifierLike returns true if a string looks like a JSON key or simple identifier
// (only letters, digits, underscores, no spaces or special chars).
func isIdentifierLike(s string) bool {
	if len(s) == 0 || len(s) > 50 {
		return false
	}
	for _, c := range s {
		if !((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_') {
			return false
		}
	}
	return true
}
