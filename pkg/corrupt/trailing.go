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

// AddTrailingCommas adds commas after the last element in objects and arrays.
func AddTrailingCommas(json string, rng *rand.Rand) string {
	var b strings.Builder
	b.Grow(len(json))

	inString := false
	for i := 0; i < len(json); i++ {
		ch := json[i]

		if ch == '"' && !isEscaped(json, i) {
			inString = !inString
		}

		// Look for patterns like `}` or `]` that close a container.
		// If the previous non-whitespace character isn't already a comma,
		// potentially insert one.
		if !inString && (ch == '}' || ch == ']') && rng.Float64() < 0.5 {
			// Walk backwards to find the last non-whitespace char.
			prev := lastNonWhitespace(b.String())
			if prev != ',' && prev != '{' && prev != '[' && prev != 0 {
				b.WriteByte(',')
			}
		}

		b.WriteByte(ch)
	}
	return b.String()
}

func lastNonWhitespace(s string) byte {
	for i := len(s) - 1; i >= 0; i-- {
		if s[i] != ' ' && s[i] != '\t' && s[i] != '\n' && s[i] != '\r' {
			return s[i]
		}
	}
	return 0
}
