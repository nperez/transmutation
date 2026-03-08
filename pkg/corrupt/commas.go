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

// DropCommas removes some commas that separate JSON elements.
func DropCommas(json string, rng *rand.Rand) string {
	var b strings.Builder
	b.Grow(len(json))

	inString := false
	for i := 0; i < len(json); i++ {
		ch := json[i]

		if ch == '"' && !isEscaped(json, i) {
			inString = !inString
		}

		if ch == ',' && !inString && rng.Float64() < 0.4 {
			// Drop this comma — optionally replace with whitespace so
			// tokens don't merge.
			if rng.Float64() < 0.5 {
				b.WriteByte(' ')
			}
			continue
		}

		b.WriteByte(ch)
	}
	return b.String()
}

// isEscaped checks if the character at pos is preceded by an odd number of backslashes.
func isEscaped(s string, pos int) bool {
	count := 0
	for i := pos - 1; i >= 0 && s[i] == '\\'; i-- {
		count++
	}
	return count%2 == 1
}
