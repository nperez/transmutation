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

// MangleWhitespace introduces inconsistent indentation and spacing.
func MangleWhitespace(json string, rng *rand.Rand) string {
	lines := strings.Split(json, "\n")
	result := make([]string, 0, len(lines))

	for _, line := range lines {
		trimmed := strings.TrimLeft(line, " \t")
		if trimmed == "" {
			// Randomly keep or drop blank lines.
			if rng.Float64() < 0.5 {
				result = append(result, "")
			}
			continue
		}

		// Randomize indentation.
		switch rng.IntN(5) {
		case 0:
			// No indent.
			result = append(result, trimmed)
		case 1:
			// Tabs instead of spaces.
			depth := rng.IntN(4)
			result = append(result, strings.Repeat("\t", depth)+trimmed)
		case 2:
			// Inconsistent spaces.
			depth := rng.IntN(8)
			result = append(result, strings.Repeat(" ", depth)+trimmed)
		case 3:
			// Mixed tabs and spaces.
			result = append(result, "\t "+strings.Repeat(" ", rng.IntN(3))+trimmed)
		default:
			// Keep original.
			result = append(result, line)
		}

		// Randomly insert extra blank lines.
		if rng.Float64() < 0.1 {
			result = append(result, "")
		}
	}

	return strings.Join(result, "\n")
}
