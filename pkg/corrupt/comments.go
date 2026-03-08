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
	"fmt"
	"math/rand/v2"
	"strings"
)

var (
	lineComments = []string{
		"// this is the main object",
		"// user configuration",
		"// TODO: validate this field",
		"// required field",
		"// optional",
		"// deprecated, use new_field instead",
		"// see docs for valid values",
		"// auto-generated",
		"// timestamps are in UTC",
		"// max length: 255",
	}

	blockComments = []string{
		"/* primary key */",
		"/* response payload */",
		"/* configuration section */",
		"/* this should be an array */",
		"/* see API docs for details */",
		"/* temporary workaround */",
	}
)

// InsertComments adds // and /* */ comments at random positions in the JSON.
func InsertComments(json string, rng *rand.Rand) string {
	lines := strings.Split(json, "\n")
	result := make([]string, 0, len(lines)*2)

	for _, line := range lines {
		result = append(result, line)

		if rng.Float64() < 0.25 {
			if rng.Float64() < 0.7 {
				// Line comment at end of this line or on its own line.
				if rng.Float64() < 0.5 {
					// Append to current line.
					result[len(result)-1] = line + " " + lineComments[rng.IntN(len(lineComments))]
				} else {
					// Insert as separate line.
					indent := leadingWhitespace(line)
					result = append(result, indent+lineComments[rng.IntN(len(lineComments))])
				}
			} else {
				// Block comment.
				indent := leadingWhitespace(line)
				result = append(result, indent+blockComments[rng.IntN(len(blockComments))])
			}
		}
	}

	// Occasionally add a multi-line block comment.
	if rng.Float64() < 0.3 && len(result) > 2 {
		pos := 1 + rng.IntN(len(result)-1)
		comment := fmt.Sprintf("/*\n * %s\n * %s\n */",
			pick(rng, []string{"Configuration object for the API client",
				"This structure represents the response payload",
				"User settings and preferences",
				"Database query parameters"}),
			pick(rng, []string{"All fields are required unless noted",
				"See documentation for valid values",
				"Modified by the preprocessing step",
				"Validated on the server side"}))
		// Insert at pos.
		result = append(result[:pos+1], result[pos:]...)
		result[pos] = comment
	}

	return strings.Join(result, "\n")
}

func leadingWhitespace(s string) string {
	for i, c := range s {
		if c != ' ' && c != '\t' {
			return s[:i]
		}
	}
	return s
}

func pick(rng *rand.Rand, items []string) string {
	return items[rng.IntN(len(items))]
}
