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
)

var (
	preambles = []string{
		"Here is the JSON response:\n",
		"Here's the data you requested:\n\n",
		"Sure! Here's the JSON:\n\n",
		"The API returned the following JSON:\n",
		"Based on my analysis, here is the result:\n\n",
		"I've processed the request. Here's the output:\n\n",
		"Let me format that as JSON for you:\n\n",
		"Response:\n",
		"Output:\n\n",
		"```json\n",
		"The query results are:\n\n",
		"After executing the action, I got:\n\n",
		"Thought: I need to return the data in JSON format.\nAction: respond\nAction Input:\n",
		"I'll provide the structured data below:\n\n",
		"Here are the results in JSON format:\n\n",
	}

	postambles = []string{
		"\n\nLet me know if you need any changes.",
		"\n\nIs there anything else you'd like me to modify?",
		"\n\nI hope this helps! Let me know if you have questions.",
		"\n```",
		"\n\nNote: some fields may be null if data was not available.",
		"\n\nThis response includes all requested fields.",
		"\n\nPlease verify the data before proceeding.",
		"\n\nThe above JSON contains the complete response.",
		"\n\nObservation: The data has been formatted successfully.",
		"",
		"",
		"",
	}
)

// AddWrapper adds explanatory preamble and/or postamble text around the JSON.
func AddWrapper(json string, rng *rand.Rand) string {
	result := json

	// Add preamble.
	if rng.Float64() < 0.7 {
		result = preambles[rng.IntN(len(preambles))] + result
	}

	// Add postamble.
	if rng.Float64() < 0.5 {
		result = result + postambles[rng.IntN(len(postambles))]
	}

	return result
}
