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
	"encoding/json"
	"math/rand/v2"
)

// DropKeys removes 1-2 random top-level keys from a JSON object string.
// Returns the reduced JSON (still valid) and the list of surviving keys.
// This simulates LLM truncation where entire fields are missing.
// The caller is responsible for regenerating the target from the reduced JSON.
func DropKeys(jsonStr string, rng *rand.Rand) (string, error) {
	var obj map[string]json.RawMessage
	if err := json.Unmarshal([]byte(jsonStr), &obj); err != nil {
		return jsonStr, err
	}

	keys := make([]string, 0, len(obj))
	for k := range obj {
		keys = append(keys, k)
	}
	if len(keys) <= 1 {
		return jsonStr, nil // nothing to drop
	}

	// Shuffle and drop 1-2 keys.
	rng.Shuffle(len(keys), func(i, j int) {
		keys[i], keys[j] = keys[j], keys[i]
	})
	nDrop := 1
	if len(keys) > 2 && rng.Float64() < 0.3 {
		nDrop = 2
	}
	for i := 0; i < nDrop && i < len(keys); i++ {
		delete(obj, keys[i])
	}

	out, err := json.Marshal(obj)
	if err != nil {
		return jsonStr, err
	}
	return string(out), nil
}

// TruncateJSON cuts a JSON string at a random point between 50-90% through,
// simulating LLM output that was cut off mid-generation.
// The result is intentionally invalid JSON (no closing braces).
func TruncateJSON(jsonStr string, rng *rand.Rand) string {
	n := len(jsonStr)
	if n < 20 {
		return jsonStr
	}

	// Pick a cut point between 50-90% through the string.
	minCut := n / 2
	maxCut := n * 9 / 10
	if maxCut <= minCut {
		maxCut = minCut + 1
	}
	cut := minCut + rng.IntN(maxCut-minCut)

	// Walk backwards from cut to find a clean break point:
	// after a comma, closing bracket, or end of string value.
	for i := cut; i > minCut; i-- {
		ch := jsonStr[i]
		if ch == ',' || ch == ']' || ch == '}' {
			return jsonStr[:i+1]
		}
		// After a closing quote (end of string value).
		if ch == '"' && i > 0 && jsonStr[i-1] != '\\' {
			return jsonStr[:i+1]
		}
	}

	// Fallback: just cut at the computed point.
	return jsonStr[:cut]
}
