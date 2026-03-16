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

// Command repair validates LLM-repaired JSON and generates training pairs.
//
// It reads JSONL from stdin with fields: {"original", "fix_a", "fix_b"}.
// For each record it checks agreement between the two fixes, validates the
// fix is a structural JSON repair (not a content rewrite), and outputs
// training pairs as JSONL: {"input": original, "target": xml}.
package main

import (
	"bufio"
	"encoding/json"
	"encoding/xml"
	"flag"
	"fmt"
	"os"
	"strings"

	"nickandperla.net/transmutation/pkg/xmlconv"
)

type RepairRecord struct {
	Original string `json:"original"`
	FixA     string `json:"fix_a"`
	FixB     string `json:"fix_b"`
}

type TrainingPair struct {
	Input  string `json:"input"`
	Target string `json:"target"`
}

func main() {
	maxDist := flag.Int("max-dist", 50, "maximum Levenshtein distance between fix_a and fix_b")
	maxChangePct := flag.Float64("max-change-pct", 10, "maximum percentage of characters changed from original (0-100)")
	flag.Parse()

	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 0, 1024*1024), 10*1024*1024)
	enc := json.NewEncoder(os.Stdout)

	total := 0
	accepted := 0
	rejected := 0
	reasons := map[string]int{}

	for scanner.Scan() {
		total++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		var rec RepairRecord
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			reasons["bad_record"]++
			rejected++
			continue
		}

		// Gate 1: Both fixes must be valid JSON.
		if !json.Valid([]byte(rec.FixA)) {
			reasons["fix_a_invalid_json"]++
			rejected++
			continue
		}
		if !json.Valid([]byte(rec.FixB)) {
			reasons["fix_b_invalid_json"]++
			rejected++
			continue
		}

		// Gate 2: Fixes must agree (low Levenshtein distance).
		dist := levenshtein(rec.FixA, rec.FixB)
		if dist > *maxDist {
			reasons["fixes_disagree"]++
			rejected++
			fmt.Fprintf(os.Stderr, "REJECT %d: fixes disagree (dist=%d > %d)\n", total, dist, *maxDist)
			continue
		}

		// Gate 3: Fix must be close to original (not a rewrite).
		changeDist := levenshtein(rec.Original, rec.FixA)
		changePct := float64(changeDist) / float64(max(len(rec.Original), 1)) * 100
		if changePct > *maxChangePct {
			reasons["too_many_changes"]++
			rejected++
			fmt.Fprintf(os.Stderr, "REJECT %d: too many changes (%.1f%% > %.1f%%)\n", total, changePct, *maxChangePct)
			continue
		}

		// Use fix_a as the canonical fix (they agree closely).
		fixedJSON := rec.FixA

		// Gate 4: Pretty-print for consistent formatting.
		var obj any
		if err := json.Unmarshal([]byte(fixedJSON), &obj); err != nil {
			reasons["unmarshal_fail"]++
			rejected++
			continue
		}
		pretty, err := json.MarshalIndent(obj, "", "  ")
		if err != nil {
			reasons["marshal_fail"]++
			rejected++
			continue
		}

		// Gate 5: Convert to XML.
		xmlOut, err := xmlconv.Convert(pretty)
		if err != nil {
			reasons["xmlconv_fail"]++
			rejected++
			fmt.Fprintf(os.Stderr, "REJECT %d: xmlconv: %v\n", total, err)
			continue
		}

		// Gate 6: Verify XML is parseable.
		dec := xml.NewDecoder(strings.NewReader("<root>" + xmlOut + "</root>"))
		xmlValid := true
		for {
			_, err := dec.Token()
			if err != nil {
				if err.Error() != "EOF" {
					xmlValid = false
				}
				break
			}
		}
		if !xmlValid {
			reasons["xml_invalid"]++
			rejected++
			continue
		}

		// Output training pair: original broken JSON → correct XML.
		pair := TrainingPair{Input: rec.Original, Target: xmlOut}
		enc.Encode(pair)
		accepted++
	}

	if err := scanner.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "error reading stdin: %v\n", err)
		os.Exit(1)
	}

	fmt.Fprintf(os.Stderr, "\nRepair results: %d accepted, %d rejected out of %d\n", accepted, rejected, total)
	for reason, count := range reasons {
		fmt.Fprintf(os.Stderr, "  %s: %d\n", reason, count)
	}
}

// levenshtein computes the Levenshtein distance between two strings.
func levenshtein(a, b string) int {
	if len(a) == 0 {
		return len(b)
	}
	if len(b) == 0 {
		return len(a)
	}

	// Use two rows instead of full matrix for memory efficiency.
	prev := make([]int, len(b)+1)
	curr := make([]int, len(b)+1)

	for j := range prev {
		prev[j] = j
	}

	for i := 1; i <= len(a); i++ {
		curr[0] = i
		for j := 1; j <= len(b); j++ {
			cost := 1
			if a[i-1] == b[j-1] {
				cost = 0
			}
			curr[j] = min(curr[j-1]+1, min(prev[j]+1, prev[j-1]+cost))
		}
		prev, curr = curr, prev
	}
	return prev[len(b)]
}
