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

package main

import (
	"bufio"
	"encoding/json"
	"encoding/xml"
	"flag"
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"

	"nickandperla.net/transmutation/pkg/corrupt"
	"nickandperla.net/transmutation/pkg/randtext"
	"nickandperla.net/transmutation/pkg/xmlconv"
)

type TrainingPair struct {
	Input  string `json:"input"`
	Target string `json:"target"`
}

var specialProb float64
var corruptPct float64
var compactPct float64

func main() {
	var (
		haikuDir   string
		samplePct  float64
		augRatio   int
		seed       uint64
		isVal      bool
		sampleType string
	)

	flag.StringVar(&haikuDir, "dir", "data/haiku", "haiku JSONL directory")
	flag.Float64Var(&samplePct, "sample-pct", 5, "percentage of corpus to sample (0-100)")
	flag.IntVar(&augRatio, "aug-ratio", 10, "augmented variants per natural sample")
	flag.Uint64Var(&seed, "seed", 42, "random seed")
	flag.BoolVar(&isVal, "val", false, "generate val split (uses offset seed, disjoint from train)")
	flag.Float64Var(&specialProb, "special-prob", 0.15, "probability of XML special char injection per word boundary (0-1)")
	flag.Float64Var(&corruptPct, "corrupt-pct", 0, "percentage of samples to corrupt input JSON (0-100)")
	flag.Float64Var(&compactPct, "compact-pct", 0, "percentage of samples to compact input to single-line JSON (0-100)")
	var minChars int
	flag.IntVar(&minChars, "min-chars", 0, "minimum input character length (0 = no filter)")
	flag.StringVar(&sampleType, "type", "all", "sample type filter: answer, tool, or all")
	flag.Parse()

	samples, err := loadHaiku(haikuDir, sampleType)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error loading haiku: %v\n", err)
		os.Exit(1)
	}
	if minChars > 0 {
		filtered := samples[:0]
		for _, s := range samples {
			if len(s.Input) >= minChars {
				filtered = append(filtered, s)
			}
		}
		fmt.Fprintf(os.Stderr, "Loaded %d haiku samples, %d passed min-chars=%d filter\n", len(samples), len(filtered), minChars)
		samples = filtered
	} else {
		fmt.Fprintf(os.Stderr, "Loaded %d haiku samples\n", len(samples))
	}

	// Deterministic shuffle of indices — same seed always gives same order.
	// Val uses offset seed so train and val sample disjoint subsets.
	sampSeed := seed
	if isVal {
		sampSeed = seed + 7777777
	}
	rng := rand.New(rand.NewPCG(sampSeed, sampSeed^0xdeadbeef))

	keep := int(float64(len(samples)) * samplePct / 100)
	if keep <= 0 {
		fmt.Fprintf(os.Stderr, "error: sample-pct=%.1f would keep 0 of %d samples\n", samplePct, len(samples))
		os.Exit(1)
	}
	if keep > len(samples) {
		keep = len(samples)
	}

	selected := stratifiedSample(samples, keep, rng)

	bw := bufio.NewWriterSize(os.Stdout, 256*1024)
	defer bw.Flush()
	enc := json.NewEncoder(bw)

	natural := 0
	augmented := 0
	augFailed := 0
	corrupted := 0
	compacted := 0

	for _, idx := range selected {
		sample := samples[idx]

		// Emit the natural (unmodified) sample, optionally compacted/corrupted.
		out := sample
		cSeed := seed + uint64(idx)*uint64(augRatio+1) + 999999
		cRng := rand.New(rand.NewPCG(cSeed, cSeed^0xf00d))
		if compactPct > 0 && cRng.Float64()*100 < compactPct {
			out.Input = compactJSON(out.Input)
			compacted++
		}
		if corruptPct > 0 && cRng.Float64()*100 < corruptPct {
			out.Input = corrupt.Apply(out.Input, corruptionConfig(cRng), cRng)
			corrupted++
		}
		enc.Encode(out)
		natural++

		// Emit augRatio augmented variants, optionally compacted/corrupted.
		for v := range augRatio {
			augSeed := seed + uint64(idx)*uint64(augRatio+1) + uint64(v) + 1
			augRng := rand.New(rand.NewPCG(augSeed, augSeed^0xcafebabe))

			aug, err := augmentSample(sample, augRng)
			if err != nil {
				fmt.Fprintf(os.Stderr, "  augment fail: %v\n", err)
				augFailed++
				continue
			}
			if compactPct > 0 && augRng.Float64()*100 < compactPct {
				aug.Input = compactJSON(aug.Input)
				compacted++
			}
			if corruptPct > 0 {
				cSeed := augSeed ^ 0xf00d
				cRng := rand.New(rand.NewPCG(cSeed, cSeed^0xbeef))
				if cRng.Float64()*100 < corruptPct {
					aug.Input = corrupt.Apply(aug.Input, corruptionConfig(cRng), cRng)
					corrupted++
				}
			}
			enc.Encode(aug)
			augmented++
		}
	}

	bw.Flush()

	split := "train"
	if isVal {
		split = "val"
	}
	fmt.Fprintf(os.Stderr, "Haiku augment (%s): %d natural + %d augmented = %d total (sampled %d = %.1f%% of %d",
		split, natural, augmented, natural+augmented, keep, samplePct, len(samples))
	if compacted > 0 {
		fmt.Fprintf(os.Stderr, ", %d compacted", compacted)
	}
	if corrupted > 0 {
		fmt.Fprintf(os.Stderr, ", %d corrupted", corrupted)
	}
	if augFailed > 0 {
		fmt.Fprintf(os.Stderr, ", %d augment failures", augFailed)
	}
	fmt.Fprintf(os.Stderr, ")\n")
}

// isToolSample checks if a haiku input JSON contains a non-null tool field.
func isToolSample(input string) bool {
	// Parse the input JSON to check the tool field.
	var obj map[string]json.RawMessage
	if err := json.Unmarshal([]byte(input), &obj); err != nil {
		return false
	}
	raw, ok := obj["tool"]
	if !ok {
		return false
	}
	return string(raw) != "null"
}

func loadHaiku(dir string, sampleType string) ([]TrainingPair, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("readdir %s: %w", dir, err)
	}

	var all []TrainingPair
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".jsonl") {
			continue
		}
		f, err := os.Open(filepath.Join(dir, e.Name()))
		if err != nil {
			fmt.Fprintf(os.Stderr, "warn: skip %s: %v\n", e.Name(), err)
			continue
		}
		scanner := bufio.NewScanner(f)
		scanner.Buffer(make([]byte, 0, 512*1024), 10*1024*1024)
		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line == "" {
				continue
			}
			var pair TrainingPair
			if err := json.Unmarshal([]byte(line), &pair); err != nil {
				continue
			}
			if pair.Input == "" || pair.Target == "" {
				continue
			}
			// Filter by type.
			if sampleType != "all" {
				isTool := isToolSample(pair.Input)
				if sampleType == "answer" && isTool {
					continue
				}
				if sampleType == "tool" && !isTool {
					continue
				}
			}
			all = append(all, pair)
		}
		f.Close()
	}

	if len(all) == 0 {
		return nil, fmt.Errorf("no valid %s samples found in %s", sampleType, dir)
	}
	return all, nil
}

// augmentSample takes a natural haiku sample, replaces all string values
// with augmented content (dict words or shuffled + special char injection),
// then regenerates XML from the modified JSON.
func augmentSample(sample TrainingPair, rng *rand.Rand) (TrainingPair, error) {
	var obj any
	if err := json.Unmarshal([]byte(sample.Input), &obj); err != nil {
		return TrainingPair{}, fmt.Errorf("parse input: %w", err)
	}

	augmentValues(obj, rng)

	pretty, err := json.MarshalIndent(obj, "", "  ")
	if err != nil {
		return TrainingPair{}, fmt.Errorf("marshal: %w", err)
	}

	xmlOut, err := xmlconv.Convert(pretty)
	if err != nil {
		return TrainingPair{}, fmt.Errorf("xmlconv: %w", err)
	}

	// Verify the XML is parseable.
	dec := xml.NewDecoder(strings.NewReader("<root>" + xmlOut + "</root>"))
	for {
		_, err := dec.Token()
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			return TrainingPair{}, fmt.Errorf("invalid xml: %w", err)
		}
	}

	return TrainingPair{Input: string(pretty), Target: xmlOut}, nil
}

// augmentValues recursively walks a parsed JSON value and replaces all
// string values with augmented content. Keys are preserved.
func augmentValues(v any, rng *rand.Rand) {
	switch val := v.(type) {
	case map[string]any:
		for k, child := range val {
			if s, ok := child.(string); ok {
				val[k] = augmentString(s, rng)
			} else {
				augmentValues(child, rng)
			}
		}
	case []any:
		for i, child := range val {
			if s, ok := child.(string); ok {
				val[i] = augmentString(s, rng)
			} else {
				augmentValues(child, rng)
			}
		}
	}
}

// augmentString replaces a string value with either dictionary words or
// shuffled words from the original, both with XML special char injection
// at the configured specialProb rate.
func augmentString(s string, rng *rand.Rand) string {
	words := strings.Fields(s)
	n := len(words)
	if n == 0 {
		n = 1 + rng.IntN(3)
	}

	if rng.Float64() < 0.5 {
		// Dictionary words.
		newWords := make([]string, n)
		for i := range newWords {
			newWords[i] = randtext.DictWord(rng)
		}
		return randtext.InjectSpecialChars(rng, strings.Join(newWords, " "), specialProb)
	}

	// Shuffle original words + inject special chars.
	rng.Shuffle(len(words), func(i, j int) {
		words[i], words[j] = words[j], words[i]
	})
	return randtext.InjectSpecialChars(rng, strings.Join(words, " "), specialProb)
}

// Length bins for stratified sampling (input character counts).
// Tuned to the haiku corpus distribution where pretty-printed JSON
// starts around 2000 chars. Bins cover short→long evenly.
var lengthBins = []int{0, 2500, 3500, 4500, 6000, 8000, 11000, 16000}

// stratifiedSample bins samples by input character length and samples equally
// from each bin. Bins with fewer samples than their quota contribute all they
// have; surplus quota is redistributed to larger bins.
func stratifiedSample(samples []TrainingPair, total int, rng *rand.Rand) []int {
	numBins := len(lengthBins)
	bins := make([][]int, numBins)
	for i := range bins {
		bins[i] = []int{}
	}

	// Assign each sample to a bin.
	for idx, s := range samples {
		charLen := len(s.Input)
		bin := numBins - 1
		for b := numBins - 1; b >= 0; b-- {
			if charLen >= lengthBins[b] {
				bin = b
				break
			}
		}
		bins[bin] = append(bins[bin], idx)
	}

	// Shuffle each bin.
	for _, bin := range bins {
		rng.Shuffle(len(bin), func(i, j int) {
			bin[i], bin[j] = bin[j], bin[i]
		})
	}

	// Sample equally, redistribute surplus from small bins.
	perBin := total / numBins
	var selected []int
	surplus := 0
	for _, bin := range bins {
		quota := perBin
		if len(bin) < quota {
			surplus += quota - len(bin)
			quota = len(bin)
		}
		selected = append(selected, bin[:quota]...)
	}

	// Distribute surplus across bins that have remaining capacity.
	if surplus > 0 {
		for i, bin := range bins {
			used := perBin
			if len(bin) < perBin {
				continue // already exhausted
			}
			remaining := len(bin) - used
			take := min(remaining, surplus)
			if take > 0 {
				selected = append(selected, bin[used:used+take]...)
				surplus -= take
				_ = i
			}
			if surplus <= 0 {
				break
			}
		}
	}

	return selected
}

// compactJSON re-marshals pretty-printed JSON into single-line compact form.
// This simulates real LLM output which is typically one-line JSON with escaped
// newlines (\n) inside string values rather than pretty-printed.
func compactJSON(prettyJSON string) string {
	var obj any
	if err := json.Unmarshal([]byte(prettyJSON), &obj); err != nil {
		return prettyJSON // fallback: return as-is
	}
	compact, err := json.Marshal(obj)
	if err != nil {
		return prettyJSON
	}
	return string(compact)
}

// corruptionConfig returns a corruption config with a distribution matching
// the old stage 5 pipeline: mostly subtle/light, occasional medium.
func corruptionConfig(rng *rand.Rand) corrupt.Config {
	r := rng.Float64()
	switch {
	case r < 0.40:
		return corrupt.SubtleConfig()
	case r < 0.75:
		return corrupt.LightConfig()
	case r < 0.90:
		cfg := corrupt.LightConfig()
		cfg.WrapperProb = 1.0
		return cfg
	default:
		return corrupt.MediumConfig()
	}
}
