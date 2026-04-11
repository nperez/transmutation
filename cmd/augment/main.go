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
	"runtime"
	"strings"
	"sync"

	"nickandperla.net/transmutation/pkg/corrupt"
	"nickandperla.net/transmutation/pkg/randtext"
	"nickandperla.net/transmutation/pkg/sentencepiece"
	"nickandperla.net/transmutation/pkg/xmlconv"
)

type TrainingPair struct {
	Input      string `json:"input"`
	Target     string `json:"target"`
	AugType    string `json:"aug_type"`
	Complexity int    `json:"complexity"`
}

var specialProb float64
var corruptPct float64
var compactPct float64
var dictWordPct float64
var truncatePct float64
var shortenPct float64
var shortenNullProb float64
var shortenTruncProb float64

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
	flag.Float64Var(&dictWordPct, "dict-word-pct", 50, "percentage of augmented strings to replace with dictionary words vs shuffle (0=shuffle only, 100=all dict words)")
	flag.Float64Var(&truncatePct, "truncate-pct", 0, "percentage of samples to truncate (drop keys + cut string) (0-100)")
	flag.Float64Var(&shortenPct, "shorten-pct", 0, "percentage of samples to shorten by nulling verbose fields (0-100)")
	flag.Float64Var(&shortenNullProb, "shorten-null-prob", 0.5, "per-field probability of nulling when shortening (0-1)")
	flag.Float64Var(&shortenTruncProb, "shorten-trunc-prob", 0.5, "probability of truncating thought when shortening (0-1)")
	var dropMemoryPct float64
	flag.Float64Var(&dropMemoryPct, "drop-memory-pct", 20, "percentage of augmented samples to drop the memory key (0-100)")
	var minChars int
	flag.IntVar(&minChars, "min-chars", 0, "minimum input character length (0 = no filter)")
	var maxChars int
	flag.IntVar(&maxChars, "max-chars", 4769, "maximum input character length (0 = no filter, default fits 1152 tokens)")
	flag.StringVar(&sampleType, "type", "all", "sample type filter: answer, tool, or all")
	var maxComplexity int
	flag.IntVar(&maxComplexity, "max-complexity", 0, "filter samples above this complexity score (1-8, 0=no filter)")
	var tokenizerPath string
	flag.StringVar(&tokenizerPath, "tokenizer", "", "path to sentencepiece .model file for token-length binning")
	flag.Parse()

	// Load tokenizer for token-length binning (optional, falls back to char-length).
	var sp *sentencepiece.Processor
	if tokenizerPath != "" {
		var err error
		sp, err = sentencepiece.Load(tokenizerPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error loading tokenizer: %v\n", err)
			os.Exit(1)
		}
	}

	samples, err := loadHaiku(haikuDir, sampleType)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error loading haiku: %v\n", err)
		os.Exit(1)
	}
	orig := len(samples)
	if minChars > 0 || maxChars > 0 {
		filtered := samples[:0]
		for _, s := range samples {
			l := len(s.Input)
			if minChars > 0 && l < minChars {
				continue
			}
			if maxChars > 0 && l > maxChars {
				continue
			}
			filtered = append(filtered, s)
		}
		fmt.Fprintf(os.Stderr, "Loaded %d haiku samples, %d passed char filters (min=%d max=%d)\n", orig, len(filtered), minChars, maxChars)
		samples = filtered
	} else {
		fmt.Fprintf(os.Stderr, "Loaded %d haiku samples\n", orig)
	}

	// Filter by structural complexity.
	if maxComplexity > 0 {
		before := len(samples)
		filtered := samples[:0]
		for _, s := range samples {
			if complexityScore(s.Input) <= maxComplexity {
				filtered = append(filtered, s)
			}
		}
		samples = filtered
		fmt.Fprintf(os.Stderr, "  Complexity filter (max=%d): %d → %d samples\n", maxComplexity, before, len(samples))
	}

	// Deterministic shuffle of indices — same seed always gives same order.
	// Val uses offset seed so train and val sample disjoint subsets.
	sampSeed := seed
	if isVal {
		sampSeed = seed + 7777777
	}
	rng := rand.New(rand.NewPCG(sampSeed, sampSeed^0xdeadbeef))

	// Process ALL samples — stratification happens after augmentation.
	selected := make([]int, len(samples))
	for i := range selected {
		selected[i] = i
	}
	// Shuffle for determinism with seed.
	rng.Shuffle(len(selected), func(i, j int) {
		selected[i], selected[j] = selected[j], selected[i]
	})

	// Process samples in parallel, write results in order.
	type result struct {
		pairs     []TrainingPair
		natural   int
		augmented int
		augFailed int
		corrupted int
		compacted int
		truncated int
	}

	nWorkers := runtime.NumCPU()
	results := make([]result, len(selected))
	var wg sync.WaitGroup
	sem := make(chan struct{}, nWorkers)

	for si, idx := range selected {
		wg.Add(1)
		sem <- struct{}{}
		go func(si, idx int) {
			defer wg.Done()
			defer func() { <-sem }()

			sample := samples[idx]
			var r result

			// Natural sample — always emit the original.
			sample.AugType = "clean"
			r.pairs = append(r.pairs, sample)
			r.natural++

			// Emit additional variants (shortened, compacted, etc).
			// The original stays in the pool AND the variant is added.
			cSeed := seed + uint64(idx)*uint64(augRatio+1) + 999999
			cRng := rand.New(rand.NewPCG(cSeed, cSeed^0xf00d))
			if shortenPct > 0 && cRng.Float64()*100 < shortenPct {
				if s, err := shortenSample(sample, cRng); err == nil {
					s.AugType = "shortened"
					r.pairs = append(r.pairs, s)
				}
			}
			if compactPct > 0 && cRng.Float64()*100 < compactPct {
				compacted := TrainingPair{Input: compactJSON(sample.Input), Target: sample.Target, AugType: "compacted"}
				r.pairs = append(r.pairs, compacted)
				r.compacted++
			}
			if truncatePct > 0 && cRng.Float64()*100 < truncatePct {
				if t, err := truncateSample(sample, cRng); err == nil {
					t.AugType = "truncated"
					r.pairs = append(r.pairs, t)
					r.truncated++
				}
			}
			if corruptPct > 0 && cRng.Float64()*100 < corruptPct {
				corrupted := TrainingPair{
					Input:  corrupt.Apply(sample.Input, corruptionConfig(cRng), cRng),
					Target: sample.Target,
					AugType: "corrupted",
				}
				r.pairs = append(r.pairs, corrupted)
				r.corrupted++
			}

			// Augmented variants.
			for v := range augRatio {
				augSeed := seed + uint64(idx)*uint64(augRatio+1) + uint64(v) + 1
				augRng := rand.New(rand.NewPCG(augSeed, augSeed^0xcafebabe))

				aug, err := augmentSample(sample, augRng, dropMemoryPct)
				if err != nil {
					r.augFailed++
					continue
				}
				// Always emit the augmented variant as-is.
				aug.AugType = "augmented"
				r.pairs = append(r.pairs, aug)
				r.augmented++

				// Emit additional variants alongside the original augmented.
				if shortenPct > 0 && augRng.Float64()*100 < shortenPct {
					if s, err := shortenSample(aug, augRng); err == nil {
						s.AugType = "shortened"
						r.pairs = append(r.pairs, s)
					}
				}
				if compactPct > 0 && augRng.Float64()*100 < compactPct {
					compacted := TrainingPair{Input: compactJSON(aug.Input), Target: aug.Target, AugType: "compacted"}
					r.pairs = append(r.pairs, compacted)
					r.compacted++
				}
				if truncatePct > 0 && augRng.Float64()*100 < truncatePct {
					if t, err := truncateSample(aug, augRng); err == nil {
						t.AugType = "truncated"
						r.pairs = append(r.pairs, t)
						r.truncated++
					}
				}
				if corruptPct > 0 {
					cSeed := augSeed ^ 0xf00d
					cRng := rand.New(rand.NewPCG(cSeed, cSeed^0xbeef))
					if cRng.Float64()*100 < corruptPct {
						corrupted := TrainingPair{
							Input:  corrupt.Apply(aug.Input, corruptionConfig(cRng), cRng),
							Target: aug.Target,
							AugType: "corrupted",
						}
						r.pairs = append(r.pairs, corrupted)
						r.corrupted++
					}
				}
			}

			results[si] = r
		}(si, idx)
	}
	wg.Wait()

	// Collect all output pairs, then stratify by output input length.
	var allPairs []TrainingPair
	natural := 0
	augmented := 0
	augFailed := 0
	corrupted := 0
	compacted := 0
	truncated := 0

	for _, r := range results {
		allPairs = append(allPairs, r.pairs...)
		natural += r.natural
		augmented += r.augmented
		augFailed += r.augFailed
		corrupted += r.corrupted
		compacted += r.compacted
		truncated += r.truncated
	}

	// Stratified sampling on final output based on input char length.
	keep := int(float64(len(allPairs)) * samplePct / 100)
	if keep <= 0 {
		keep = 1
	}
	if keep > len(allPairs) {
		keep = len(allPairs)
	}
	outputIndices := stratifiedSample(allPairs, keep, rng, sp)

	bw := bufio.NewWriterSize(os.Stdout, 256*1024)
	defer bw.Flush()
	enc := json.NewEncoder(bw)
	for _, idx := range outputIndices {
		allPairs[idx].Complexity = complexityScore(allPairs[idx].Input)
		enc.Encode(allPairs[idx])
	}
	bw.Flush()

	split := "train"
	if isVal {
		split = "val"
	}
	fmt.Fprintf(os.Stderr, "Haiku augment (%s): %d natural + %d augmented = %d total (sampled %d = %.1f%% of %d",
		split, natural, augmented, natural+augmented, keep, samplePct, len(allPairs))
	if truncated > 0 {
		fmt.Fprintf(os.Stderr, ", %d truncated", truncated)
	}
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

// complexityScore assigns a 1-8 score based on structural complexity.
// Tool and answer are mutually exclusive in the agent schema, so max is 8.
//
//	Memory:  0 items=0, 1-3=+1, 4+=+2
//	Tool:    null=0, present=+2, 3+ args=+1, nested arg values=+1
//	Answer:  null=0, plain text=+1, markdown=+1, code blocks=+1, tables=+1
//	Length:  >2500 chars=+1
//
// Scores roughly map to curriculum stages:
//
//	1-3: flat answer, short, no/small memory
//	4-5: markdown answer or basic tool, some memory
//	6-8: code/tables or complex tool, full memory, long
func complexityScore(input string) int {
	score := 1 // base

	var obj map[string]json.RawMessage
	if err := json.Unmarshal([]byte(input), &obj); err != nil {
		return 1
	}

	// Memory array length.
	if raw, ok := obj["memory"]; ok {
		var mem []json.RawMessage
		if json.Unmarshal(raw, &mem) == nil && len(mem) > 0 {
			score++
			if len(mem) >= 4 {
				score++
			}
		}
	}

	// Tool presence and nesting depth.
	if raw, ok := obj["tool"]; ok && string(raw) != "null" {
		score += 2 // tool calls add object nesting
		var tool map[string]json.RawMessage
		if json.Unmarshal(raw, &tool) == nil {
			if args, aok := tool["arguments"]; aok {
				var argMap map[string]json.RawMessage
				if json.Unmarshal(args, &argMap) == nil {
					if len(argMap) >= 3 {
						score++ // many arguments
					}
					for _, v := range argMap {
						s := strings.TrimSpace(string(v))
						if len(s) > 0 && (s[0] == '{' || s[0] == '[') {
							score++ // nested object/array in args
							break
						}
					}
				}
			}
		}
	}

	// Answer content complexity.
	if raw, ok := obj["answer"]; ok && string(raw) != "null" {
		var answer string
		if json.Unmarshal(raw, &answer) == nil && len(answer) > 0 {
			score++ // has answer content
			if strings.Contains(answer, "## ") || strings.Contains(answer, "- ") {
				score++ // markdown
			}
			if strings.Contains(answer, "```") {
				score++ // code blocks
			}
			if strings.Contains(answer, "| ---") || strings.Contains(answer, "|---") {
				score++ // tables
			}
		}
	}

	// Overall length.
	if len(input) > 2500 {
		score++
	}

	if score > 8 {
		score = 8
	}
	return score
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
func augmentSample(sample TrainingPair, rng *rand.Rand, dropMemoryPct float64) (TrainingPair, error) {
	var obj any
	if err := json.Unmarshal([]byte(sample.Input), &obj); err != nil {
		return TrainingPair{}, fmt.Errorf("parse input: %w", err)
	}

	// Randomly drop the memory key to teach the model it's optional.
	if m, ok := obj.(map[string]any); ok && dropMemoryPct > 0 {
		if _, has := m["memory"]; has && rng.Float64()*100 < dropMemoryPct {
			delete(m, "memory")
		}
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

// augmentStringDictWords replaces a string value with random dictionary words
// and injects XML special characters.
func augmentStringDictWords(s string, rng *rand.Rand) string {
	n := len(strings.Fields(s))
	if n == 0 {
		n = 1 + rng.IntN(3)
	}
	newWords := make([]string, n)
	for i := range newWords {
		newWords[i] = randtext.DictWord(rng)
	}
	return randtext.InjectSpecialChars(rng, strings.Join(newWords, " "), specialProb)
}

// augmentStringShuffle shuffles the original words in place, preserving
// the real token complexity (code syntax, markdown, punctuation).
func augmentStringShuffle(s string, rng *rand.Rand) string {
	words := strings.Fields(s)
	if len(words) == 0 {
		return s
	}
	rng.Shuffle(len(words), func(i, j int) {
		words[i], words[j] = words[j], words[i]
	})
	return randtext.InjectSpecialChars(rng, strings.Join(words, " "), specialProb)
}

// augmentString dispatches to dict words or shuffle based on dictWordPct.
func augmentString(s string, rng *rand.Rand) string {
	if rng.Float64()*100 < dictWordPct {
		return augmentStringDictWords(s, rng)
	}
	return augmentStringShuffle(s, rng)
}

// shortenSample nulls verbose fields to create short pretty-printed samples.
// Randomly removes keys and truncates thought. Each decision is coin-flip
// controlled by the rng — no hardcoded biases. The result is a spectrum of
// shortened variants from mildly reduced to minimal.
func shortenSample(sample TrainingPair, rng *rand.Rand) (TrainingPair, error) {
	var obj map[string]any
	if err := json.Unmarshal([]byte(sample.Input), &obj); err != nil {
		return TrainingPair{}, fmt.Errorf("parse input: %w", err)
	}

	// Each field gets an independent probability of being nulled.
	if _, has := obj["memory"]; has && rng.Float64() < shortenNullProb {
		obj["memory"] = nil
	}
	if _, has := obj["answer"]; has && rng.Float64() < shortenNullProb {
		obj["answer"] = nil
	}
	if _, has := obj["tool"]; has && rng.Float64() < shortenNullProb {
		obj["tool"] = nil
	}

	// Truncate thought to 1-3 sentences.
	if thought, ok := obj["thought"].(string); ok && rng.Float64() < shortenTruncProb {
		sentences := splitSentences(thought)
		if len(sentences) > 1 {
			keep := 1 + rng.IntN(min(3, len(sentences)))
			obj["thought"] = strings.Join(sentences[:keep], " ")
		}
	}

	pretty, err := json.MarshalIndent(obj, "", "  ")
	if err != nil {
		return TrainingPair{}, fmt.Errorf("marshal: %w", err)
	}

	xmlOut, err := xmlconv.Convert(pretty)
	if err != nil {
		return TrainingPair{}, fmt.Errorf("xmlconv: %w", err)
	}

	return TrainingPair{Input: string(pretty), Target: xmlOut}, nil
}

// splitSentences splits text on sentence boundaries (. ! ? followed by space or end).
func splitSentences(s string) []string {
	var sentences []string
	start := 0
	for i := 0; i < len(s)-1; i++ {
		if (s[i] == '.' || s[i] == '!' || s[i] == '?') && (s[i+1] == ' ' || s[i+1] == '\n') {
			sentences = append(sentences, strings.TrimSpace(s[start:i+1]))
			start = i + 2
		}
	}
	if start < len(s) {
		sentences = append(sentences, strings.TrimSpace(s[start:]))
	}
	return sentences
}


// Length bins for stratified sampling.
// Token bins: finer in the 256-768 range where val exact drops.
// Char bins: fallback when no tokenizer loaded.
var tokenBins = []int{0, 64, 128, 256, 384, 512, 768, 1024}
var charBins = []int{0, 300, 600, 900, 1200, 1600, 2200, 3000}

// Augmentation types for 2D stratification.
var augTypes = []string{"clean", "augmented", "shortened", "compacted", "corrupted", "truncated"}

// stratifiedSample does 2D stratification: length bin × augmentation type.
// Each (length, augType) cell gets equal representation in the output.
// Cells with fewer samples than quota contribute what they have.
// When sp is non-nil, bins by token count; otherwise falls back to char count.
func stratifiedSample(samples []TrainingPair, total int, rng *rand.Rand, sp *sentencepiece.Processor) []int {
	bins := charBins
	if sp != nil {
		bins = tokenBins
	}
	numLenBins := len(bins)
	numAugTypes := len(augTypes)

	// Build augType index for fast lookup.
	augIdx := make(map[string]int, numAugTypes)
	for i, t := range augTypes {
		augIdx[t] = i
	}

	// 2D grid: bins[lenBin][augType] = []sampleIndex
	grid := make([][][]int, numLenBins)
	for i := range grid {
		grid[i] = make([][]int, numAugTypes)
		for j := range grid[i] {
			grid[i][j] = []int{}
		}
	}

	// Pre-compute lengths (parallel tokenization when sp is available).
	sampleLens := make([]int, len(samples))
	if sp != nil {
		var wg sync.WaitGroup
		nWorkers := runtime.NumCPU()
		chunkSize := (len(samples) + nWorkers - 1) / nWorkers
		for w := range nWorkers {
			lo := w * chunkSize
			hi := min(lo+chunkSize, len(samples))
			if lo >= hi {
				break
			}
			wg.Add(1)
			go func(lo, hi int) {
				defer wg.Done()
				for i := lo; i < hi; i++ {
					sampleLens[i] = len(sp.Encode(samples[i].Input, false, false))
				}
			}(lo, hi)
		}
		wg.Wait()
	} else {
		for i, s := range samples {
			sampleLens[i] = len(s.Input)
		}
	}

	// Assign each sample to a (length, augType) cell.
	for idx := range samples {
		lb := numLenBins - 1
		for b := numLenBins - 1; b >= 0; b-- {
			if sampleLens[idx] >= bins[b] {
				lb = b
				break
			}
		}
		at, ok := augIdx[samples[idx].AugType]
		if !ok {
			at = 0 // default to "clean" if untagged
		}
		grid[lb][at] = append(grid[lb][at], idx)
	}

	// Shuffle each cell.
	for i := range grid {
		for j := range grid[i] {
			rng.Shuffle(len(grid[i][j]), func(a, b int) {
				grid[i][j][a], grid[i][j][b] = grid[i][j][b], grid[i][j][a]
			})
		}
	}

	// Count non-empty cells to determine per-cell quota.
	nonEmpty := 0
	for i := range grid {
		for j := range grid[i] {
			if len(grid[i][j]) > 0 {
				nonEmpty++
			}
		}
	}
	if nonEmpty == 0 {
		return nil
	}

	perCell := max(total/nonEmpty, 1)

	var selected []int
	for i := range grid {
		for j := range grid[i] {
			cell := grid[i][j]
			quota := min(perCell, len(cell))
			selected = append(selected, cell[:quota]...)
		}
	}

	return selected
}

// truncateSample simulates LLM output truncation by dropping 1-2 random keys
// from the JSON input and regenerating the target XML to match. Optionally
// also truncates the JSON string mid-stream (50% chance) to simulate hard cutoff.
func truncateSample(sample TrainingPair, rng *rand.Rand) (TrainingPair, error) {
	// Drop keys from the input JSON.
	reduced, err := corrupt.DropKeys(sample.Input, rng)
	if err != nil {
		return TrainingPair{}, fmt.Errorf("drop keys: %w", err)
	}

	// Regenerate target XML from the reduced JSON.
	xmlOut, err := xmlconv.Convert([]byte(reduced))
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

	// 50% chance: also truncate the JSON string mid-stream (hard cutoff).
	if rng.Float64() < 0.5 {
		reduced = corrupt.TruncateJSON(reduced, rng)
	}

	return TrainingPair{Input: reduced, Target: xmlOut}, nil
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
