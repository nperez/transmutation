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
	"io"
	"math/rand/v2"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"nickandperla.net/transmutation/pkg/agent"
	"nickandperla.net/transmutation/pkg/corrupt"
	"nickandperla.net/transmutation/pkg/jsongen"
	"nickandperla.net/transmutation/pkg/languages"
	"nickandperla.net/transmutation/pkg/xmlconv"
)

// TrainingPair is one input/output example written as a JSONL record.
type TrainingPair struct {
	Input  string `json:"input"`
	Target string `json:"target"`
}

var stage int

func main() {
	var (
		trainCount int
		valCount   int
		outDir     string
		seed       uint64
		workers    int
		toStdout   bool
		wrapStdin  bool
		mixDir     string
	)
	flag.IntVar(&trainCount, "train", 1000000, "number of training pairs")
	flag.IntVar(&valCount, "val", 50000, "number of validation pairs")
	flag.StringVar(&outDir, "out", "data", "output directory")
	flag.Uint64Var(&seed, "seed", 42, "base random seed")
	flag.IntVar(&workers, "workers", runtime.NumCPU(), "number of parallel workers")
	flag.BoolVar(&toStdout, "stdout", false, "write JSONL to stdout instead of files")
	flag.IntVar(&stage, "stage", 0, "curriculum stage (1-5); 0 = legacy random JSON")
	flag.BoolVar(&wrapStdin, "wrap", false, "read JSON from stdin, convert to JSONL training pairs on stdout")
	var mixPct float64
	flag.StringVar(&mixDir, "mix", "", "directory of extra JSONL files to copy into train output")
	flag.Float64Var(&mixPct, "mix-pct", 100, "percentage of mix lines to include (0-100)")
	flag.Parse()

	if stage < 0 || stage > 5 {
		fmt.Fprintf(os.Stderr, "error: stage must be 0-5\n")
		os.Exit(1)
	}

	// Wrap mode: read JSON objects from stdin, convert each to a training pair.
	if wrapStdin {
		if err := wrapFromStdin(); err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
		return
	}

	// Stdout mode: generate count pairs directly to stdout, no files.
	if toStdout {
		count := trainCount + valCount
		var generated atomic.Int64
		if err := generateToWriter(os.Stdout, count, seed, &generated); err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
		return
	}

	if err := os.MkdirAll(filepath.Join(outDir, "train"), 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "error creating output dir: %v\n", err)
		os.Exit(1)
	}
	if err := os.MkdirAll(filepath.Join(outDir, "val"), 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "error creating output dir: %v\n", err)
		os.Exit(1)
	}

	stageLabel := "legacy"
	if stage > 0 {
		stageLabel = fmt.Sprintf("stage %d", stage)
	}
	fmt.Printf("Generating %d train + %d val pairs (%s) with %d workers\n",
		trainCount, valCount, stageLabel, workers)

	var trainDur, valDur time.Duration

	if trainCount > 0 {
		start := time.Now()
		generateSplit(filepath.Join(outDir, "train"), trainCount, seed, workers)
		trainDur = time.Since(start)
		fmt.Printf("Train: %d pairs in %s (%.0f pairs/sec)\n", trainCount, trainDur, float64(trainCount)/trainDur.Seconds())

		if mixDir != "" {
			mixed := mixExtraData(mixDir, filepath.Join(outDir, "train"), mixPct, seed)
			if mixed > 0 {
				fmt.Printf("Mixed (train): %d extra pairs from %s (%.0f%%)\n", mixed, mixDir, mixPct)
			}
		}
	}

	if valCount > 0 {
		start := time.Now()
		generateSplit(filepath.Join(outDir, "val"), valCount, seed+1000000, workers)
		valDur = time.Since(start)
		fmt.Printf("Val:   %d pairs in %s (%.0f pairs/sec)\n", valCount, valDur, float64(valCount)/valDur.Seconds())

		if mixDir != "" {
			mixed := mixExtraData(mixDir, filepath.Join(outDir, "val"), mixPct, seed+2000000)
			if mixed > 0 {
				fmt.Printf("Mixed (val): %d extra pairs from %s (%.0f%%)\n", mixed, mixDir, mixPct)
			}
		}
	}

	fmt.Printf("Total: %s\n", trainDur+valDur)
}

func generateSplit(dir string, count int, baseSeed uint64, workers int) {
	pairsPerWorker := count / workers
	remainder := count % workers

	var wg sync.WaitGroup
	var generated atomic.Int64

	done := make(chan struct{})
	go func() {
		ticker := time.NewTicker(2 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-done:
				return
			case <-ticker.C:
				n := generated.Load()
				fmt.Printf("  %s: %d / %d (%.1f%%)\n", dir, n, count, float64(n)/float64(count)*100)
			}
		}
	}()

	for w := range workers {
		wg.Add(1)
		n := pairsPerWorker
		if w < remainder {
			n++
		}
		workerSeed := baseSeed + uint64(w)*1000000
		outPath := filepath.Join(dir, fmt.Sprintf("shard_%04d.jsonl", w))

		go func(n int, seed uint64, path string) {
			defer wg.Done()
			if err := generateShard(path, n, seed, &generated); err != nil {
				fmt.Fprintf(os.Stderr, "worker error: %v\n", err)
			}
		}(n, workerSeed, outPath)
	}

	wg.Wait()
	close(done)
}

func generateShard(path string, count int, baseSeed uint64, generated *atomic.Int64) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create %s: %w", path, err)
	}
	defer f.Close()

	return generateToWriter(f, count, baseSeed, generated)
}

func generateToWriter(w io.Writer, count int, baseSeed uint64, generated *atomic.Int64) error {
	bw := bufio.NewWriterSize(w, 256*1024)
	defer bw.Flush()

	enc := json.NewEncoder(bw)
	langGens := languages.AsJsongenGenerators()

	for i := range count {
		seed := baseSeed + uint64(i)
		rng := rand.New(rand.NewPCG(seed, seed+1))

		var pair TrainingPair
		if stage > 0 {
			pair = generateAgentPair(rng)
		} else {
			pair = generateLegacyPair(rng, langGens)
		}
		if err := enc.Encode(pair); err != nil {
			return fmt.Errorf("encode: %w", err)
		}
		generated.Add(1)
	}

	return bw.Flush()
}

// generateAgentPair produces a training pair using the agent response schema.
func generateAgentPair(rng *rand.Rand) TrainingPair {
	gen := agent.NewGenerator(rng, agent.Stage(stage))
	cleanJSON, xmlOut := gen.Generate()

	corrCfg := stageCorruptionConfig(rng)
	input := corrupt.Apply(cleanJSON, corrCfg, rng)

	return TrainingPair{Input: input, Target: xmlOut}
}

// stageCorruptionConfig returns a corruption config appropriate for the
// current curriculum stage.
func stageCorruptionConfig(rng *rand.Rand) corrupt.Config {
	switch stage {
	case 1, 2, 3:
		// Stages 1-3: 100% clean (model learns faithful transcription).
		return corrupt.NoneConfig()
	case 4:
		// Stage 4: 95% clean, 3% subtle, 2% light.
		r := rng.Float64()
		if r < 0.95 {
			return corrupt.NoneConfig()
		} else if r < 0.98 {
			return corrupt.SubtleConfig()
		}
		return corrupt.LightConfig()
	case 5:
		// Stage 5: 90% clean, 5% subtle, 3% light, 2% light+wrapper.
		r := rng.Float64()
		if r < 0.90 {
			return corrupt.NoneConfig()
		} else if r < 0.95 {
			return corrupt.SubtleConfig()
		} else if r < 0.98 {
			return corrupt.LightConfig()
		}
		cfg := corrupt.LightConfig()
		cfg.WrapperProb = 1.0
		return cfg
	default:
		return corrupt.NoneConfig()
	}
}

// --- Legacy pipeline (stage 0) ---

func generateLegacyPair(rng *rand.Rand, langGens []jsongen.LanguageGenerator) TrainingPair {
	cfg := randomConfig(rng)
	gen := jsongen.NewGenerator(cfg, rng)
	node := gen.Generate()

	pop := jsongen.NewValuePopulator(rng, langGens)
	pop.Populate(node)

	cleanJSON := jsongen.Serialize(node)

	xmlOut, err := xmlconv.Convert([]byte(cleanJSON))
	if err != nil {
		panic(fmt.Sprintf("xmlconv failed on generated JSON: %v", err))
	}

	corrCfg := randomCorruptionConfig(rng)
	corrupted := corrupt.Apply(cleanJSON, corrCfg, rng)

	return TrainingPair{Input: corrupted, Target: xmlOut}
}

func randomConfig(rng *rand.Rand) jsongen.Config {
	var sizeBudget int
	r := rng.Float64()
	switch {
	case r < 0.3:
		sizeBudget = 5 + rng.IntN(15)
	case r < 0.7:
		sizeBudget = 20 + rng.IntN(80)
	case r < 0.9:
		sizeBudget = 100 + rng.IntN(400)
	default:
		sizeBudget = 500 + rng.IntN(1500)
	}

	return jsongen.Config{
		MaxDepth:   2 + rng.IntN(6),
		MaxBreadth: 2 + rng.IntN(12),
		MinBreadth: 1 + rng.IntN(3),
		SizeBudget: sizeBudget,
		ArrayProb:  0.1 + rng.Float64()*0.4,
		ScalarDist: jsongen.ScalarDistribution{
			String: 0.3 + rng.Float64()*0.4,
			Number: 0.1 + rng.Float64()*0.2,
			Bool:   0.05 + rng.Float64()*0.15,
			Null:   0.05 + rng.Float64()*0.1,
		},
	}
}

func randomCorruptionConfig(rng *rand.Rand) corrupt.Config {
	r := rng.Float64()
	switch {
	case r < 0.25:
		return corrupt.NoneConfig()
	case r < 0.60:
		return corrupt.LightConfig()
	case r < 0.90:
		return corrupt.MediumConfig()
	default:
		return corrupt.HeavyConfig()
	}
}

// mixExtraData samples lines from JSONL files in srcDir and writes a single
// mixed shard to dstDir. pct controls what percentage of lines to include
// (0-100). seed ensures different samples each epoch.
func mixExtraData(srcDir, dstDir string, pct float64, seed uint64) int {
	entries, err := os.ReadDir(srcDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "warn: cannot read mix dir %s: %v\n", srcDir, err)
		return 0
	}

	// Collect all lines from all JSONL files.
	var allLines []string
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".jsonl") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(srcDir, e.Name()))
		if err != nil {
			continue
		}
		for _, line := range strings.Split(string(data), "\n") {
			if line = strings.TrimSpace(line); line != "" {
				allLines = append(allLines, line)
			}
		}
	}

	if len(allLines) == 0 {
		return 0
	}

	// Sample lines.
	rng := rand.New(rand.NewPCG(seed, seed^0xdeadbeef))
	keep := int(float64(len(allLines)) * pct / 100)
	if keep <= 0 {
		return 0
	}
	if keep >= len(allLines) {
		keep = len(allLines)
	}

	// Fisher-Yates shuffle, take first 'keep'.
	for i := len(allLines) - 1; i > 0; i-- {
		j := rng.IntN(i + 1)
		allLines[i], allLines[j] = allLines[j], allLines[i]
	}
	sampled := allLines[:keep]

	// Write single mixed shard.
	dst := filepath.Join(dstDir, "mix_haiku.jsonl")
	var b strings.Builder
	for _, line := range sampled {
		b.WriteString(line)
		b.WriteByte('\n')
	}
	if err := os.WriteFile(dst, []byte(b.String()), 0o644); err != nil {
		fmt.Fprintf(os.Stderr, "warn: cannot write %s: %v\n", dst, err)
		return 0
	}
	return keep
}

// wrapFromStdin reads JSON objects from stdin (one per line or as a stream),
// validates each thoroughly, converts to XML, verifies XML validity,
// and writes JSONL pairs to stdout. Rejects any sample that doesn't meet
// quality gates.
func wrapFromStdin() error {
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 0, 1024*1024), 10*1024*1024) // 10MB max line
	enc := json.NewEncoder(os.Stdout)

	lineNum := 0
	accepted := 0
	rejected := 0

	for scanner.Scan() {
		lineNum++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		// Gate 1: valid JSON.
		if !json.Valid([]byte(line)) {
			fmt.Fprintf(os.Stderr, "REJECT line %d: invalid JSON\n", lineNum)
			rejected++
			continue
		}

		// Gate 2: must be a JSON object with expected schema fields.
		var obj map[string]any
		if err := json.Unmarshal([]byte(line), &obj); err != nil {
			fmt.Fprintf(os.Stderr, "REJECT line %d: not a JSON object: %v\n", lineNum, err)
			rejected++
			continue
		}

		// Check required fields exist.
		if _, ok := obj["thought"]; !ok {
			fmt.Fprintf(os.Stderr, "REJECT line %d: missing 'thought' field\n", lineNum)
			rejected++
			continue
		}
		if _, ok := obj["memory"]; !ok {
			fmt.Fprintf(os.Stderr, "REJECT line %d: missing 'memory' field\n", lineNum)
			rejected++
			continue
		}

		// Must have exactly one of answer/tool non-null.
		answerNull := obj["answer"] == nil
		toolNull := obj["tool"] == nil
		if answerNull == toolNull {
			fmt.Fprintf(os.Stderr, "REJECT line %d: need exactly one of answer/tool non-null (answer=%v, tool=%v)\n",
				lineNum, !answerNull, !toolNull)
			rejected++
			continue
		}

		// thought must be a non-empty string.
		thought, ok := obj["thought"].(string)
		if !ok || len(thought) < 10 {
			fmt.Fprintf(os.Stderr, "REJECT line %d: thought must be a string with 10+ chars\n", lineNum)
			rejected++
			continue
		}

		// memory must be a non-empty array.
		mem, ok := obj["memory"].([]any)
		if !ok || len(mem) < 1 {
			fmt.Fprintf(os.Stderr, "REJECT line %d: memory must be a non-empty array\n", lineNum)
			rejected++
			continue
		}

		// Gate 3: pretty-print for consistent formatting.
		pretty, err := json.MarshalIndent(obj, "", "  ")
		if err != nil {
			fmt.Fprintf(os.Stderr, "REJECT line %d: marshal error: %v\n", lineNum, err)
			rejected++
			continue
		}

		// Gate 4: convert to XML.
		xmlOut, err := xmlconv.Convert(pretty)
		if err != nil {
			fmt.Fprintf(os.Stderr, "REJECT line %d: xmlconv error: %v\n", lineNum, err)
			rejected++
			continue
		}

		// Gate 5: verify XML is parseable.
		xmlWrapped := "<root>" + xmlOut + "</root>"
		decoder := xmlDecoder(xmlWrapped)
		xmlValid := true
		for {
			_, err := decoder.Token()
			if err != nil {
				if err.Error() != "EOF" {
					fmt.Fprintf(os.Stderr, "REJECT line %d: generated XML is not parseable: %v\n", lineNum, err)
					xmlValid = false
				}
				break
			}
		}
		if !xmlValid {
			rejected++
			continue
		}

		pair := TrainingPair{Input: string(pretty), Target: xmlOut}
		if err := enc.Encode(pair); err != nil {
			return fmt.Errorf("encode pair %d: %w", lineNum, err)
		}
		accepted++
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("reading stdin: %w", err)
	}

	fmt.Fprintf(os.Stderr, "Wrap results: %d accepted, %d rejected out of %d lines\n", accepted, rejected, lineNum)
	return nil
}

func xmlDecoder(s string) *xml.Decoder {
	return xml.NewDecoder(strings.NewReader(s))
}
