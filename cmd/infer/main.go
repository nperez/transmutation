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

// Package main provides ONNX-based inference for the transmutation model.
// Reads JSONL from stdin, runs encoder + single-step decoder, compares output to targets.
package main

import (
	"bufio"
	"encoding/json"
	"encoding/xml"
	"flag"
	"fmt"
	"math"
	"os"
	"regexp"
	"strings"
	"time"

	"nickandperla.net/transmutation/pkg/sentencepiece"

	ort "github.com/yalue/onnxruntime_go"
)

type Record struct {
	Input  string `json:"input"`
	Target string `json:"target"`
}

func main() {
	var (
		encoderPath   string
		decoderPath   string
		tokenizerPath string
		ortLibPath    string
		nSamples      int
		maxSrcLen     int
		maxTgtLen     int
		nLayers       int
		dInner        int
		dState        int
		dConv         int
	)
	flag.StringVar(&encoderPath, "encoder", "models/onnx/encoder.onnx", "path to encoder ONNX")
	flag.StringVar(&decoderPath, "decoder", "models/onnx/decoder.onnx", "path to decoder ONNX")
	flag.StringVar(&tokenizerPath, "tokenizer", "models/tokenizer.model", "path to sentencepiece model")
	flag.StringVar(&ortLibPath, "ort-lib", "", "path to onnxruntime shared library")
	flag.IntVar(&nSamples, "n", 10, "number of samples to run")
	flag.IntVar(&maxSrcLen, "max-src-len", 256, "max source token length")
	flag.IntVar(&maxTgtLen, "max-tgt-len", 512, "max target generation length")
	flag.IntVar(&nLayers, "n-layers", 6, "number of decoder layers")
	flag.IntVar(&dInner, "d-inner", 768, "Mamba d_inner (d_model * expand)")
	flag.IntVar(&dState, "d-state", 16, "Mamba d_state")
	flag.IntVar(&dConv, "d-conv", 4, "Mamba d_conv")
	flag.Parse()

	// Initialize tokenizer.
	sp, err := sentencepiece.Load(tokenizerPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to load tokenizer: %v\n", err)
		os.Exit(1)
	}
	bosID := int64(sp.BOS())
	eosID := int64(sp.EOS())
	fmt.Printf("Tokenizer loaded: vocab=%d bos=%d eos=%d\n", sp.VocabSize(), bosID, eosID)

	// Initialize ONNX Runtime.
	if ortLibPath != "" {
		ort.SetSharedLibraryPath(ortLibPath)
	}
	if err := ort.InitializeEnvironment(); err != nil {
		fmt.Fprintf(os.Stderr, "failed to initialize onnxruntime: %v\n", err)
		os.Exit(1)
	}
	defer ort.DestroyEnvironment()

	// Create encoder session.
	encSession, err := ort.NewDynamicAdvancedSession(
		encoderPath, []string{"src_ids"}, []string{"memory"}, nil,
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to create encoder session: %v\n", err)
		os.Exit(1)
	}
	defer encSession.Destroy()

	// Create decoder session (single-step: token + memory + state → logits + state).
	decSession, err := ort.NewDynamicAdvancedSession(
		decoderPath,
		[]string{"tgt_token", "memory", "all_h", "all_conv"},
		[]string{"logits", "all_h_out", "all_conv_out"},
		nil,
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to create decoder session: %v\n", err)
		os.Exit(1)
	}
	defer decSession.Destroy()

	fmt.Println("ONNX sessions loaded")

	// Read and process samples from stdin.
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	wsNorm := regexp.MustCompile(`\s+`)
	exactCount := 0
	xmlOKCount := 0
	total := 0

	for scanner.Scan() {
		if total >= nSamples {
			break
		}
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var rec Record
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			fmt.Fprintf(os.Stderr, "skipping bad JSON line: %v\n", err)
			continue
		}

		// Tokenize and filter by length.
		srcTokens := sp.Encode(rec.Input, false, false)
		if len(srcTokens) > maxSrcLen {
			continue
		}
		srcIDs := make([]int64, len(srcTokens))
		for i, t := range srcTokens {
			srcIDs[i] = int64(t)
		}

		total++
		t0 := time.Now()

		// Encode.
		memory, err := runEncoder(encSession, srcIDs)
		if err != nil {
			fmt.Fprintf(os.Stderr, "encoder error: %v\n", err)
			continue
		}

		// Decode (greedy, single-step autoregressive).
		predIDs, err := greedyDecode(decSession, memory, bosID, eosID, maxTgtLen, nLayers, dInner, dState, dConv)
		if err != nil {
			fmt.Fprintf(os.Stderr, "decoder error: %v\n", err)
			continue
		}

		elapsed := time.Since(t0)

		// Decode tokens back to text.
		predInts := make([]int, len(predIDs))
		for i, id := range predIDs {
			predInts[i] = int(id)
		}
		pred := sp.Decode(predInts)

		normPred := wsNorm.ReplaceAllString(strings.TrimSpace(pred), " ")
		normTgt := wsNorm.ReplaceAllString(strings.TrimSpace(rec.Target), " ")
		exact := normPred == normTgt

		xmlOK := isValidXML(strings.TrimSpace(pred))

		if exact {
			exactCount++
		}
		if xmlOK {
			xmlOKCount++
		}

		tag := "FAIL"
		if exact {
			tag = "EXACT"
		} else if xmlOK {
			tag = "XML_OK"
		}

		fmt.Printf("=== Sample %d [%s] %.2fs, %d tokens ===\n", total, tag, elapsed.Seconds(), len(predIDs))
		fmt.Printf("INPUT:\n%s\n\n", rec.Input)
		if exact {
			fmt.Printf("OUTPUT (matches target):\n%s\n\n", strings.TrimSpace(pred))
		} else {
			fmt.Printf("TARGET:\n%s\n\n", strings.TrimSpace(rec.Target))
			fmt.Printf("OUTPUT:\n%s\n\n", strings.TrimSpace(pred))
		}
		fmt.Println()

		memory.Destroy()
	}

	fmt.Printf("===== %d samples: exact=%d xml_ok=%d =====\n", total, exactCount, xmlOKCount)
}

// runEncoder runs the encoder ONNX model and returns the memory tensor.
func runEncoder(session *ort.DynamicAdvancedSession, srcIDs []int64) (*ort.Tensor[float32], error) {
	srcShape := ort.NewShape(1, int64(len(srcIDs)))
	srcTensor, err := ort.NewTensor(srcShape, srcIDs)
	if err != nil {
		return nil, fmt.Errorf("create src tensor: %w", err)
	}
	defer srcTensor.Destroy()

	outputs := []ort.Value{nil}
	err = session.Run([]ort.Value{srcTensor}, outputs)
	if err != nil {
		return nil, fmt.Errorf("encoder run: %w", err)
	}

	memTensor, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("unexpected encoder output type")
	}
	return memTensor, nil
}

// greedyDecode runs single-step autoregressive greedy decoding.
func greedyDecode(
	session *ort.DynamicAdvancedSession,
	memory *ort.Tensor[float32],
	bosID, eosID int64,
	maxLen, nLayers, dInner, dState, dConv int,
) ([]int64, error) {
	memShape := memory.GetShape()
	memData := memory.GetData()

	// Initialize Mamba state: all zeros.
	hSize := nLayers * dInner * dState
	convSize := nLayers * dInner * (dConv - 1)
	hData := make([]float32, hSize)
	convData := make([]float32, convSize)

	tgtIDs := []int64{}
	currentToken := bosID

	for range maxLen {
		// Create single-token tensor.
		tokenTensor, err := ort.NewTensor(ort.NewShape(1, 1), []int64{currentToken})
		if err != nil {
			return nil, fmt.Errorf("create token tensor: %w", err)
		}

		// Clone memory.
		memClone, err := ort.NewTensor(ort.NewShape(memShape...), memData)
		if err != nil {
			tokenTensor.Destroy()
			return nil, fmt.Errorf("clone memory: %w", err)
		}

		// Create state tensors.
		hTensor, err := ort.NewTensor(
			ort.NewShape(int64(nLayers), int64(dInner), int64(dState)), hData,
		)
		if err != nil {
			tokenTensor.Destroy()
			memClone.Destroy()
			return nil, fmt.Errorf("create h tensor: %w", err)
		}

		convTensor, err := ort.NewTensor(
			ort.NewShape(int64(nLayers), int64(dInner), int64(dConv-1)), convData,
		)
		if err != nil {
			tokenTensor.Destroy()
			memClone.Destroy()
			hTensor.Destroy()
			return nil, fmt.Errorf("create conv tensor: %w", err)
		}

		// Run decoder step.
		outputs := []ort.Value{nil, nil, nil}
		err = session.Run(
			[]ort.Value{tokenTensor, memClone, hTensor, convTensor},
			outputs,
		)
		tokenTensor.Destroy()
		memClone.Destroy()
		hTensor.Destroy()
		convTensor.Destroy()

		if err != nil {
			return nil, fmt.Errorf("decoder run: %w", err)
		}

		// Extract logits.
		logitsTensor, ok := outputs[0].(*ort.Tensor[float32])
		if !ok {
			return nil, fmt.Errorf("unexpected logits type")
		}
		logitsData := logitsTensor.GetData()
		nextID := argmax(logitsData)
		logitsTensor.Destroy()

		// Extract updated state.
		hOutTensor, ok := outputs[1].(*ort.Tensor[float32])
		if !ok {
			return nil, fmt.Errorf("unexpected h_out type")
		}
		hData = make([]float32, hSize)
		copy(hData, hOutTensor.GetData())
		hOutTensor.Destroy()

		convOutTensor, ok := outputs[2].(*ort.Tensor[float32])
		if !ok {
			return nil, fmt.Errorf("unexpected conv_out type")
		}
		convData = make([]float32, convSize)
		copy(convData, convOutTensor.GetData())
		convOutTensor.Destroy()

		if int64(nextID) == eosID {
			break
		}
		tgtIDs = append(tgtIDs, int64(nextID))
		currentToken = int64(nextID)
	}

	return tgtIDs, nil
}

func argmax(data []float32) int {
	maxIdx := 0
	maxVal := float32(math.Inf(-1))
	for i, v := range data {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	return maxIdx
}

func isValidXML(s string) bool {
	d := xml.NewDecoder(strings.NewReader(s))
	for {
		_, err := d.Token()
		if err != nil {
			return err.Error() == "EOF"
		}
	}
}
