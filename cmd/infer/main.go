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
	"runtime"
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
	flag.IntVar(&maxSrcLen, "max-src-len", 1536, "max source token length")
	flag.IntVar(&maxTgtLen, "max-tgt-len", 2048, "max target generation length")
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

	// Session options with threading for CPU performance.
	sessionOpts, err := ort.NewSessionOptions()
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to create session options: %v\n", err)
		os.Exit(1)
	}
	defer sessionOpts.Destroy()
	nThreads := runtime.NumCPU()
	if err := sessionOpts.SetIntraOpNumThreads(nThreads); err != nil {
		fmt.Fprintf(os.Stderr, "warning: failed to set thread count: %v\n", err)
	}
	fmt.Printf("ORT threads: %d\n", nThreads)

	// Create encoder session (outputs cached K/V for cross-attention).
	encSession, err := ort.NewDynamicAdvancedSession(
		encoderPath, []string{"src_ids"}, []string{"all_k", "all_v"}, sessionOpts,
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to create encoder session: %v\n", err)
		os.Exit(1)
	}
	defer encSession.Destroy()

	// Create decoder session (single-step with KV cache).
	decSession, err := ort.NewDynamicAdvancedSession(
		decoderPath,
		[]string{"tgt_token", "all_k", "all_v", "all_h", "all_conv"},
		[]string{"logits", "all_h_out", "all_conv_out"},
		sessionOpts,
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

		// Encode (returns cached K/V).
		allK, allV, err := runEncoder(encSession, srcIDs)
		if err != nil {
			fmt.Fprintf(os.Stderr, "encoder error: %v\n", err)
			continue
		}

		// Decode (greedy, single-step autoregressive).
		predIDs, err := greedyDecode(decSession, allK, allV, bosID, eosID, maxTgtLen, nLayers, dInner, dState, dConv)
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

		allK.Destroy()
		allV.Destroy()
	}

	fmt.Printf("===== %d samples: exact=%d xml_ok=%d =====\n", total, exactCount, xmlOKCount)
}

// runEncoder runs the encoder ONNX model and returns cached K/V tensors.
func runEncoder(session *ort.DynamicAdvancedSession, srcIDs []int64) (*ort.Tensor[float32], *ort.Tensor[float32], error) {
	srcShape := ort.NewShape(1, int64(len(srcIDs)))
	srcTensor, err := ort.NewTensor(srcShape, srcIDs)
	if err != nil {
		return nil, nil, fmt.Errorf("create src tensor: %w", err)
	}
	defer srcTensor.Destroy()

	outputs := []ort.Value{nil, nil}
	err = session.Run([]ort.Value{srcTensor}, outputs)
	if err != nil {
		return nil, nil, fmt.Errorf("encoder run: %w", err)
	}

	kTensor, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return nil, nil, fmt.Errorf("unexpected encoder K output type")
	}
	vTensor, ok := outputs[1].(*ort.Tensor[float32])
	if !ok {
		return nil, nil, fmt.Errorf("unexpected encoder V output type")
	}
	return kTensor, vTensor, nil
}

// greedyDecode runs single-step autoregressive greedy decoding with KV cache.
func greedyDecode(
	session *ort.DynamicAdvancedSession,
	allK, allV *ort.Tensor[float32],
	bosID, eosID int64,
	maxLen, nLayers, dInner, dState, dConv int,
) ([]int64, error) {
	// Initialize Mamba state: all zeros.
	hSize := nLayers * dInner * dState
	convSize := nLayers * dInner * (dConv - 1)
	hData := make([]float32, hSize)
	convData := make([]float32, convSize)

	tgtIDs := []int64{}
	currentToken := bosID

	// Pre-allocate reusable tensors to avoid per-step allocation.
	tokenData := []int64{bosID}
	tokenTensor, err := ort.NewTensor(ort.NewShape(1, 1), tokenData)
	if err != nil {
		return nil, fmt.Errorf("create token tensor: %w", err)
	}
	defer tokenTensor.Destroy()

	hTensor, err := ort.NewTensor(
		ort.NewShape(int64(nLayers), int64(dInner), int64(dState)), hData,
	)
	if err != nil {
		return nil, fmt.Errorf("create h tensor: %w", err)
	}
	defer hTensor.Destroy()

	convTensor, err := ort.NewTensor(
		ort.NewShape(int64(nLayers), int64(dInner), int64(dConv-1)), convData,
	)
	if err != nil {
		return nil, fmt.Errorf("create conv tensor: %w", err)
	}
	defer convTensor.Destroy()

	for range maxLen {
		// Update token in-place.
		tokenTensor.GetData()[0] = currentToken

		// Run decoder step. K/V are read-only from encoder.
		outputs := []ort.Value{nil, nil, nil}
		err = session.Run(
			[]ort.Value{tokenTensor, allK, allV, hTensor, convTensor},
			outputs,
		)
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

		// Copy updated state back into reusable tensors.
		hOutTensor, ok := outputs[1].(*ort.Tensor[float32])
		if !ok {
			return nil, fmt.Errorf("unexpected h_out type")
		}
		copy(hTensor.GetData(), hOutTensor.GetData())
		hOutTensor.Destroy()

		convOutTensor, ok := outputs[2].(*ort.Tensor[float32])
		if !ok {
			return nil, fmt.Errorf("unexpected conv_out type")
		}
		copy(convTensor.GetData(), convOutTensor.GetData())
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
