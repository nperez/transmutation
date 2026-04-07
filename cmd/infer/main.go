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
	"sort"
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
		beamWidth     int
		lengthPenalty float64
		debugSteps    int
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
	flag.IntVar(&dState, "d-state", 64, "Mamba d_state")
	flag.IntVar(&dConv, "d-conv", 4, "Mamba1 d_conv (ignored for Mamba3)")
	var nHeadsSSM int
	var headDimSSM int
	var numRopeAngles int
	flag.IntVar(&nHeadsSSM, "n-heads-ssm", 12, "Mamba3 nheads (d_inner/headdim)")
	flag.IntVar(&headDimSSM, "headdim-ssm", 64, "Mamba3 headdim")
	flag.IntVar(&numRopeAngles, "num-rope-angles", 16, "Mamba3 num_rope_angles")
	flag.IntVar(&beamWidth, "beam-width", 1, "beam width (1 = greedy)")
	flag.Float64Var(&lengthPenalty, "length-penalty", 0.6, "length normalization exponent for beam search")
	flag.IntVar(&debugSteps, "debug-steps", 0, "print per-step token IDs and logit stats for first N steps of sample 1")
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

	// Create decoder session — auto-detect Mamba3 vs Mamba1 from ONNX input names.
	// Mamba3: 8 inputs (tgt_token, all_k, all_v, all_angle, all_ssm, all_k_state, all_v_state, src_ids)
	// Mamba1: 6 inputs (tgt_token, all_k, all_v, all_h, all_conv, src_ids)
	isMamba3 := false
	decInputs := []string{"tgt_token", "all_k", "all_v", "all_h", "all_conv", "src_ids"}
	decOutputs := []string{"log_probs", "all_h_out", "all_conv_out"}

	// Try Mamba3 first, fall back to Mamba1.
	decSession, err := ort.NewDynamicAdvancedSession(
		decoderPath,
		[]string{"tgt_token", "all_k", "all_v", "all_angle", "all_ssm", "all_k_state", "all_v_state", "src_ids"},
		[]string{"log_probs", "all_angle_out", "all_ssm_out", "all_k_state_out", "all_v_state_out"},
		sessionOpts,
	)
	if err == nil {
		isMamba3 = true
		decInputs = []string{"tgt_token", "all_k", "all_v", "all_angle", "all_ssm", "all_k_state", "all_v_state", "src_ids"}
		decOutputs = []string{"log_probs", "all_angle_out", "all_ssm_out", "all_k_state_out", "all_v_state_out"}
	} else {
		// Fall back to Mamba1.
		decSession, err = ort.NewDynamicAdvancedSession(
			decoderPath, decInputs, decOutputs, sessionOpts,
		)
		if err != nil {
			fmt.Fprintf(os.Stderr, "failed to create decoder session: %v\n", err)
			os.Exit(1)
		}
	}
	defer decSession.Destroy()

	if isMamba3 {
		fmt.Println("ONNX sessions loaded (Mamba3)")
	} else {
		fmt.Println("ONNX sessions loaded (Mamba1)")
	}
	if beamWidth > 1 {
		fmt.Printf("Beam search: width=%d, length_penalty=%.2f\n", beamWidth, lengthPenalty)
	}
	_ = decInputs
	_ = decOutputs

	// Read and process samples from stdin.
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	wsNorm := regexp.MustCompile(`\s+`)
	exactCount := 0
	semanticCount := 0
	xmlOKCount := 0
	total := 0
	totalCEREdits := 0
	totalCERChars := 0
	totalWEREdits := 0
	totalWERWords := 0

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

		// Decode (greedy or beam search).
		debug := 0
		if total == 1 && debugSteps > 0 {
			debug = debugSteps
		}

		var predIDs []int64
		if isMamba3 {
			predIDs, err = greedyDecodeMamba3(decSession, allK, allV, srcIDs, bosID, eosID,
				maxTgtLen, nLayers, nHeadsSSM, headDimSSM, dState, numRopeAngles, debug)
		} else if beamWidth > 1 {
			predIDs, err = beamDecode(decSession, allK, allV, srcIDs, bosID, eosID,
				maxTgtLen, nLayers, dInner, dState, dConv, beamWidth, lengthPenalty)
		} else {
			predIDs, err = greedyDecode(decSession, allK, allV, srcIDs, bosID, eosID,
				maxTgtLen, nLayers, dInner, dState, dConv, debug)
		}
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
		semantic := false
		if !exact && xmlOK {
			semantic = xmlSemanticallyEqual(strings.TrimSpace(pred), strings.TrimSpace(rec.Target))
		}

		if exact {
			exactCount++
		}
		if semantic {
			semanticCount++
		}
		if xmlOK {
			xmlOKCount++
		}

		// CER on whitespace-normalized text (character-level Levenshtein).
		// WER: character-weighted word error rate — Levenshtein on words,
		// but each edit weighted by the character length of the affected word.
		charEdits := 0
		werChars := 0
		if !exact {
			charEdits = levenshtein(toChars(normPred), toChars(normTgt))
			werChars = charWeightedWER(strings.Fields(normPred), strings.Fields(normTgt))
		}
		totalCEREdits += charEdits
		totalCERChars += len([]rune(normTgt))
		totalWEREdits += werChars
		totalWERWords += len([]rune(normTgt))

		tag := "FAIL"
		if exact {
			tag = "EXACT"
		} else if semantic {
			tag = "SEMANTIC"
		} else if xmlOK {
			tag = "XML_OK"
		}

		cerDetail := ""
		if !exact && len(normTgt) > 0 {
			refChars := float64(len([]rune(normTgt)))
			sampleCER := float64(charEdits) / refChars * 100
			sampleWER := float64(werChars) / refChars * 100
			cerDetail = fmt.Sprintf(" cer=%.1f%% wer=%.1f%%", sampleCER, sampleWER)
		}

		fmt.Printf("=== Sample %d [%s] %.2fs, %d tokens%s ===\n", total, tag, elapsed.Seconds(), len(predIDs), cerDetail)
		fmt.Printf("INPUT:\n%s\n\n", rec.Input)
		if exact || semantic {
			fmt.Printf("OUTPUT (matches target):\n%s\n\n", strings.TrimSpace(pred))
		} else {
			fmt.Printf("TARGET:\n%s\n\n", strings.TrimSpace(rec.Target))
			fmt.Printf("OUTPUT:\n%s\n\n", strings.TrimSpace(pred))
		}
		fmt.Println()

		allK.Destroy()
		allV.Destroy()
	}

	overallCER := float64(0)
	if totalCERChars > 0 {
		overallCER = float64(totalCEREdits) / float64(totalCERChars) * 100
	}
	overallWER := float64(0)
	if totalWERWords > 0 {
		// WER denominator is total ref chars (same as CER) since edits are char-weighted.
		overallWER = float64(totalWEREdits) / float64(totalWERWords) * 100
	}
	fmt.Printf("===== %d samples: exact=%d semantic=%d xml_ok=%d fail=%d CER=%.2f%% WER=%.2f%% =====\n",
		total, exactCount, semanticCount, xmlOKCount-exactCount-semanticCount, total-xmlOKCount, overallCER, overallWER)
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
	srcIDs []int64,
	bosID, eosID int64,
	maxLen, nLayers, dInner, dState, dConv int,
	debugSteps int,
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

	// Source IDs tensor for copy mechanism (constant per sample).
	srcIDsTensor, err := ort.NewTensor(ort.NewShape(1, int64(len(srcIDs))), srcIDs)
	if err != nil {
		return nil, fmt.Errorf("create src_ids tensor: %w", err)
	}
	defer srcIDsTensor.Destroy()

	for range maxLen {
		// Update token in-place.
		tokenTensor.GetData()[0] = currentToken

		// Run decoder step. K/V and src_ids are read-only from encoder.
		outputs := []ort.Value{nil, nil, nil}
		err = session.Run(
			[]ort.Value{tokenTensor, allK, allV, hTensor, convTensor, srcIDsTensor},
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

		// Save top-3 for debug before destroying tensor.
		var debugTop3 []int
		var debugLogitMax float32
		step := len(tgtIDs)
		if debugSteps > 0 && step < debugSteps {
			debugTop3 = topKIndices(func() []float64 {
				f := make([]float64, len(logitsData))
				for i, v := range logitsData {
					f[i] = float64(v)
				}
				return f
			}(), 3)
			for _, v := range logitsData {
				if v > debugLogitMax {
					debugLogitMax = v
				}
				if -v > debugLogitMax {
					debugLogitMax = -v
				}
			}
		}
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

		if debugSteps > 0 && step < debugSteps {
			// Read h from the copied-into tensor (post-copy, pre-next-step).
			hMax := float32(0)
			for _, v := range hTensor.GetData() {
				if v > hMax {
					hMax = v
				}
				if -v > hMax {
					hMax = -v
				}
			}
			fmt.Fprintf(os.Stderr, "  GO step %3d: id=%5d  logit_max=%.4f  h_absmax=%.6f  top3=%v\n",
				step, nextID, debugLogitMax, hMax, debugTop3)
		}

		if int64(nextID) == eosID {
			break
		}
		tgtIDs = append(tgtIDs, int64(nextID))
		currentToken = int64(nextID)
	}

	return tgtIDs, nil
}

func greedyDecodeMamba3(
	session *ort.DynamicAdvancedSession,
	allK, allV *ort.Tensor[float32],
	srcIDs []int64,
	bosID, eosID int64,
	maxLen, nLayers, nHeads, headDim, dState, numRopeAngles int,
	debugSteps int,
) ([]int64, error) {
	// Initialize Mamba3 state: 4 tensors, all zeros.
	angleData := make([]float32, nLayers*nHeads*numRopeAngles)
	ssmData := make([]float32, nLayers*nHeads*headDim*dState)
	ksData := make([]float32, nLayers*nHeads*dState)
	vsData := make([]float32, nLayers*nHeads*headDim)

	tgtIDs := []int64{}
	currentToken := bosID

	tokenData := []int64{bosID}
	tokenTensor, err := ort.NewTensor(ort.NewShape(1, 1), tokenData)
	if err != nil {
		return nil, fmt.Errorf("create token tensor: %w", err)
	}
	defer tokenTensor.Destroy()

	angleTensor, err := ort.NewTensor(
		ort.NewShape(int64(nLayers), int64(nHeads), int64(numRopeAngles)), angleData)
	if err != nil {
		return nil, fmt.Errorf("create angle tensor: %w", err)
	}
	defer angleTensor.Destroy()

	ssmTensor, err := ort.NewTensor(
		ort.NewShape(int64(nLayers), int64(nHeads), int64(headDim), int64(dState)), ssmData)
	if err != nil {
		return nil, fmt.Errorf("create ssm tensor: %w", err)
	}
	defer ssmTensor.Destroy()

	ksTensor, err := ort.NewTensor(
		ort.NewShape(int64(nLayers), int64(nHeads), int64(dState)), ksData)
	if err != nil {
		return nil, fmt.Errorf("create k_state tensor: %w", err)
	}
	defer ksTensor.Destroy()

	vsTensor, err := ort.NewTensor(
		ort.NewShape(int64(nLayers), int64(nHeads), int64(headDim)), vsData)
	if err != nil {
		return nil, fmt.Errorf("create v_state tensor: %w", err)
	}
	defer vsTensor.Destroy()

	srcIDsTensor, err := ort.NewTensor(ort.NewShape(1, int64(len(srcIDs))), srcIDs)
	if err != nil {
		return nil, fmt.Errorf("create src_ids tensor: %w", err)
	}
	defer srcIDsTensor.Destroy()

	for range maxLen {
		tokenTensor.GetData()[0] = currentToken

		outputs := []ort.Value{nil, nil, nil, nil, nil}
		err = session.Run(
			[]ort.Value{tokenTensor, allK, allV, angleTensor, ssmTensor, ksTensor, vsTensor, srcIDsTensor},
			outputs,
		)
		if err != nil {
			return nil, fmt.Errorf("decoder run: %w", err)
		}

		logitsTensor, ok := outputs[0].(*ort.Tensor[float32])
		if !ok {
			return nil, fmt.Errorf("unexpected logits type")
		}
		logitsData := logitsTensor.GetData()
		nextID := argmax(logitsData)
		logitsTensor.Destroy()

		// Copy updated states back into reusable tensors.
		angleOut, ok := outputs[1].(*ort.Tensor[float32])
		if !ok {
			return nil, fmt.Errorf("unexpected angle_out type")
		}
		copy(angleTensor.GetData(), angleOut.GetData())
		angleOut.Destroy()

		ssmOut, ok := outputs[2].(*ort.Tensor[float32])
		if !ok {
			return nil, fmt.Errorf("unexpected ssm_out type")
		}
		copy(ssmTensor.GetData(), ssmOut.GetData())
		ssmOut.Destroy()

		ksOut, ok := outputs[3].(*ort.Tensor[float32])
		if !ok {
			return nil, fmt.Errorf("unexpected k_state_out type")
		}
		copy(ksTensor.GetData(), ksOut.GetData())
		ksOut.Destroy()

		vsOut, ok := outputs[4].(*ort.Tensor[float32])
		if !ok {
			return nil, fmt.Errorf("unexpected v_state_out type")
		}
		copy(vsTensor.GetData(), vsOut.GetData())
		vsOut.Destroy()

		if debugSteps > 0 && len(tgtIDs) < debugSteps {
			fmt.Fprintf(os.Stderr, "  GO step %3d: id=%5d\n", len(tgtIDs), nextID)
		}

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


// topKIndices returns the indices of the k largest values in data.
func topKIndices(data []float64, k int) []int {
	type iv struct {
		idx int
		val float64
	}
	items := make([]iv, len(data))
	for i, v := range data {
		items[i] = iv{i, v}
	}
	sort.Slice(items, func(a, b int) bool { return items[a].val > items[b].val })
	if k > len(items) {
		k = len(items)
	}
	out := make([]int, k)
	for i := 0; i < k; i++ {
		out[i] = items[i].idx
	}
	return out
}

type beamState struct {
	score    float64
	ids      []int64
	hData    []float32
	convData []float32
}

// beamDecode runs beam search decoding. Each beam runs a separate ONNX
// decoder step since the exported model has batch=1.
func beamDecode(
	session *ort.DynamicAdvancedSession,
	allK, allV *ort.Tensor[float32],
	srcIDs []int64,
	bosID, eosID int64,
	maxLen, nLayers, dInner, dState, dConv, beamWidth int,
	lengthPenalty float64,
) ([]int64, error) {
	hSize := nLayers * dInner * dState
	convSize := nLayers * dInner * (dConv - 1)

	// Reusable tensors for single-beam decoder steps.
	tokenData := []int64{bosID}
	tokenTensor, err := ort.NewTensor(ort.NewShape(1, 1), tokenData)
	if err != nil {
		return nil, fmt.Errorf("create token tensor: %w", err)
	}
	defer tokenTensor.Destroy()

	hBuf := make([]float32, hSize)
	hTensor, err := ort.NewTensor(
		ort.NewShape(int64(nLayers), int64(dInner), int64(dState)), hBuf,
	)
	if err != nil {
		return nil, fmt.Errorf("create h tensor: %w", err)
	}
	defer hTensor.Destroy()

	convBuf := make([]float32, convSize)
	convTensor, err := ort.NewTensor(
		ort.NewShape(int64(nLayers), int64(dInner), int64(dConv-1)), convBuf,
	)
	if err != nil {
		return nil, fmt.Errorf("create conv tensor: %w", err)
	}
	defer convTensor.Destroy()

	srcIDsTensor, err := ort.NewTensor(ort.NewShape(1, int64(len(srcIDs))), srcIDs)
	if err != nil {
		return nil, fmt.Errorf("create src_ids tensor: %w", err)
	}
	defer convTensor.Destroy()

	// Start with a single beam.
	active := []*beamState{{
		score:    0,
		ids:      nil,
		hData:    make([]float32, hSize),
		convData: make([]float32, convSize),
	}}
	var completed []*beamState

	type candidate struct {
		score     float64
		parentIdx int
		tokenID   int
	}

	for step := range maxLen {
		var candidates []candidate

		for bi, b := range active {
			// Load this beam's state into the reusable tensors.
			copy(hTensor.GetData(), b.hData)
			copy(convTensor.GetData(), b.convData)

			// Token: BOS on first step, last token otherwise.
			if step == 0 {
				tokenTensor.GetData()[0] = bosID
			} else {
				tokenTensor.GetData()[0] = b.ids[len(b.ids)-1]
			}

			// Run one decoder step.
			outputs := []ort.Value{nil, nil, nil}
			if err := session.Run(
				[]ort.Value{tokenTensor, allK, allV, hTensor, convTensor, srcIDsTensor},
				outputs,
			); err != nil {
				return nil, fmt.Errorf("decoder run (beam %d, step %d): %w", bi, step, err)
			}

			// Extract log-probs (already log-softmax from copy mechanism).
			logProbsTensor, ok := outputs[0].(*ort.Tensor[float32])
			if !ok {
				return nil, fmt.Errorf("unexpected log_probs type")
			}
			logProbsRaw := logProbsTensor.GetData()
			logProbs := make([]float64, len(logProbsRaw))
			for lpi, lpv := range logProbsRaw {
				logProbs[lpi] = float64(lpv)
			}
			logProbsTensor.Destroy()

			// Save updated state back to beam.
			hOut, ok := outputs[1].(*ort.Tensor[float32])
			if !ok {
				return nil, fmt.Errorf("unexpected h_out type")
			}
			copy(b.hData, hOut.GetData())
			hOut.Destroy()

			convOut, ok := outputs[2].(*ort.Tensor[float32])
			if !ok {
				return nil, fmt.Errorf("unexpected conv_out type")
			}
			copy(b.convData, convOut.GetData())
			convOut.Destroy()

			// Top-K tokens from this beam.
			topK := topKIndices(logProbs, beamWidth)
			for _, idx := range topK {
				candidates = append(candidates, candidate{
					score:     b.score + logProbs[idx],
					parentIdx: bi,
					tokenID:   idx,
				})
			}
		}

		// Sort candidates by score (descending).
		sort.Slice(candidates, func(i, j int) bool {
			return candidates[i].score > candidates[j].score
		})

		// Select top beamWidth non-EOS candidates.
		var newActive []*beamState
		for _, c := range candidates {
			parent := active[c.parentIdx]
			if int64(c.tokenID) == eosID {
				completed = append(completed, &beamState{
					score: c.score,
					ids:   append([]int64(nil), parent.ids...),
				})
			} else if len(newActive) < beamWidth {
				newActive = append(newActive, &beamState{
					score:    c.score,
					ids:      append(append([]int64(nil), parent.ids...), int64(c.tokenID)),
					hData:    append([]float32(nil), parent.hData...),
					convData: append([]float32(nil), parent.convData...),
				})
			}
		}

		active = newActive
		if len(active) == 0 {
			break
		}

		// Early stop: best completed raw score >= best active raw score.
		// Active scores can only decrease (log-probs are non-positive).
		if len(completed) > 0 {
			bestCompleted := completed[0].score
			for _, c := range completed[1:] {
				if c.score > bestCompleted {
					bestCompleted = c.score
				}
			}
			if bestCompleted >= active[0].score {
				break
			}
		}
	}

	// Add remaining active beams.
	for _, b := range active {
		completed = append(completed, b)
	}

	if len(completed) == 0 {
		return nil, nil
	}

	// Return best by length-normalized score.
	var bestIdx int
	bestScore := math.Inf(-1)
	for i, c := range completed {
		length := float64(len(c.ids))
		if length == 0 {
			length = 1
		}
		normed := c.score / math.Pow(length, lengthPenalty)
		if normed > bestScore {
			bestScore = normed
			bestIdx = i
		}
	}
	return completed[bestIdx].ids, nil
}

// levenshtein computes the edit distance between two string slices.
// Used for WER when called with words, CER when called with single-char strings.
func levenshtein(a, b []string) int {
	if len(a) < len(b) {
		return levenshtein(b, a)
	}
	if len(b) == 0 {
		return len(a)
	}
	prev := make([]int, len(b)+1)
	for i := range prev {
		prev[i] = i
	}
	for i, ca := range a {
		curr := make([]int, len(b)+1)
		curr[0] = i + 1
		for j, cb := range b {
			del := prev[j+1] + 1
			ins := curr[j] + 1
			sub := prev[j]
			if ca != cb {
				sub++
			}
			curr[j+1] = min(del, min(ins, sub))
		}
		prev = curr
	}
	return prev[len(b)]
}

// toChars splits a string into individual character strings for CER computation.
func toChars(s string) []string {
	runes := []rune(s)
	out := make([]string, len(runes))
	for i, r := range runes {
		out[i] = string(r)
	}
	return out
}

// charWeightedWER computes word-level edit distance but returns the total
// character count of the affected words rather than the word count.
// This prevents a single large block deletion from counting as "1 word edit."
func charWeightedWER(pred, ref []string) int {
	if len(ref) < len(pred) {
		// Ensure ref is the longer side for consistent weighting.
		pred, ref = ref, pred
	}
	if len(ref) == 0 {
		total := 0
		for _, w := range pred {
			total += len([]rune(w))
		}
		return total
	}

	// Standard Levenshtein DP, but track which words are edited.
	n, m := len(ref), len(pred)
	// cost[j] = character-weighted edit cost to transform pred[:j] into ref[:i]
	prev := make([]int, m+1)
	for j := 1; j <= m; j++ {
		prev[j] = prev[j-1] + len([]rune(pred[j-1]))
	}
	for i := 1; i <= n; i++ {
		curr := make([]int, m+1)
		curr[0] = prev[0] + len([]rune(ref[i-1]))
		for j := 1; j <= m; j++ {
			if ref[i-1] == pred[j-1] {
				curr[j] = prev[j-1] // no edit
			} else {
				// Cost of substitution: chars in both words
				subCost := prev[j-1] + max(len([]rune(ref[i-1])), len([]rune(pred[j-1])))
				// Cost of deletion (skip ref word): chars in ref word
				delCost := prev[j] + len([]rune(ref[i-1]))
				// Cost of insertion (skip pred word): chars in pred word
				insCost := curr[j-1] + len([]rune(pred[j-1]))
				curr[j] = min(subCost, min(delCost, insCost))
			}
		}
		prev = curr
	}
	return prev[m]
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

// xmlSemanticallyEqual parses both XML strings, flattens each into a canonical
// sequence of (element, text) tokens with normalized whitespace, and compares.
// This treats CDATA vs plain text as equivalent and ignores insignificant whitespace.
func xmlSemanticallyEqual(a, b string) bool {
	af := flattenXML(a)
	bf := flattenXML(b)
	if af == nil || bf == nil {
		return false
	}
	if len(af) != len(bf) {
		return false
	}
	for i := range af {
		if af[i] != bf[i] {
			return false
		}
	}
	return true
}

type xmlToken struct {
	kind string // "start", "end", "text"
	val  string
}

var wsNormRe = regexp.MustCompile(`\s+`)

func flattenXML(s string) []xmlToken {
	d := xml.NewDecoder(strings.NewReader(s))
	var tokens []xmlToken
	for {
		tok, err := d.Token()
		if err != nil {
			if err.Error() == "EOF" {
				return tokens
			}
			return nil // parse error
		}
		switch t := tok.(type) {
		case xml.StartElement:
			tokens = append(tokens, xmlToken{"start", t.Name.Local})
		case xml.EndElement:
			tokens = append(tokens, xmlToken{"end", t.Name.Local})
		case xml.CharData:
			text := wsNormRe.ReplaceAllString(strings.TrimSpace(string(t)), " ")
			if text != "" {
				tokens = append(tokens, xmlToken{"text", text})
			}
		}
	}
}
