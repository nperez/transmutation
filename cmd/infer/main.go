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
	flag.IntVar(&dState, "d-state", 16, "Mamba d_state")
	flag.IntVar(&dConv, "d-conv", 4, "Mamba d_conv")
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
	if beamWidth > 1 {
		fmt.Printf("Beam search: width=%d, length_penalty=%.2f\n", beamWidth, lengthPenalty)
	}

	// Read and process samples from stdin.
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	wsNorm := regexp.MustCompile(`\s+`)
	exactCount := 0
	semanticCount := 0
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

		// Decode (greedy or beam search).
		debug := 0
		if total == 1 && debugSteps > 0 {
			debug = debugSteps
		}

		var predIDs []int64
		if beamWidth > 1 {
			predIDs, err = beamDecode(decSession, allK, allV, bosID, eosID,
				maxTgtLen, nLayers, dInner, dState, dConv, beamWidth, lengthPenalty)
		} else {
			predIDs, err = greedyDecode(decSession, allK, allV, bosID, eosID,
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

		tag := "FAIL"
		if exact {
			tag = "EXACT"
		} else if semantic {
			tag = "SEMANTIC"
		} else if xmlOK {
			tag = "XML_OK"
		}

		fmt.Printf("=== Sample %d [%s] %.2fs, %d tokens ===\n", total, tag, elapsed.Seconds(), len(predIDs))
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

	fmt.Printf("===== %d samples: exact=%d semantic=%d xml_ok=%d fail=%d =====\n",
		total, exactCount, semanticCount, xmlOKCount-exactCount-semanticCount, total-xmlOKCount)
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

func logSoftmax(logits []float32) []float64 {
	max := float64(logits[0])
	for _, v := range logits[1:] {
		if float64(v) > max {
			max = float64(v)
		}
	}
	var sumExp float64
	for _, v := range logits {
		sumExp += math.Exp(float64(v) - max)
	}
	logSumExp := max + math.Log(sumExp)

	out := make([]float64, len(logits))
	for i, v := range logits {
		out[i] = float64(v) - logSumExp
	}
	return out
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
				[]ort.Value{tokenTensor, allK, allV, hTensor, convTensor},
				outputs,
			); err != nil {
				return nil, fmt.Errorf("decoder run (beam %d, step %d): %w", bi, step, err)
			}

			// Extract logits and compute log-softmax.
			logitsTensor, ok := outputs[0].(*ort.Tensor[float32])
			if !ok {
				return nil, fmt.Errorf("unexpected logits type")
			}
			logProbs := logSoftmax(logitsTensor.GetData())
			logitsTensor.Destroy()

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
