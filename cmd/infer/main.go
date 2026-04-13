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

// Package main provides ONNX-based diffusion inference for the transmutation model.
// Reads JSONL from stdin, runs iterative denoising, compares output to targets.
package main

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"encoding/xml"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"regexp"
	"runtime"
	"strings"
	"time"

	"nickandperla.net/transmutation/pkg/sentencepiece"

	ort "github.com/yalue/onnxruntime_go"
)

// LENGTH_BUCKETS must match model.py LENGTH_BUCKETS.
var LENGTH_BUCKETS = []int{64, 128, 256, 384, 512, 768, 1024, 1536}

type Record struct {
	Input  string `json:"input"`
	Target string `json:"target"`
}

func main() {
	var (
		modelPath    string
		lengthPath   string
		embDownPath  string
		embUpPath    string
		tokenizPath  string
		ortLibPath   string
		nSamples     int
		maxSrcLen    int
		denoiseSteps int
		dModel       int
		embRank      int
	)
	flag.StringVar(&modelPath, "model", "models/onnx/diffusion.onnx", "path to denoiser ONNX")
	flag.StringVar(&lengthPath, "length-model", "models/onnx/length_predictor.onnx", "path to length predictor ONNX")
	flag.StringVar(&embDownPath, "emb-down", "models/onnx/emb_down.npy", "path to embedding down matrix (vocab, rank)")
	flag.StringVar(&embUpPath, "emb-up", "models/onnx/emb_up.npy", "path to embedding up matrix (d_model, rank)")
	flag.StringVar(&tokenizPath, "tokenizer", "models/tokenizer.model", "path to sentencepiece model")
	flag.StringVar(&ortLibPath, "ort-lib", "", "path to onnxruntime shared library")
	flag.IntVar(&nSamples, "n", 10, "number of samples to run")
	flag.IntVar(&maxSrcLen, "max-src-len", 1152, "max source token length")
	flag.IntVar(&denoiseSteps, "denoise-steps", 4, "number of denoising steps")
	flag.IntVar(&dModel, "d-model", 512, "model dimension")
	flag.IntVar(&embRank, "emb-rank", 128, "embedding factorization rank")
	flag.Parse()

	// Initialize tokenizer.
	sp, err := sentencepiece.Load(tokenizPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to load tokenizer: %v\n", err)
		os.Exit(1)
	}
	eosID := int64(sp.EOS())
	padID := int64(sp.PAD())
	vocabSize := sp.VocabSize()
	fmt.Printf("Tokenizer: vocab=%d eos=%d pad=%d\n", vocabSize, eosID, padID)

	// Initialize ONNX Runtime.
	if ortLibPath != "" {
		ort.SetSharedLibraryPath(ortLibPath)
	}
	if err := ort.InitializeEnvironment(); err != nil {
		fmt.Fprintf(os.Stderr, "failed to initialize onnxruntime: %v\n", err)
		os.Exit(1)
	}
	defer ort.DestroyEnvironment()

	sessionOpts, err := ort.NewSessionOptions()
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to create session options: %v\n", err)
		os.Exit(1)
	}
	defer sessionOpts.Destroy()
	nThreads := runtime.NumCPU()
	sessionOpts.SetIntraOpNumThreads(nThreads)
	fmt.Printf("ORT threads: %d\n", nThreads)

	// Create sessions.
	denoiserSession, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{"src_ids", "noised_emb", "timestep"},
		[]string{"pred_emb"},
		sessionOpts,
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to create denoiser session: %v\n", err)
		os.Exit(1)
	}
	defer denoiserSession.Destroy()

	lengthSession, err := ort.NewDynamicAdvancedSession(
		lengthPath,
		[]string{"src_ids"},
		[]string{"length_logits"},
		sessionOpts,
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to create length session: %v\n", err)
		os.Exit(1)
	}
	defer lengthSession.Destroy()

	// Load embedding tables for discretization.
	embDown, err := loadNpyFloat32(embDownPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to load emb_down.npy: %v\n", err)
		os.Exit(1)
	}
	embUp, err := loadNpyFloat32(embUpPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to load emb_up.npy: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Embeddings: down=%d up=%d\n", len(embDown), len(embUp))

	// Read records from stdin.
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 10*1024*1024), 10*1024*1024)
	var records []Record
	for scanner.Scan() {
		if len(records) >= nSamples {
			break
		}
		line := strings.TrimSpace(scanner.Text())
		if line == "" || line[0] != '{' {
			continue
		}
		var rec Record
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			continue
		}
		ids := sp.Encode(rec.Input, false, false)
		if len(ids) > maxSrcLen {
			continue
		}
		records = append(records, rec)
	}
	fmt.Printf("\nLoaded %d records\n\n", len(records))

	// Process each record.
	exactCount := 0
	semanticCount := 0
	xmlOKCount := 0

	for i, rec := range records {
		srcIDs := sp.Encode(rec.Input, false, false)
		srcLen := len(srcIDs)
		t0 := time.Now()

		// 1. Predict output length bucket.
		srcTensor, _ := ort.NewTensor(ort.NewShape(1, int64(srcLen)), toInt64(srcIDs))
		lengthOut, _ := ort.NewEmptyTensor[float32](ort.NewShape(1, int64(len(LENGTH_BUCKETS))))
		err = lengthSession.Run([]ort.Value{srcTensor}, []ort.Value{lengthOut})
		if err != nil {
			fmt.Fprintf(os.Stderr, "length prediction failed: %v\n", err)
			continue
		}
		lengthLogits := lengthOut.GetData()
		bucketIdx := argmax(lengthLogits)
		lengthOut.Destroy()
		// One bucket up for safety.
		if bucketIdx < len(LENGTH_BUCKETS)-1 {
			bucketIdx++
		}
		tgtLen := LENGTH_BUCKETS[bucketIdx]

		// 2. Initialize from noise.
		x := make([]float32, tgtLen*dModel)
		for j := range x {
			x[j] = float32(rand.NormFloat64())
		}

		// 3. Denoise loop.
		for step := 0; step < denoiseSteps; step++ {
			t := 1.0 - float64(step)/float64(denoiseSteps)

			noisedTensor, _ := ort.NewTensor(ort.NewShape(1, int64(tgtLen), int64(dModel)), x)
			tsTensor, _ := ort.NewTensor(ort.NewShape(1), []float32{float32(t)})
			predOut, _ := ort.NewEmptyTensor[float32](ort.NewShape(1, int64(tgtLen), int64(dModel)))

			err = denoiserSession.Run(
				[]ort.Value{srcTensor, noisedTensor, tsTensor},
				[]ort.Value{predOut},
			)
			if err != nil {
				fmt.Fprintf(os.Stderr, "denoise step %d failed: %v\n", step, err)
				break
			}
			predX0 := predOut.GetData()
			noisedTensor.Destroy()
			tsTensor.Destroy()

			if step < denoiseSteps-1 {
				tNext := 1.0 - float64(step+1)/float64(denoiseSteps)
				for j := range x {
					x[j] = float32(1.0-tNext)*predX0[j] + float32(tNext)*x[j]
				}
			} else {
				copy(x, predX0)
			}
			predOut.Destroy()
		}

		// 4. Discretize: x @ embUp.T -> (tgtLen, rank) @ embDown.T -> argmax.
		tokenIDs := discretize(x, embDown, embUp, tgtLen, dModel, vocabSize, embRank)

		// 5. Trim at EOS/PAD.
		var trimmed []int
		for _, tid := range tokenIDs {
			if int64(tid) == eosID || int64(tid) == padID {
				break
			}
			trimmed = append(trimmed, tid)
		}

		elapsed := time.Since(t0)
		pred := sp.Decode(trimmed)
		target := rec.Target

		// Score.
		normPred := normalizeWS(pred)
		normTgt := normalizeWS(target)
		exact := normPred == normTgt

		xmlOK := isValidXML(pred)

		semantic := false
		if !exact && xmlOK {
			semantic = xmlSemanticallyEqual(pred, target)
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

		fmt.Printf("=== Sample %d [%s] %.2fs, %d tokens ===\n", i+1, tag, elapsed.Seconds(), len(trimmed))
		if exact || semantic {
			fmt.Printf("OUTPUT (matches target):\n%s\n\n", strings.TrimSpace(pred))
		} else {
			fmt.Printf("TARGET:\n%s\n\nOUTPUT:\n%s\n\n", strings.TrimSpace(target), strings.TrimSpace(pred))
		}

		srcTensor.Destroy()
	}

	total := len(records)
	fmt.Printf("===== %d samples: exact=%d semantic=%d xml_ok=%d fail=%d =====\n",
		total, exactCount, semanticCount, xmlOKCount-exactCount-semanticCount, total-xmlOKCount)
}

// discretize projects from d_model to rank via embUp, then scores against all
// vocab entries via embDown, returning the argmax token ID per position.
func discretize(predEmb, embDown, embUp []float32, tgtLen, dModel, vocabSize, rank int) []int {
	tokens := make([]int, tgtLen)
	for pos := range tgtLen {
		// Project from d_model to rank: pred[pos] @ embUp^T
		// embUp is (dModel, rank), stored row-major.
		projected := make([]float32, rank)
		for r := range rank {
			var sum float32
			for d := range dModel {
				sum += predEmb[pos*dModel+d] * embUp[d*rank+r]
			}
			projected[r] = sum
		}

		// Score against all vocab: dot(projected, embDown[v]) for each v.
		// embDown is (vocabSize, rank), stored row-major.
		bestID := 0
		bestScore := float32(-math.MaxFloat32)
		for v := range vocabSize {
			var score float32
			for r := range rank {
				score += projected[r] * embDown[v*rank+r]
			}
			if score > bestScore {
				bestScore = score
				bestID = v
			}
		}
		tokens[pos] = bestID
	}
	return tokens
}

func toInt64(ids []int) []int64 {
	out := make([]int64, len(ids))
	for i, v := range ids {
		out[i] = int64(v)
	}
	return out
}

func argmax(data []float32) int {
	best := 0
	for i := 1; i < len(data); i++ {
		if data[i] > data[best] {
			best = i
		}
	}
	return best
}

func normalizeWS(s string) string {
	s = strings.TrimSpace(s)
	re := regexp.MustCompile(`\s+`)
	return re.ReplaceAllString(s, " ")
}

func isValidXML(s string) bool {
	d := xml.NewDecoder(strings.NewReader(strings.TrimSpace(s)))
	for {
		_, err := d.Token()
		if err != nil {
			return err.Error() == "EOF"
		}
	}
}

func xmlSemanticallyEqual(a, b string) bool {
	af := flattenXML(strings.TrimSpace(a))
	bf := flattenXML(strings.TrimSpace(b))
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

func flattenXML(s string) []xmlToken {
	d := xml.NewDecoder(strings.NewReader(s))
	var tokens []xmlToken
	for {
		tok, err := d.Token()
		if err != nil {
			break
		}
		switch t := tok.(type) {
		case xml.StartElement:
			tokens = append(tokens, xmlToken{"start", t.Name.Local})
		case xml.EndElement:
			tokens = append(tokens, xmlToken{"end", t.Name.Local})
		case xml.CharData:
			text := normalizeWS(string(t))
			if text != "" {
				tokens = append(tokens, xmlToken{"text", text})
			}
		}
	}
	return tokens
}

// loadNpyFloat32 reads a numpy .npy file containing a float32 array.
// Supports numpy format 1.0 and 2.0.
func loadNpyFloat32(path string) ([]float32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	// Magic: \x93NUMPY
	magic := make([]byte, 6)
	if _, err := io.ReadFull(f, magic); err != nil {
		return nil, fmt.Errorf("reading magic: %w", err)
	}
	if magic[0] != 0x93 || string(magic[1:6]) != "NUMPY" {
		return nil, fmt.Errorf("not a numpy file: bad magic")
	}

	// Version (2 bytes)
	ver := make([]byte, 2)
	if _, err := io.ReadFull(f, ver); err != nil {
		return nil, fmt.Errorf("reading version: %w", err)
	}

	// Header length
	var headerLen int
	if ver[0] == 1 {
		var hl uint16
		if err := binary.Read(f, binary.LittleEndian, &hl); err != nil {
			return nil, fmt.Errorf("reading header length: %w", err)
		}
		headerLen = int(hl)
	} else {
		var hl uint32
		if err := binary.Read(f, binary.LittleEndian, &hl); err != nil {
			return nil, fmt.Errorf("reading header length v2: %w", err)
		}
		headerLen = int(hl)
	}

	// Read and verify header
	header := make([]byte, headerLen)
	if _, err := io.ReadFull(f, header); err != nil {
		return nil, fmt.Errorf("reading header: %w", err)
	}
	headerStr := string(header)
	if !strings.Contains(headerStr, "<f4") && !strings.Contains(headerStr, "float32") {
		return nil, fmt.Errorf("expected float32 array, got header: %s", headerStr)
	}

	// Read remaining data as float32 LE
	stat, err := f.Stat()
	if err != nil {
		return nil, err
	}
	pos, _ := f.Seek(0, io.SeekCurrent)
	dataBytes := stat.Size() - pos
	nFloats := int(dataBytes / 4)

	data := make([]float32, nFloats)
	if err := binary.Read(f, binary.LittleEndian, data); err != nil {
		return nil, fmt.Errorf("reading data: %w", err)
	}
	return data, nil
}
