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

// Package sentencepiece implements a SentencePiece BPE tokenizer in pure Go.
//
// It reads standard SentencePiece .model files (protobuf format) and provides
// Encode (text to token IDs) and Decode (token IDs to text) methods.
// All piece types are supported: NORMAL, UNKNOWN, CONTROL, USER_DEFINED, BYTE, UNUSED.
package sentencepiece

import (
	"fmt"
	"io"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"

	"google.golang.org/protobuf/proto"
)

var wsRun = regexp.MustCompile(`\s+`)

const separator = "▁" // U+2581, SentencePiece whitespace marker

// Processor tokenizes and detokenizes text using a SentencePiece BPE model.
type Processor struct {
	pieces []*ModelProto_SentencePiece // all pieces, indexed by ID

	pieceToID map[string]int // piece string → ID (NORMAL + USER_DEFINED + BYTE)
	bytePiece [256]int       // byte value → piece ID (for byte fallback)

	userDefined []string // sorted longest-first for greedy prefix matching

	unkID         int
	bosID         int
	eosID         int
	padID         int
	byteFallback  bool
	addDummyPfx   bool
	removeExtraWS bool
	maxPieceLen   int
}

// Load reads a SentencePiece .model file from disk.
func Load(path string) (*Processor, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return LoadFrom(f)
}

// LoadFrom reads a SentencePiece model from a reader.
func LoadFrom(r io.Reader) (*Processor, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}
	var mp ModelProto
	if err := proto.Unmarshal(data, &mp); err != nil {
		return nil, fmt.Errorf("unmarshal sentencepiece model: %w", err)
	}

	p := &Processor{
		pieces:    mp.GetPieces(),
		pieceToID: make(map[string]int),
		unkID:     -1,
		bosID:     -1,
		eosID:     -1,
		padID:     -1,
	}

	if ts := mp.GetTrainerSpec(); ts != nil {
		p.byteFallback = ts.GetByteFallback()
	}
	if ns := mp.GetNormalizerSpec(); ns != nil {
		p.addDummyPfx = ns.GetAddDummyPrefix()
		p.removeExtraWS = ns.GetRemoveExtraWhitespaces()
	}

	for i, piece := range p.pieces {
		s := piece.GetPiece()
		t := piece.GetType()

		switch t {
		case ModelProto_SentencePiece_NORMAL:
			p.pieceToID[s] = i
		case ModelProto_SentencePiece_USER_DEFINED:
			p.pieceToID[s] = i
			p.userDefined = append(p.userDefined, s)
		case ModelProto_SentencePiece_UNKNOWN:
			p.unkID = i
			p.pieceToID[s] = i
		case ModelProto_SentencePiece_CONTROL:
			// Identify special IDs by piece string.
			switch s {
			case "<s>":
				p.bosID = i
			case "</s>":
				p.eosID = i
			case "<pad>":
				p.padID = i
			}
		case ModelProto_SentencePiece_BYTE:
			bv := parseByteHex(s)
			if bv >= 0 && bv < 256 {
				p.bytePiece[bv] = i
			}
			p.pieceToID[s] = i
		case ModelProto_SentencePiece_UNUSED:
			// ignored
		}

		if len(s) > p.maxPieceLen {
			p.maxPieceLen = len(s)
		}
	}

	// Sort user-defined symbols longest-first for greedy matching.
	sort.Slice(p.userDefined, func(i, j int) bool {
		return len(p.userDefined[i]) > len(p.userDefined[j])
	})

	if p.unkID < 0 {
		return nil, fmt.Errorf("sentencepiece model has no <unk> piece")
	}

	return p, nil
}

// BOS returns the beginning-of-sentence token ID, or -1 if not defined.
func (p *Processor) BOS() int { return p.bosID }

// EOS returns the end-of-sentence token ID, or -1 if not defined.
func (p *Processor) EOS() int { return p.eosID }

// PAD returns the padding token ID, or -1 if not defined.
func (p *Processor) PAD() int { return p.padID }

// UNK returns the unknown token ID.
func (p *Processor) UNK() int { return p.unkID }

// VocabSize returns the total number of pieces in the model.
func (p *Processor) VocabSize() int { return len(p.pieces) }

// Encode tokenizes text into a sequence of token IDs.
// If addBOS/addEOS are true, the respective special tokens are prepended/appended.
func (p *Processor) Encode(text string, addBOS, addEOS bool) []int {
	// Normalize whitespace: convert \n, \t, \r etc to spaces (matches C++ NFKC).
	text = wsRun.ReplaceAllString(text, " ")
	if p.removeExtraWS {
		text = strings.TrimSpace(text)
	}

	// Replace spaces with separator.
	text = strings.ReplaceAll(text, " ", separator)

	// Add dummy prefix if the model was trained with it.
	if p.addDummyPfx && len(text) > 0 {
		text = separator + text
	}

	// Split into initial symbols: user-defined pieces are kept whole,
	// everything else is split into individual runes.
	type symbol struct {
		text    string
		noMerge bool // user-defined symbols can't be merged
	}
	var symbols []symbol

	for len(text) > 0 {
		// Try user-defined symbols (longest match first).
		matched := false
		for _, ud := range p.userDefined {
			if strings.HasPrefix(text, ud) {
				symbols = append(symbols, symbol{text: ud, noMerge: true})
				text = text[len(ud):]
				matched = true
				break
			}
		}
		if matched {
			continue
		}

		// Otherwise take one rune.
		_, rlen := utf8.DecodeRuneInString(text)
		if rlen == 0 {
			rlen = 1 // shouldn't happen, but be safe
		}
		symbols = append(symbols, symbol{text: text[:rlen]})
		text = text[rlen:]
	}

	if len(symbols) == 0 {
		return nil
	}

	// BPE merge loop: repeatedly find the highest-scoring adjacent pair and merge.
	for {
		bestScore := float32(-1e18)
		bestIdx := -1

		for i := 0; i < len(symbols)-1; i++ {
			if symbols[i].noMerge || symbols[i+1].noMerge {
				continue
			}
			merged := symbols[i].text + symbols[i+1].text
			if id, ok := p.pieceToID[merged]; ok {
				score := p.pieces[id].GetScore()
				if score > bestScore {
					bestScore = score
					bestIdx = i
				}
			}
		}

		if bestIdx < 0 {
			break
		}

		// Merge symbols[bestIdx] and symbols[bestIdx+1].
		symbols[bestIdx].text = symbols[bestIdx].text + symbols[bestIdx+1].text
		symbols = append(symbols[:bestIdx+1], symbols[bestIdx+2:]...)
	}

	// Convert symbols to IDs.
	var ids []int
	if addBOS && p.bosID >= 0 {
		ids = append(ids, p.bosID)
	}

	for _, sym := range symbols {
		if id, ok := p.pieceToID[sym.text]; ok {
			ids = append(ids, id)
		} else if p.byteFallback {
			// Decompose unknown symbol into byte pieces.
			for i := 0; i < len(sym.text); i++ {
				ids = append(ids, p.bytePiece[sym.text[i]])
			}
		} else {
			ids = append(ids, p.unkID)
		}
	}

	if addEOS && p.eosID >= 0 {
		ids = append(ids, p.eosID)
	}

	return ids
}

// Decode converts a sequence of token IDs back into text.
func (p *Processor) Decode(ids []int) string {
	var sb strings.Builder

	for i := 0; i < len(ids); {
		id := ids[i]

		// Check for byte piece runs: consecutive byte-type pieces get
		// decoded together as raw UTF-8 bytes.
		if p.isByte(id) {
			var buf []byte
			for i < len(ids) && p.isByte(ids[i]) {
				bv := parseByteHex(p.pieces[ids[i]].GetPiece())
				if bv >= 0 {
					buf = append(buf, byte(bv))
				}
				i++
			}
			sb.Write(buf)
			continue
		}

		// Skip control tokens (BOS, EOS, PAD).
		if p.isControl(id) {
			i++
			continue
		}

		// Unknown token.
		if id == p.unkID {
			sb.WriteString("\u2047") // double question mark
			i++
			continue
		}

		// Normal, USER_DEFINED, or anything else with a piece string.
		if id >= 0 && id < len(p.pieces) {
			piece := p.pieces[id].GetPiece()
			sb.WriteString(strings.ReplaceAll(piece, separator, " "))
		}
		i++
	}

	return sb.String()
}

func (p *Processor) isByte(id int) bool {
	if id < 0 || id >= len(p.pieces) {
		return false
	}
	return p.pieces[id].GetType() == ModelProto_SentencePiece_BYTE
}

func (p *Processor) isControl(id int) bool {
	if id < 0 || id >= len(p.pieces) {
		return false
	}
	return p.pieces[id].GetType() == ModelProto_SentencePiece_CONTROL
}

// parseByteHex converts "<0xAB>" to 0xAB.
func parseByteHex(s string) int {
	if !strings.HasPrefix(s, "<0x") || !strings.HasSuffix(s, ">") {
		return -1
	}
	hex := s[3 : len(s)-1]
	v, err := strconv.ParseUint(hex, 16, 8)
	if err != nil {
		return -1
	}
	return int(v)
}
