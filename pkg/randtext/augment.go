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

package randtext

import (
	"bufio"
	"math/rand/v2"
	"os"
	"strings"
	"sync"
	"unicode"
)

var (
	dictWords []string
	dictOnce  sync.Once
)

const dictPath = "/usr/share/dict/american-english"

func loadDict() {
	f, err := os.Open(dictPath)
	if err != nil {
		// Fallback: combine existing word pools.
		dictWords = make([]string, 0, len(nouns)+len(verbs)+len(adjectives)+len(adverbs))
		dictWords = append(dictWords, nouns...)
		dictWords = append(dictWords, verbs...)
		dictWords = append(dictWords, adjectives...)
		dictWords = append(dictWords, adverbs...)
		return
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		w := strings.TrimSpace(scanner.Text())
		if len(w) < 3 || strings.ContainsAny(w, "''") {
			continue
		}
		dictWords = append(dictWords, strings.ToLower(w))
	}
	if len(dictWords) == 0 {
		dictWords = append(dictWords, nouns...)
		dictWords = append(dictWords, verbs...)
	}
}

// XML-special characters that trigger CDATA wrapping.
var specialChars = []string{"<", ">", "&", "'", "\""}

// DictWord returns a random word from the system dictionary (~100k words).
func DictWord(rng *rand.Rand) string {
	dictOnce.Do(loadDict)
	return dictWords[rng.IntN(len(dictWords))]
}

// InjectSpecialChars inserts random bare special characters at word boundaries.
// prob controls the likelihood of injection at each boundary.
func InjectSpecialChars(rng *rand.Rand, text string, prob float64) string {
	words := strings.Fields(text)
	if len(words) == 0 {
		return text
	}
	var b strings.Builder
	b.Grow(len(text) + len(text)/4)
	for i, w := range words {
		if i > 0 {
			b.WriteByte(' ')
		}
		b.WriteString(w)
		if rng.Float64() < prob {
			ch := specialChars[rng.IntN(len(specialChars))]
			switch rng.IntN(4) {
			case 0:
				b.WriteString(ch) // glued after: word<
			case 1:
				b.WriteByte(' ')
				b.WriteString(ch) // spaced: word <
			case 2:
				b.WriteString(ch)
				b.WriteByte(' ') // leading next: word< next
			case 3:
				b.WriteByte(' ')
				b.WriteString(ch)
				b.WriteByte(' ') // isolated: word < next
			}
		}
	}
	return b.String()
}

// DictSentence builds a sentence from random dictionary words with special
// character injection.
func DictSentence(rng *rand.Rand) string {
	n := 5 + rng.IntN(16) // 5-20 words
	words := make([]string, n)
	for i := range words {
		words[i] = DictWord(rng)
	}
	if len(words[0]) > 0 {
		r := []rune(words[0])
		r[0] = unicode.ToUpper(r[0])
		words[0] = string(r)
	}
	text := strings.Join(words, " ") + "."
	return InjectSpecialChars(rng, text, 0.40)
}

// ShuffleWords randomizes word order in text and injects special characters.
func ShuffleWords(rng *rand.Rand, text string) string {
	words := strings.Fields(text)
	rng.Shuffle(len(words), func(i, j int) {
		words[i], words[j] = words[j], words[i]
	})
	return InjectSpecialChars(rng, strings.Join(words, " "), 0.40)
}

// AugmentedSentence returns either a dictionary-word sentence or a shuffled
// normal sentence, both with random special character injection.
func AugmentedSentence(rng *rand.Rand) string {
	if rng.Float64() < 0.5 {
		return DictSentence(rng)
	}
	return ShuffleWords(rng, Sentence(rng))
}

// AugmentedParagraph generates n augmented sentences.
func AugmentedParagraph(rng *rand.Rand, n int) string {
	parts := make([]string, n)
	for i := range parts {
		parts[i] = AugmentedSentence(rng)
	}
	return strings.Join(parts, " ")
}

// AugmentedThought generates an augmented thought block of min-max sentences.
func AugmentedThought(rng *rand.Rand, min, max int) string {
	n := min + rng.IntN(max-min+1)
	return AugmentedParagraph(rng, n)
}

// AugmentedMemoryEntry returns an augmented memory entry — either random
// dictionary words or a shuffled normal memory entry, with elevated special
// character injection.
func AugmentedMemoryEntry(rng *rand.Rand) string {
	if rng.Float64() < 0.5 {
		n := 4 + rng.IntN(10) // 4-13 words
		words := make([]string, n)
		for i := range words {
			words[i] = DictWord(rng)
		}
		return InjectSpecialChars(rng, strings.Join(words, " "), 0.50)
	}
	return ShuffleWords(rng, MemoryEntry(rng))
}

// AugmentedMarkdownHeader returns a 1-3 word header from dictionary words.
func AugmentedMarkdownHeader(rng *rand.Rand) string {
	n := 1 + rng.IntN(3)
	words := make([]string, n)
	for i := range words {
		w := DictWord(rng)
		r := []rune(w)
		r[0] = unicode.ToUpper(r[0])
		words[i] = string(r)
	}
	return strings.Join(words, " ")
}

// AugmentedIntroSentence returns an augmented intro sentence.
func AugmentedIntroSentence(rng *rand.Rand) string {
	return DictSentence(rng)
}

// AugmentedListItem returns an augmented list item.
func AugmentedListItem(rng *rand.Rand) string {
	n := 3 + rng.IntN(8) // 3-10 words
	words := make([]string, n)
	for i := range words {
		words[i] = DictWord(rng)
	}
	r := []rune(words[0])
	r[0] = unicode.ToUpper(r[0])
	words[0] = string(r)
	return InjectSpecialChars(rng, strings.Join(words, " "), 0.40)
}

// AugmentedTableCell returns an augmented table cell value.
func AugmentedTableCell(rng *rand.Rand) string {
	w := DictWord(rng)
	if rng.Float64() < 0.3 {
		return InjectSpecialChars(rng, w, 0.5)
	}
	return w
}
