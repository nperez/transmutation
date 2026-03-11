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
	"fmt"
	"math/rand/v2"
	"strings"
)

// Word returns a random word from the full vocabulary.
func Word(rng *rand.Rand) string {
	switch rng.IntN(10) {
	case 0, 1, 2, 3:
		return pick(rng, nouns)
	case 4, 5:
		return pick(rng, verbs)
	case 6, 7:
		return pick(rng, adjectives)
	case 8:
		return pick(rng, adverbs)
	default:
		return pick(rng, nouns)
	}
}

// Noun returns a random noun.
func Noun(rng *rand.Rand) string { return pick(rng, nouns) }

// Verb returns a random verb.
func Verb(rng *rand.Rand) string { return pick(rng, verbs) }

// Adjective returns a random adjective.
func Adjective(rng *rand.Rand) string { return pick(rng, adjectives) }

// TechName returns a random technology/product name.
func TechName(rng *rand.Rand) string { return pick(rng, techNames) }

// Quantity returns a random numeric quantity with optional units.
func Quantity(rng *rand.Rand) string { return pick(rng, quantities) }

// Topic generates a random 2-4 word topic phrase like "distributed cache
// partitioning" or "user authentication flow".
func Topic(rng *rand.Rand) string {
	n := 2 + rng.IntN(3) // 2-4 words
	parts := make([]string, n)
	for i := range parts {
		if i == 0 && rng.Float64() < 0.4 {
			parts[i] = pick(rng, adjectives)
		} else {
			parts[i] = pick(rng, nouns)
		}
	}
	return strings.Join(parts, " ")
}

// NounPhrase generates a random noun phrase like "the encrypted payload"
// or "a distributed cache".
func NounPhrase(rng *rand.Rand) string {
	switch rng.IntN(5) {
	case 0:
		return "the " + pick(rng, nouns)
	case 1:
		return "the " + pick(rng, adjectives) + " " + pick(rng, nouns)
	case 2:
		return "a " + pick(rng, adjectives) + " " + pick(rng, nouns)
	case 3:
		return pick(rng, nouns) + " " + pick(rng, nouns)
	default:
		return "the " + pick(rng, nouns) + " " + pick(rng, nouns)
	}
}

// VerbPhrase generates a random verb phrase like "configure the proxy"
// or "gracefully restart the scheduler".
func VerbPhrase(rng *rand.Rand) string {
	switch rng.IntN(4) {
	case 0:
		return pick(rng, verbs) + " the " + pick(rng, nouns)
	case 1:
		return pick(rng, adverbs) + " " + pick(rng, verbs) + " the " + pick(rng, nouns)
	case 2:
		return pick(rng, verbs) + " the " + pick(rng, adjectives) + " " + pick(rng, nouns)
	default:
		return pick(rng, verbs) + " " + pick(rng, nouns) + " " + pick(rng, prepositions) + " the " + pick(rng, nouns)
	}
}

// Sentence generates a random sentence. Every call produces a structurally
// and lexically unique sentence from compositional generation.
func Sentence(rng *rand.Rand) string {
	var s string
	switch rng.IntN(20) {
	case 0:
		s = fmt.Sprintf("The %s %s the %s %s.",
			pick(rng, nouns), pick(rng, verbs), pick(rng, adjectives), pick(rng, nouns))
	case 1:
		s = fmt.Sprintf("The %s is %s by the %s.",
			pick(rng, nouns), pick(rng, verbs)+"d", pick(rng, nouns))
	case 2:
		s = fmt.Sprintf("%s the %s %s %s the %s.",
			capitalize(pick(rng, verbs)), pick(rng, nouns), pick(rng, conjunctions),
			pick(rng, verbs), pick(rng, nouns))
	case 3:
		s = fmt.Sprintf("If the %s %s, the %s will %s the %s.",
			pick(rng, nouns), pick(rng, verbs)+"s", pick(rng, nouns),
			pick(rng, verbs), pick(rng, nouns))
	case 4:
		s = fmt.Sprintf("The %s %s %s %s %s the %s.",
			pick(rng, adjectives), pick(rng, nouns), pick(rng, adverbs),
			pick(rng, verbs)+"s", pick(rng, prepositions), pick(rng, nouns))
	case 5:
		s = fmt.Sprintf("When the %s %s, %s the %s %s the %s.",
			pick(rng, nouns), pick(rng, verbs)+"s",
			pick(rng, verbs), pick(rng, nouns), pick(rng, prepositions), pick(rng, nouns))
	case 6:
		s = fmt.Sprintf("The %s requires %s to %s the %s.",
			pick(rng, nouns), pick(rng, nouns), pick(rng, verbs), pick(rng, nouns))
	case 7:
		s = fmt.Sprintf("Use %s to %s the %s %s.",
			pick(rng, techNames), pick(rng, verbs), pick(rng, adjectives), pick(rng, nouns))
	case 8:
		s = fmt.Sprintf("The %s %s is %s for the %s.",
			pick(rng, nouns), pick(rng, nouns), pick(rng, adjectives), pick(rng, nouns))
	case 9:
		s = fmt.Sprintf("After the %s %s, the %s should %s %s.",
			pick(rng, nouns), pick(rng, verbs)+"s",
			pick(rng, nouns), pick(rng, verbs), pick(rng, adverbs))
	case 10:
		s = fmt.Sprintf("The %s between %s and %s is %s.",
			pick(rng, nouns), pick(rng, nouns), pick(rng, nouns), pick(rng, adjectives))
	case 11:
		s = fmt.Sprintf("To %s the %s, first %s the %s.",
			pick(rng, verbs), pick(rng, nouns), pick(rng, verbs), pick(rng, nouns))
	case 12:
		s = fmt.Sprintf("The %s %s %s when the %s exceeds %s.",
			pick(rng, nouns), pick(rng, verbs)+"s", pick(rng, adverbs),
			pick(rng, nouns), pick(rng, quantities))
	case 13:
		s = fmt.Sprintf("%s is %s than %s for this %s.",
			capitalize(pick(rng, techNames)), pick(rng, adjectives),
			pick(rng, techNames), pick(rng, nouns))
	case 14:
		s = fmt.Sprintf("The %s should %s %s before the %s %s.",
			pick(rng, nouns), pick(rng, verbs), pick(rng, adverbs),
			pick(rng, nouns), pick(rng, verbs)+"s")
	case 15:
		s = fmt.Sprintf("Without the %s, the %s cannot %s the %s.",
			pick(rng, adjectives)+" "+pick(rng, nouns), pick(rng, nouns),
			pick(rng, verbs), pick(rng, nouns))
	case 16:
		s = fmt.Sprintf("Set the %s to %s %s the %s starts.",
			pick(rng, nouns), pick(rng, quantities), pick(rng, prepositions),
			pick(rng, nouns))
	case 17:
		s = fmt.Sprintf("The %s %s handles %s %s from the %s.",
			pick(rng, adjectives), pick(rng, nouns), pick(rng, adjectives),
			pick(rng, nouns)+"s", pick(rng, nouns))
	case 18:
		s = fmt.Sprintf("Both the %s and the %s need to %s the %s.",
			pick(rng, nouns), pick(rng, nouns), pick(rng, verbs), pick(rng, nouns))
	case 19:
		s = fmt.Sprintf("The %s reported %s %s %s the last %s.",
			pick(rng, nouns), pick(rng, quantities), pick(rng, nouns)+"s",
			pick(rng, prepositions), pick(rng, nouns))
	}
	return s
}

// Paragraph generates a paragraph of n sentences.
func Paragraph(rng *rand.Rand, n int) string {
	parts := make([]string, n)
	for i := range parts {
		parts[i] = Sentence(rng)
	}
	return strings.Join(parts, " ")
}

// Thought generates a multi-sentence thought/reasoning block.
// Length is 3-12 sentences.
func Thought(rng *rand.Rand, minSentences, maxSentences int) string {
	n := minSentences + rng.IntN(maxSentences-minSentences+1)
	return Paragraph(rng, n)
}

// MarkdownHeader generates a random 1-3 word section header.
func MarkdownHeader(rng *rand.Rand) string {
	switch rng.IntN(4) {
	case 0:
		return capitalize(pick(rng, nouns))
	case 1:
		return capitalize(pick(rng, nouns)) + " " + capitalize(pick(rng, nouns))
	case 2:
		return capitalize(pick(rng, adjectives)) + " " + capitalize(pick(rng, nouns))
	default:
		return capitalize(pick(rng, nouns)) + " " + capitalize(pick(rng, nouns)) + " " + capitalize(pick(rng, nouns))
	}
}

// ListItem generates a random action item for bulleted/numbered lists.
func ListItem(rng *rand.Rand) string {
	return capitalize(VerbPhrase(rng))
}

// MemoryEntry generates a random context memory statement.
func MemoryEntry(rng *rand.Rand) string {
	switch rng.IntN(12) {
	case 0:
		return fmt.Sprintf("User prefers %s for %s. Established during %s discussion.",
			TechName(rng), Topic(rng), Topic(rng))
	case 1:
		return fmt.Sprintf("The %s service runs on port %s. Migrated from %s last week.",
			Noun(rng), Quantity(rng), TechName(rng))
	case 2:
		return fmt.Sprintf("Default %s timeout is %s. Increase for %s operations.",
			Noun(rng), Quantity(rng), Adjective(rng))
	case 3:
		return fmt.Sprintf("The %s %s has %s active %s. Partitioned by %s.",
			Adjective(rng), Noun(rng), Quantity(rng), Noun(rng)+"s", Noun(rng))
	case 4:
		return fmt.Sprintf("Known issue: %s returns incorrect results when %s contains %s. Workaround: %s the %s.",
			Noun(rng), Noun(rng), Adjective(rng)+" "+Noun(rng)+"s",
			Verb(rng), Noun(rng))
	case 5:
		return fmt.Sprintf("Architecture decision: use %s for all %s services. Approved by %s team.",
			TechName(rng), Adjective(rng), Noun(rng))
	case 6:
		return fmt.Sprintf("The %s %s was deployed %s. Currently running version %s.",
			Adjective(rng), Noun(rng), Adverb(rng), Quantity(rng))
	case 7:
		return fmt.Sprintf("Sprint planning: %s is blocked by %s. Expected to unblock after %s.",
			Topic(rng), Topic(rng), Topic(rng))
	case 8:
		return fmt.Sprintf("The %s rotates every %s. Current %s was issued for %s.",
			Noun(rng), Quantity(rng), Noun(rng), TechName(rng))
	case 9:
		return fmt.Sprintf("Peak traffic for %s occurs between %s and %s. Auto-scaling is configured.",
			Noun(rng), Quantity(rng), Quantity(rng))
	case 10:
		return fmt.Sprintf("Previous %s caused a %s outage. Root cause was %s %s.",
			Noun(rng), Quantity(rng), Adjective(rng), Noun(rng))
	default:
		return fmt.Sprintf("The %s %s uses %s. Do not change without updating %s.",
			Noun(rng), Noun(rng), TechName(rng), Noun(rng))
	}
}

// Adverb returns a random adverb.
func Adverb(rng *rand.Rand) string { return pick(rng, adverbs) }

// IntroSentence generates a random intro sentence for markdown sections.
func IntroSentence(rng *rand.Rand) string {
	switch rng.IntN(16) {
	case 0:
		return fmt.Sprintf("Here is how to %s the %s:", Verb(rng), Noun(rng))
	case 1:
		return fmt.Sprintf("The following shows %s in action:", Topic(rng))
	case 2:
		return fmt.Sprintf("Use this approach for %s:", Topic(rng))
	case 3:
		return fmt.Sprintf("This is the recommended way to %s:", VerbPhrase(rng))
	case 4:
		return fmt.Sprintf("Below is a complete example of %s:", Topic(rng))
	case 5:
		return fmt.Sprintf("To %s, follow these steps:", VerbPhrase(rng))
	case 6:
		return fmt.Sprintf("The %s provides several options:", Topic(rng))
	case 7:
		return fmt.Sprintf("When working with %s, keep the following in mind:", Topic(rng))
	case 8:
		return fmt.Sprintf("Before modifying %s, review these guidelines:", Topic(rng))
	case 9:
		return fmt.Sprintf("Common mistakes when setting up %s include:", Topic(rng))
	case 10:
		return fmt.Sprintf("If you encounter issues with %s, try these solutions:", Topic(rng))
	case 11:
		return fmt.Sprintf("Recent changes to %s affect the following areas:", Topic(rng))
	case 12:
		return fmt.Sprintf("The security implications of %s deserve attention:", Topic(rng))
	case 13:
		return fmt.Sprintf("Performance characteristics of %s depend on these factors:", Topic(rng))
	case 14:
		return fmt.Sprintf("Key differences between %s implementations are listed below:", Topic(rng))
	default:
		return fmt.Sprintf("The migration path for %s requires careful planning:", Topic(rng))
	}
}

// TableHeader generates a random table header row.
func TableHeader(rng *rand.Rand) []string {
	nCols := 3 + rng.IntN(3) // 3-5 columns
	headers := make([]string, nCols)
	for i := range headers {
		headers[i] = capitalize(Noun(rng))
	}
	return headers
}

// TableCell generates a random table cell value.
func TableCell(rng *rand.Rand) string {
	switch rng.IntN(6) {
	case 0:
		return "`" + Noun(rng) + "`"
	case 1:
		return Quantity(rng)
	case 2:
		return Noun(rng)
	case 3:
		return capitalize(Adjective(rng))
	case 4:
		return pick(rng, []string{"Yes", "No", "Optional", "Required", "N/A", "true", "false", "null"})
	default:
		return TechName(rng)
	}
}

// ToolReason generates a random reason for using a tool.
func ToolReason(rng *rand.Rand, toolName string) string {
	switch rng.IntN(6) {
	case 0:
		return fmt.Sprintf("I need to %s the %s to find the relevant %s.",
			Verb(rng), Noun(rng), Noun(rng))
	case 1:
		return fmt.Sprintf("Let me %s to %s the %s data.",
			VerbPhrase(rng), Verb(rng), Adjective(rng))
	case 2:
		return fmt.Sprintf("The user needs the %s %s. Let me %s it.",
			Adjective(rng), Noun(rng), Verb(rng))
	case 3:
		return fmt.Sprintf("I should %s the %s before making changes to the %s.",
			Verb(rng), Noun(rng), Noun(rng))
	case 4:
		return fmt.Sprintf("A %s would be the best way to %s this %s.",
			toolName, Verb(rng), Noun(rng))
	default:
		return fmt.Sprintf("To %s the %s, I need to %s the %s first.",
			Verb(rng), Noun(rng), Verb(rng), Noun(rng))
	}
}

// SearchQuery generates a random search query string.
func SearchQuery(rng *rand.Rand) string {
	return Topic(rng)
}

// FilePath generates a random file path.
func FilePath(rng *rand.Rand) string {
	dirs := make([]string, 1+rng.IntN(3))
	for i := range dirs {
		dirs[i] = Noun(rng)
	}
	exts := []string{".json", ".yaml", ".yml", ".txt", ".csv", ".log", ".py", ".js",
		".ts", ".go", ".rs", ".java", ".sql", ".xml", ".toml", ".conf", ".cfg", ".env"}
	return "/" + strings.Join(dirs, "/") + "/" + Noun(rng) + pick(rng, exts)
}

// URL generates a random URL.
func URL(rng *rand.Rand) string {
	nParts := 2 + rng.IntN(3)
	parts := make([]string, nParts)
	for i := range parts {
		parts[i] = Noun(rng)
	}
	schemes := []string{"https://", "http://"}
	domains := []string{"example.com", "api.example.com", "internal.corp", "service.local",
		"gateway.prod", "staging.example.io", "dev.example.net"}
	return pick(rng, schemes) + pick(rng, domains) + "/" + strings.Join(parts, "/")
}

// JSONBody generates a random JSON body string with key-value pairs.
func JSONBody(rng *rand.Rand) string {
	nFields := 3 + rng.IntN(5)
	fields := make([]string, nFields)
	for i := range fields {
		fields[i] = fmt.Sprintf(`"%s": "%s"`, Noun(rng), Word(rng))
	}
	return "{" + strings.Join(fields, ", ") + "}"
}

func capitalize(s string) string {
	if len(s) == 0 {
		return s
	}
	return strings.ToUpper(s[:1]) + s[1:]
}
