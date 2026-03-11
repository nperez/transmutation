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

package agent

import (
	"fmt"
	"strings"

	"nickandperla.net/transmutation/pkg/languages"
	"nickandperla.net/transmutation/pkg/randtext"
)

var mdLangTags = []string{"python", "javascript", "sql", "go", "bash", "json", "yaml", "html", "css"}

func (g *Generator) simpleAnswer() string {
	n := 10 + g.rng.IntN(16) // 10-25 sentences
	parts := make([]string, n)
	for i := range parts {
		parts[i] = randtext.Sentence(g.rng)
	}
	return strings.Join(parts, " ")
}

func (g *Generator) markdownAnswer() string {
	var b strings.Builder

	numSections := 2 + g.rng.IntN(3) // 2-4 sections

	for i := range numSections {
		if i > 0 {
			b.WriteString("\n\n")
		}

		// Section header.
		level := "##"
		if i == 0 && g.rng.Float64() < 0.3 {
			level = "#"
		}
		b.WriteString(level + " " + randtext.MarkdownHeader(g.rng) + "\n\n")

		// Intro sentence.
		if g.rng.Float64() < 0.7 {
			b.WriteString(randtext.IntroSentence(g.rng) + "\n\n")
		}

		// Section body — pick a content type.
		switch g.rng.IntN(5) {
		case 0:
			g.mdParagraph(&b)
		case 1:
			g.mdCodeBlock(&b)
		case 2:
			g.mdBulletList(&b)
		case 3:
			g.mdTable(&b)
		case 4:
			g.mdNumberedList(&b)
		}
	}

	return b.String()
}

func (g *Generator) mdParagraph(b *strings.Builder) {
	n := 2 + g.rng.IntN(3) // 2-4 sentences
	for i := range n {
		if i > 0 {
			b.WriteByte(' ')
		}
		b.WriteString(randtext.Sentence(g.rng))
	}

	// Maybe add inline code references.
	if g.rng.Float64() < 0.5 {
		b.WriteString(fmt.Sprintf(" Use `%s` for this.", languages.SnakeFuncName(g.rng)))
	}
}

func (g *Generator) mdCodeBlock(b *strings.Builder) {
	langTag := g.pick(mdLangTags)
	b.WriteString("```" + langTag + "\n")

	// Generate 1-2 snippets for code blocks.
	nSnippets := 1 + g.rng.IntN(2)
	for i := range nSnippets {
		if i > 0 {
			b.WriteString("\n\n")
		}
		var code string
		if gen, ok := g.langs[langTag]; ok {
			code = gen.Generate(g.rng)
		} else if langTag == "bash" {
			if gen, ok := g.langs["shell"]; ok {
				code = gen.Generate(g.rng)
			}
		}
		if code == "" {
			code = fmt.Sprintf("// %s example\n%s = %s",
				langTag, languages.VarName(g.rng), languages.RandInt(g.rng))
		}
		b.WriteString(code)
	}

	b.WriteString("\n```")
}

func (g *Generator) mdBulletList(b *strings.Builder) {
	n := 3 + g.rng.IntN(4) // 3-6 items
	for range n {
		b.WriteString("- " + randtext.ListItem(g.rng) + "\n")
	}
}

func (g *Generator) mdNumberedList(b *strings.Builder) {
	n := 3 + g.rng.IntN(4) // 3-6 items
	for i := range n {
		b.WriteString(fmt.Sprintf("%d. %s\n", i+1, randtext.ListItem(g.rng)))
	}
}

func (g *Generator) mdTable(b *strings.Builder) {
	headers := randtext.TableHeader(g.rng)
	numCols := len(headers)

	// Header row.
	b.WriteString("| " + strings.Join(headers, " | ") + " |\n")

	// Separator.
	seps := make([]string, numCols)
	for i := range seps {
		seps[i] = "---"
	}
	b.WriteString("| " + strings.Join(seps, " | ") + " |\n")

	// Data rows.
	numRows := 2 + g.rng.IntN(3) // 2-4 rows
	for range numRows {
		cells := make([]string, numCols)
		for j := range cells {
			cells[j] = randtext.TableCell(g.rng)
		}
		b.WriteString("| " + strings.Join(cells, " | ") + " |\n")
	}
}
