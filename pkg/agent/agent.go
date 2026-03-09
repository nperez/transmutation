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

// Package agent generates training data using the fixed agent response schema:
//
//	{"thought": "...", "answer": "..." | null, "tool": {...} | null, "memory": [...]}
//
// It supports curriculum stages that progressively increase content complexity
// and corruption.
package agent

import (
	"encoding/json"
	"math/rand/v2"

	"nickandperla.net/transmutation/pkg/languages"
	"nickandperla.net/transmutation/pkg/xmlconv"
)

// Stage controls what content complexity is included in generation.
type Stage int

const (
	StageSimple   Stage = 1 // Clean agent JSON, simple text values
	StageTools    Stage = 2 // + tool calls with embedded code
	StageMarkdown Stage = 3 // + markdown answers with code blocks
	StageCorrupt  Stage = 4 // + subtle corruption (applied by caller)
	StageWrapper  Stage = 5 // + preamble/postamble (applied by caller)
)

// Response is the agent response schema.
type Response struct {
	Thought string    `json:"thought"`
	Answer  *string   `json:"answer"`
	Tool    *ToolCall `json:"tool"`
	Memory  []string  `json:"memory"`
}

// ToolCall represents a tool invocation.
type ToolCall struct {
	ToolName  string `json:"tool_name"`
	Arguments any    `json:"arguments"`
}

type contentKind int

const (
	contentSimpleAnswer contentKind = iota
	contentCodeTool
	contentSimpleTool
	contentMarkdownAnswer
)

// Generator produces agent schema training data.
type Generator struct {
	rng   *rand.Rand
	langs map[string]languages.Generator
	stage Stage
}

// NewGenerator creates an agent data generator for the given curriculum stage.
func NewGenerator(rng *rand.Rand, stage Stage) *Generator {
	langMap := make(map[string]languages.Generator)
	for _, g := range languages.All() {
		langMap[g.Name()] = g
	}
	return &Generator{rng: rng, langs: langMap, stage: stage}
}

// Generate produces a clean JSON string and its target XML.
func (g *Generator) Generate() (string, string) {
	resp := g.generateResponse()

	jsonBytes, err := json.MarshalIndent(resp, "", "  ")
	if err != nil {
		panic("agent: marshal: " + err.Error())
	}

	xmlOut, err := xmlconv.Convert(jsonBytes)
	if err != nil {
		panic("agent: xmlconv: " + err.Error())
	}

	return string(jsonBytes), xmlOut
}

func (g *Generator) generateResponse() *Response {
	ck := g.pickContent()

	resp := &Response{
		Memory: g.generateMemory(),
	}

	switch ck {
	case contentSimpleAnswer:
		resp.Thought = g.answerThought()
		answer := g.simpleAnswer()
		resp.Answer = &answer
	case contentMarkdownAnswer:
		resp.Thought = g.answerThought()
		answer := g.markdownAnswer()
		resp.Answer = &answer
	case contentCodeTool:
		tool := g.codeTool()
		resp.Thought = g.toolThought(tool.ToolName)
		resp.Tool = &tool
	case contentSimpleTool:
		tool := g.simpleTool()
		resp.Thought = g.toolThought(tool.ToolName)
		resp.Tool = &tool
	}

	return resp
}

func (g *Generator) pickContent() contentKind {
	r := g.rng.Float64()
	switch g.stage {
	case StageSimple:
		// Stage 1: mix of simple and markdown answers (no tools).
		if r < 0.40 {
			return contentSimpleAnswer
		}
		return contentMarkdownAnswer
	case StageTools:
		// 40% simple answer, 35% code tool, 25% simple tool
		if r < 0.40 {
			return contentSimpleAnswer
		} else if r < 0.75 {
			return contentCodeTool
		}
		return contentSimpleTool
	default: // Stage 3, 4, 5
		// 25% simple answer, 30% code tool, 15% simple tool, 30% markdown answer
		if r < 0.25 {
			return contentSimpleAnswer
		} else if r < 0.55 {
			return contentCodeTool
		} else if r < 0.70 {
			return contentSimpleTool
		}
		return contentMarkdownAnswer
	}
}

func (g *Generator) pick(items []string) string {
	return items[g.rng.IntN(len(items))]
}
