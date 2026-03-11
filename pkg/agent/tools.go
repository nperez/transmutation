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
	"strings"

	"nickandperla.net/transmutation/pkg/randtext"
)

// Argument structs use ordered struct fields for deterministic JSON output.

type sqlArgs struct {
	Query string `json:"query"`
}

type codeArgs struct {
	Code string `json:"code"`
}

type shellArgs struct {
	Command string `json:"command"`
}

type searchArgs struct {
	Query string `json:"query"`
}

type readFileArgs struct {
	Path string `json:"path"`
}

type writeFileArgs struct {
	Path    string `json:"path"`
	Content string `json:"content"`
}

type httpRequestArgs struct {
	URL    string `json:"url"`
	Method string `json:"method"`
	Body   string `json:"body,omitempty"`
}

var codeToolNames = []string{
	"execute_sql",
	"execute_python",
	"execute_javascript",
	"execute_shell",
	"execute_go",
}

var simpleToolNames = []string{
	"search",
	"read_file",
	"write_file",
	"http_request",
}

// codeTool generates a tool call with embedded code as an argument.
// Concatenates 2-4 snippets to produce realistically long code.
func (g *Generator) codeTool() ToolCall {
	name := g.pick(codeToolNames)
	nSnippets := 1 + g.rng.IntN(3) // 1-3 snippets

	var args any
	switch name {
	case "execute_sql":
		gen := g.langs["sql"]
		parts := make([]string, nSnippets)
		for i := range parts {
			parts[i] = gen.Generate(g.rng)
		}
		args = sqlArgs{Query: strings.Join(parts, ";\n")}
	case "execute_python":
		gen := g.langs["python"]
		parts := make([]string, nSnippets)
		for i := range parts {
			parts[i] = gen.Generate(g.rng)
		}
		args = codeArgs{Code: strings.Join(parts, "\n\n")}
	case "execute_javascript":
		gen := g.langs["javascript"]
		parts := make([]string, nSnippets)
		for i := range parts {
			parts[i] = gen.Generate(g.rng)
		}
		args = codeArgs{Code: strings.Join(parts, "\n\n")}
	case "execute_shell":
		gen := g.langs["shell"]
		parts := make([]string, nSnippets)
		for i := range parts {
			parts[i] = gen.Generate(g.rng)
		}
		args = shellArgs{Command: strings.Join(parts, "\n")}
	case "execute_go":
		gen := g.langs["go"]
		parts := make([]string, nSnippets)
		for i := range parts {
			parts[i] = gen.Generate(g.rng)
		}
		args = codeArgs{Code: strings.Join(parts, "\n\n")}
	}

	return ToolCall{ToolName: name, Arguments: args}
}

// simpleTool generates a tool call with simple string/URL arguments.
func (g *Generator) simpleTool() ToolCall {
	name := g.pick(simpleToolNames)

	var args any
	switch name {
	case "search":
		args = searchArgs{
			Query: randtext.SearchQuery(g.rng),
		}
	case "read_file":
		args = readFileArgs{
			Path: randtext.FilePath(g.rng),
		}
	case "write_file":
		// Always use simpleAnswer for substantial content.
		args = writeFileArgs{
			Path:    randtext.FilePath(g.rng),
			Content: g.simpleAnswer(),
		}
	case "http_request":
		method := g.pick([]string{"GET", "POST", "PUT", "DELETE", "PATCH"})
		var body string
		if method == "POST" || method == "PUT" || method == "PATCH" {
			// Build a multi-field JSON body.
			nFields := 3 + g.rng.IntN(5) // 3-7 fields
			fields := make([]string, nFields)
			for i := range fields {
				fields[i] = `"` + randtext.Noun(g.rng) + `": "` + randtext.Noun(g.rng) + `"`
			}
			body = "{" + strings.Join(fields, ", ") + "}"
		}
		args = httpRequestArgs{
			URL:    randtext.URL(g.rng),
			Method: method,
			Body:   body,
		}
	}

	return ToolCall{ToolName: name, Arguments: args}
}
