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

package languages

import (
	"fmt"
	"math/rand/v2"
	"strings"
)

type HTML struct{}

func (HTML) Name() string { return "html" }

func (HTML) Generate(rng *rand.Rand) string {
	switch rng.IntN(4) {
	case 0:
		return genHTMLForm(rng)
	case 1:
		return genHTMLTable(rng)
	case 2:
		return genHTMLDiv(rng)
	default:
		return genHTMLPage(rng)
	}
}

var (
	htmlTags    = []string{"div", "span", "p", "section", "article", "header", "footer", "nav", "main", "aside"}
	htmlClasses = []string{"container", "wrapper", "content", "header", "footer", "sidebar", "card", "btn", "form-group", "alert", "modal", "nav-item"}
htmlInputs  = []string{"text", "email", "password", "number", "checkbox", "radio", "submit", "hidden", "date", "file"}
)

func genHTMLForm(rng *rand.Rand) string {
	fields := 2 + rng.IntN(4)
	actions := []string{"submit", "login", "register", "update", "search", "create", "import", "export"}
	lines := []string{fmt.Sprintf("<form action=\"/api/%s\" method=\"POST\">", pick(rng, actions))}
	for range fields {
		inputType := pick(rng, htmlInputs)
		name := VarName(rng)
		lines = append(lines, fmt.Sprintf("  <div class=\"%s\">", pick(rng, htmlClasses)))
		lines = append(lines, fmt.Sprintf("    <label for=\"%s\">%s</label>", name, titleCase(name)))
		lines = append(lines, fmt.Sprintf("    <input type=\"%s\" name=\"%s\" id=\"%s\" placeholder=\"Enter %s\" required />", inputType, name, name, name))
		lines = append(lines, "  </div>")
	}
	lines = append(lines, "  <button type=\"submit\" class=\"btn\">Submit</button>")
	lines = append(lines, "</form>")
	return strings.Join(lines, "\n")
}

func genHTMLTable(rng *rand.Rand) string {
	cols := pickN(rng, []string{"ID", "Name", "Email", "Status", "Created", "Actions", "Role", "Amount"}, 3+rng.IntN(3))
	rows := 2 + rng.IntN(4)
	lines := []string{"<table class=\"table\">", "  <thead>", "    <tr>"}
	for _, col := range cols {
		lines = append(lines, fmt.Sprintf("      <th>%s</th>", col))
	}
	lines = append(lines, "    </tr>", "  </thead>", "  <tbody>")
	for range rows {
		lines = append(lines, "    <tr>")
		for range cols {
			cellVals := []string{VarName(rng), RandInt(rng), "active", "pending", "admin", "$" + RandFloat(rng), "true", "false"}
			lines = append(lines, fmt.Sprintf("      <td>%s</td>", pick(rng, cellVals)))
		}
		lines = append(lines, "    </tr>")
	}
	lines = append(lines, "  </tbody>", "</table>")
	return strings.Join(lines, "\n")
}

func genHTMLDiv(rng *rand.Rand) string {
	depth := 1 + rng.IntN(3)
	return genHTMLNested(rng, depth, 0)
}

func genHTMLNested(rng *rand.Rand, maxDepth, currentDepth int) string {
	indent := strings.Repeat("  ", currentDepth)
	tag := pick(rng, htmlTags)
	class := pick(rng, htmlClasses)

	if currentDepth >= maxDepth {
		text := pick(rng, []string{"Hello World", "Click here", "Loading...", "No results found", "Welcome back!", "Error: something went wrong"})
		return fmt.Sprintf("%s<%s class=\"%s\">%s</%s>", indent, tag, class, text, tag)
	}

	children := 1 + rng.IntN(3)
	lines := []string{fmt.Sprintf("%s<%s class=\"%s\">", indent, tag, class)}
	for range children {
		lines = append(lines, genHTMLNested(rng, maxDepth, currentDepth+1))
	}
	lines = append(lines, fmt.Sprintf("%s</%s>", indent, tag))
	return strings.Join(lines, "\n")
}

func titleCase(s string) string {
	words := strings.Split(strings.ReplaceAll(s, "_", " "), " ")
	for i, w := range words {
		if len(w) > 0 {
			words[i] = strings.ToUpper(w[:1]) + w[1:]
		}
	}
	return strings.Join(words, " ")
}

func genHTMLPage(rng *rand.Rand) string {
	title := pick(rng, []string{"Dashboard", "Settings", "Profile", "Home", "Admin Panel"})
	return fmt.Sprintf("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n  <meta charset=\"UTF-8\">\n  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n  <title>%s</title>\n  <link rel=\"stylesheet\" href=\"/css/style.css\">\n</head>\n<body>\n  <div id=\"app\" class=\"%s\">\n    <h1>%s</h1>\n    <p>%s</p>\n  </div>\n  <script src=\"/js/main.js\"></script>\n</body>\n</html>",
		title, pick(rng, htmlClasses), title,
		pick(rng, []string{"Welcome to the application.", "Page content goes here.", "Loading data...", "Please log in to continue."}))
}
