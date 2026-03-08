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

type CSS struct{}

func (CSS) Name() string { return "css" }

func (CSS) Generate(rng *rand.Rand) string {
	switch rng.IntN(4) {
	case 0:
		return genCSSRules(rng)
	case 1:
		return genCSSMediaQuery(rng)
	case 2:
		return genCSSVariables(rng)
	default:
		return genCSSAnimation(rng)
	}
}

var (
	cssSelectors = []string{".container", ".card", ".btn", ".header", ".footer", ".sidebar", ".modal", ".alert", ".nav", "#app", ".form-group", ".table", "body", "main", "h1"}
	cssProps     = []string{"display", "margin", "padding", "color", "background-color", "font-size", "border", "border-radius", "width", "height", "position", "top", "left", "z-index", "opacity", "transition", "box-shadow", "flex-direction", "justify-content", "align-items", "gap", "grid-template-columns", "overflow", "text-align", "font-weight", "line-height", "cursor"}
	cssValues    = []string{"#333", "#fff", "#007bff", "1rem", "0.5rem", "16px", "24px", "100%", "auto", "flex", "grid", "block", "none", "center", "space-between", "pointer", "relative", "absolute", "fixed", "bold", "1.5", "300ms ease", "0 2px 4px rgba(0,0,0,0.1)"}
	cssColors    = []string{"#1a1a2e", "#16213e", "#0f3460", "#e94560", "#533483", "#2b2d42", "#8d99ae", "#edf2f4", "#ef233c", "#d90429"}
)

func genCSSRules(rng *rand.Rand) string {
	rules := 2 + rng.IntN(4)
	lines := make([]string, 0, rules*5)
	for range rules {
		selector := pick(rng, cssSelectors)
		props := 2 + rng.IntN(4)
		lines = append(lines, selector+" {")
		for range props {
			lines = append(lines, fmt.Sprintf("  %s: %s;", pick(rng, cssProps), pick(rng, cssValues)))
		}
		lines = append(lines, "}", "")
	}
	return strings.Join(lines, "\n")
}

func genCSSMediaQuery(rng *rand.Rand) string {
	bp := pick(rng, []string{"768px", "1024px", "1280px", "480px", "640px"})
	dir := pick(rng, []string{"min-width", "max-width"})
	selector := pick(rng, cssSelectors)
	return fmt.Sprintf("@media (%s: %s) {\n  %s {\n    %s: %s;\n    %s: %s;\n  }\n}",
		dir, bp, selector,
		pick(rng, cssProps), pick(rng, cssValues),
		pick(rng, cssProps), pick(rng, cssValues))
}

func genCSSVariables(rng *rand.Rand) string {
	vars := 3 + rng.IntN(4)
	lines := []string{":root {"}
	for range vars {
		name := pick(rng, []string{"primary", "secondary", "accent", "bg", "text", "border", "shadow", "radius", "spacing"})
		lines = append(lines, fmt.Sprintf("  --%s: %s;", name, pick(rng, cssColors)))
	}
	lines = append(lines, "}", "")
	lines = append(lines, pick(rng, cssSelectors)+" {")
	lines = append(lines, "  color: var(--"+pick(rng, []string{"primary", "text", "secondary"})+");")
	lines = append(lines, "  background-color: var(--"+pick(rng, []string{"bg", "secondary", "accent"})+");")
	lines = append(lines, "}")
	return strings.Join(lines, "\n")
}

func genCSSAnimation(rng *rand.Rand) string {
	name := pick(rng, []string{"fadeIn", "slideUp", "bounce", "pulse", "spin", "shake"})
	return fmt.Sprintf("@keyframes %s {\n  0%% {\n    opacity: 0;\n    transform: %s;\n  }\n  100%% {\n    opacity: 1;\n    transform: %s;\n  }\n}\n\n%s {\n  animation: %s %s;\n}",
		name,
		pick(rng, []string{"translateY(20px)", "scale(0.8)", "rotate(0deg)", "translateX(-100%%)"}),
		pick(rng, []string{"translateY(0)", "scale(1)", "rotate(360deg)", "translateX(0)"}),
		pick(rng, cssSelectors), name,
		pick(rng, []string{"0.3s ease", "0.5s ease-in-out", "1s linear infinite", "0.2s cubic-bezier(0.4, 0, 0.2, 1)"}))
}
