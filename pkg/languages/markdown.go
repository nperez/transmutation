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

type Markdown struct{}

func (Markdown) Name() string { return "markdown" }

func (Markdown) Generate(rng *rand.Rand) string {
	switch rng.IntN(5) {
	case 0:
		return genMDHeaders(rng)
	case 1:
		return genMDCodeBlock(rng)
	case 2:
		return genMDTable(rng)
	case 3:
		return genMDList(rng)
	default:
		return genMDMixed(rng)
	}
}

var (
	mdTitles = []string{"Getting Started", "API Reference", "Configuration", "Installation", "Usage", "Troubleshooting", "FAQ", "Contributing", "Architecture", "Deployment"}
	mdLangs  = []string{"python", "javascript", "go", "bash", "sql", "json", "yaml", "typescript", "rust", "java"}
	mdWords  = []string{"implement", "configure", "deploy", "initialize", "authenticate", "validate", "process", "transform", "optimize", "monitor"}
)

func genMDHeaders(rng *rand.Rand) string {
	lines := []string{"# " + pick(rng, mdTitles), ""}
	sections := 2 + rng.IntN(3)
	for range sections {
		lines = append(lines, "## "+pick(rng, mdTitles))
		lines = append(lines, "")
		lines = append(lines, fmt.Sprintf("To %s the system, follow the steps below. Make sure you have the prerequisites installed before proceeding.",
			pick(rng, mdWords)))
		lines = append(lines, "")
	}
	return strings.Join(lines, "\n")
}

func genMDCodeBlock(rng *rand.Rand) string {
	lang := pick(rng, mdLangs)
	var code string
	switch lang {
	case "python":
		code = "def main():\n    print(\"Hello, World!\")\n    return 0"
	case "javascript":
		code = "const app = express();\napp.get('/', (req, res) => {\n  res.json({ status: 'ok' });\n});"
	case "bash":
		code = "#!/bin/bash\nset -euo pipefail\necho \"Starting deployment...\"\n./deploy.sh --env production"
	case "sql":
		code = "SELECT u.name, COUNT(o.id) as order_count\nFROM users u\nLEFT JOIN orders o ON u.id = o.user_id\nGROUP BY u.name\nHAVING COUNT(o.id) > 5;"
	default:
		code = fmt.Sprintf("// %s code example\n// TODO: add implementation", lang)
	}
	return fmt.Sprintf("```%s\n%s\n```", lang, code)
}

func genMDTable(rng *rand.Rand) string {
	headers := pickN(rng, []string{"Name", "Type", "Default", "Description", "Required", "Example", "Version"}, 3+rng.IntN(3))
	lines := []string{
		"| " + strings.Join(headers, " | ") + " |",
		"| " + strings.Repeat("--- | ", len(headers)),
	}
	rows := 2 + rng.IntN(4)
	for range rows {
		cells := make([]string, len(headers))
		for j := range cells {
			cells[j] = pick(rng, []string{"`string`", "`int`", "`true`", "`false`", "Required", "Optional", "`null`", "N/A", "`\"example\"`", "`42`"})
		}
		lines = append(lines, "| "+strings.Join(cells, " | ")+" |")
	}
	return strings.Join(lines, "\n")
}

func genMDList(rng *rand.Rand) string {
	items := 3 + rng.IntN(5)
	lines := make([]string, items)
	ordered := rng.Float64() < 0.4
	for i := range lines {
		text := fmt.Sprintf("%s the %s module", pick(rng, []string{"Install", "Configure", "Update", "Remove", "Enable", "Disable"}),
			pick(rng, []string{"authentication", "database", "cache", "logging", "monitoring", "notification"}))
		if ordered {
			lines[i] = fmt.Sprintf("%d. %s", i+1, text)
		} else {
			lines[i] = "- " + text
		}
		if rng.Float64() < 0.3 {
			lines[i] += "\n  - " + pick(rng, []string{"See docs for details", "Requires admin access", "Optional step", "May take a few minutes"})
		}
	}
	return strings.Join(lines, "\n")
}

func genMDMixed(rng *rand.Rand) string {
	parts := []string{
		"# " + pick(rng, mdTitles),
		"",
		fmt.Sprintf("This guide explains how to %s your application.", pick(rng, mdWords)),
		"",
		"## Prerequisites",
		"",
		"- Go 1.21 or later",
		"- Docker (optional)",
		fmt.Sprintf("- %s credentials", pick(rng, []string{"API", "Database", "AWS", "GCP"})),
		"",
		"## Quick Start",
		"",
		"```bash\ngit clone https://github.com/example/project.git\ncd project\nmake build\n```",
		"",
		fmt.Sprintf("> **Note**: %s before running in production.",
			pick(rng, []string{"Review the configuration", "Set environment variables", "Run the test suite", "Check the system requirements"})),
	}
	return strings.Join(parts, "\n")
}
