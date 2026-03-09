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
)

// Simple text building blocks.
var (
	simpleStarts = []string{
		"The %s is configured correctly.",
		"You can find the %s in the settings panel.",
		"To update the %s, use the admin dashboard.",
		"The %s was last modified on %s.",
		"There are currently %s active %s records.",
		"The system shows %s for the %s field.",
		"Based on the %s logs, everything looks normal.",
		"The %s endpoint returns the expected data.",
		"I've checked the %s and it looks correct.",
		"The %s module handles this automatically.",
		"You'll need to restart the %s service after changes.",
		"The current %s version supports this feature.",
		"According to the %s documentation, this is expected behavior.",
		"The %s process completed successfully.",
		"I recommend updating the %s to the latest version.",
		"The %s integration requires %s to be enabled first.",
		"Make sure the %s environment variable is set before starting the %s.",
		"The %s runs on a %s schedule by default.",
		"If the %s fails, check the %s for error details.",
		"The %s was deprecated in favor of the new %s approach.",
		"You should back up the %s before making any changes to %s.",
		"The %s team confirmed that %s is working as intended.",
		"After updating %s, the %s will need to be reindexed.",
		"The performance of %s depends heavily on the %s configuration.",
		"Setting %s to a higher value may improve %s throughput.",
		"The %s is rate-limited to %s requests per second.",
		"To debug %s issues, enable verbose logging in the %s.",
		"The %s cluster currently has %s healthy nodes.",
		"The %s migration completed but %s records failed validation.",
		"Please verify that %s has the correct permissions for %s.",
		"The %s health check reports all %s dependencies are available.",
		"Consider using %s instead of %s for better compatibility.",
		"The %s API returns paginated results with a default page size of %s.",
		"The %s timeout is currently set to %s milliseconds.",
		"You can monitor the %s metrics through the %s dashboard.",
		"The %s connection pool is limited to %s concurrent connections.",
		"The %s feature requires %s to be running version %s or higher.",
		"Enabling %s will increase memory usage by approximately %s percent.",
		"The %s cache hit ratio is currently at %s percent.",
		"To scale the %s horizontally, add more %s instances behind the load balancer.",
	}

	fillNouns = []string{
		"configuration", "authentication", "authorization", "database",
		"server", "deployment", "cache", "queue", "scheduler", "monitor",
		"load balancer", "proxy", "firewall", "certificate", "backup",
		"endpoint", "middleware", "gateway", "registry", "pipeline",
		"worker", "broker", "index", "replica", "partition",
		"webhook", "connector", "adapter", "validator", "transformer",
	}
)

func (g *Generator) simpleAnswer() string {
	n := 50 + g.rng.IntN(31) // 50-80 sentences
	parts := make([]string, n)
	for i := range parts {
		template := g.pick(simpleStarts)
		// Fill format verbs with random nouns/values.
		nArgs := strings.Count(template, "%s")
		args := make([]any, nArgs)
		for j := range args {
			if g.rng.Float64() < 0.7 {
				args[j] = g.pick(fillNouns)
			} else {
				args[j] = languages.RandInt(g.rng)
			}
		}
		parts[i] = fmt.Sprintf(template, args...)
	}
	return strings.Join(parts, " ")
}

// Markdown building blocks.
var (
	mdHeaders = []string{
		"Overview", "Getting Started", "Configuration", "Usage",
		"Examples", "API Reference", "Troubleshooting", "Installation",
		"Prerequisites", "Authentication", "Error Handling", "Best Practices",
		"Performance", "Security", "Deployment", "Testing",
		"Database Setup", "Environment Variables", "Quick Start", "Advanced Usage",
	}

	mdLangTags = []string{"python", "javascript", "sql", "go", "bash", "json", "yaml", "html", "css"}

	mdIntroSentences = []string{
		"Here's how to set up the %s:",
		"The following example shows %s in action:",
		"You can use this approach for %s:",
		"This is the recommended way to handle %s:",
		"Below is a complete example of %s:",
		"To implement %s, follow these steps:",
		"The %s module provides several options:",
		"There are multiple ways to configure %s:",
	}

	mdListItems = []string{
		"Install the required dependencies",
		"Configure the environment variables",
		"Run the initialization script",
		"Verify the connection settings",
		"Update the configuration file",
		"Restart the service",
		"Check the logs for errors",
		"Run the test suite",
		"Deploy to the staging environment",
		"Monitor the health endpoint",
		"Set up the database schema",
		"Configure the authentication provider",
		"Enable the feature flag",
		"Review the security settings",
		"Back up the existing data",
		"Rotate the API credentials",
		"Flush the cache layer",
		"Validate the SSL certificate chain",
		"Run the database migration scripts",
		"Update the load balancer configuration",
		"Configure the rate limiting rules",
		"Set up monitoring and alerting",
		"Review the access control lists",
		"Generate a new encryption key",
		"Verify the backup restore process",
	}

	mdTableHeaders = [][]string{
		{"Parameter", "Type", "Default", "Description"},
		{"Field", "Required", "Description"},
		{"Option", "Value", "Notes"},
		{"Environment Variable", "Default", "Description"},
		{"Endpoint", "Method", "Description"},
		{"Status Code", "Meaning", "Action"},
	}
)

func (g *Generator) markdownAnswer() string {
	var b strings.Builder

	numSections := 6 + g.rng.IntN(5) // 6-10 sections

	for i := range numSections {
		if i > 0 {
			b.WriteString("\n\n")
		}

		// Section header.
		level := "##"
		if i == 0 && g.rng.Float64() < 0.3 {
			level = "#"
		}
		b.WriteString(level + " " + g.pick(mdHeaders) + "\n\n")

		// Intro sentence.
		if g.rng.Float64() < 0.7 {
			b.WriteString(fmt.Sprintf(g.pick(mdIntroSentences), g.pick(topics)) + "\n\n")
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
	n := 5 + g.rng.IntN(5) // 5-9 sentences
	for i := range n {
		if i > 0 {
			b.WriteByte(' ')
		}
		template := g.pick(simpleStarts)
		nArgs := strings.Count(template, "%s")
		args := make([]any, nArgs)
		for j := range args {
			args[j] = g.pick(fillNouns)
		}
		b.WriteString(fmt.Sprintf(template, args...))
	}

	// Maybe add inline code references.
	if g.rng.Float64() < 0.5 {
		b.WriteString(fmt.Sprintf(" Use `%s` for this.", languages.SnakeFuncName(g.rng)))
	}
}

func (g *Generator) mdCodeBlock(b *strings.Builder) {
	langTag := g.pick(mdLangTags)
	b.WriteString("```" + langTag + "\n")

	// Generate 3-6 snippets concatenated for longer code blocks.
	nSnippets := 3 + g.rng.IntN(4)
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
	n := 8 + g.rng.IntN(10) // 8-17 items
	items := make([]string, n)
	used := make(map[int]bool)
	for i := range n {
		for {
			idx := g.rng.IntN(len(mdListItems))
			if !used[idx] {
				used[idx] = true
				items[i] = mdListItems[idx]
				break
			}
		}
	}
	for _, item := range items {
		b.WriteString("- " + item + "\n")
	}
}

func (g *Generator) mdNumberedList(b *strings.Builder) {
	n := 8 + g.rng.IntN(8) // 8-15 items
	items := make([]string, n)
	used := make(map[int]bool)
	for i := range n {
		for {
			idx := g.rng.IntN(len(mdListItems))
			if !used[idx] {
				used[idx] = true
				items[i] = mdListItems[idx]
				break
			}
		}
	}
	for i, item := range items {
		b.WriteString(fmt.Sprintf("%d. %s\n", i+1, item))
	}
}

func (g *Generator) mdTable(b *strings.Builder) {
	headers := g.pickHeaders(mdTableHeaders)
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
	numRows := 4 + g.rng.IntN(5) // 4-8 rows
	for range numRows {
		cells := make([]string, numCols)
		for j := range cells {
			switch g.rng.IntN(4) {
			case 0:
				cells[j] = "`" + languages.VarName(g.rng) + "`"
			case 1:
				cells[j] = languages.RandInt(g.rng)
			case 2:
				cells[j] = g.pick(fillNouns)
			default:
				cells[j] = g.pick([]string{"Yes", "No", "Optional", "Required", "N/A", "true", "false", "null"})
			}
		}
		b.WriteString("| " + strings.Join(cells, " | ") + " |\n")
	}
}

// pick overload for string slices of slices (for mdTableHeaders).
func (g *Generator) pickHeaders(items [][]string) []string {
	return items[g.rng.IntN(len(items))]
}
