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

import "fmt"

var topics = []string{
	"database connection settings",
	"user authentication flow",
	"API endpoint design",
	"error handling patterns",
	"data validation rules",
	"file processing pipeline",
	"deployment configuration",
	"caching strategy",
	"rate limiting setup",
	"logging configuration",
	"search functionality",
	"permission system",
	"data migration",
	"webhook integration",
	"batch processing job",
	"notification system",
	"session management",
	"input sanitization",
	"output formatting",
	"test coverage gaps",
	"performance bottleneck",
	"memory usage optimization",
	"concurrency handling",
	"retry logic",
	"circuit breaker pattern",
}

var answerStarts = []string{
	"The user is asking about %s.",
	"This question is about %s.",
	"I need to explain %s.",
	"Let me think about %s.",
	"I can help with %s.",
	"Looking at this, it's about %s.",
	"The user wants to understand %s.",
	"I should provide a clear explanation of %s.",
}

var answerContinuations = []string{
	" I'll provide a detailed explanation with examples.",
	" Let me break this down step by step.",
	" I'll include code examples to illustrate.",
	" The key concepts are straightforward.",
	" There are a few important considerations here.",
	" I'll cover the main approaches and trade-offs.",
	" Let me outline the recommended approach.",
	" I should mention the common pitfalls.",
}

var thoughtElaborations = []string{
	" First, I need to consider how the %s interacts with the existing %s.",
	" The main challenge here is ensuring the %s is compatible with the %s.",
	" I recall that the %s was recently updated, which affects how %s works.",
	" Looking at the current setup, the %s uses a standard %s pattern.",
	" Before answering, I should verify that the %s supports the required %s.",
	" The documentation mentions that %s requires specific %s configuration.",
	" From previous interactions, I know the user prefers %s over %s for this.",
	" The %s implementation depends on whether %s is enabled in the environment.",
	" It's worth noting that the %s has known limitations with %s.",
	" I should check if the %s version supports the latest %s features.",
	" One consideration is whether the %s needs to handle concurrent %s.",
	" The standard approach for %s involves setting up %s first.",
}

var toolReasons = map[string][]string{
	"execute_sql": {
		"I need to query the database to find the relevant %s records.",
		"Let me write a SQL query to retrieve the %s data.",
		"I should check the database for %s information.",
		"The user needs data from the %s table. Let me query it.",
	},
	"execute_python": {
		"I need to write a Python script to process the %s.",
		"Let me implement this %s logic in Python.",
		"I should use Python to analyze the %s data.",
		"A Python script would be the best way to handle this %s task.",
	},
	"execute_javascript": {
		"I need to implement the %s frontend logic in JavaScript.",
		"Let me write a JavaScript function to handle the %s.",
		"I should create a client-side script for the %s feature.",
		"A JavaScript implementation would work well for this %s interaction.",
	},
	"execute_shell": {
		"I need to run a shell command to check the %s status.",
		"Let me use a shell command to process the %s files.",
		"I should run a command to inspect the %s configuration.",
		"A shell command is the quickest way to handle this %s operation.",
	},
	"execute_go": {
		"I need to write Go code to implement the %s handler.",
		"Let me create a Go function for the %s processing.",
		"I should implement this %s service in Go.",
		"A Go implementation would be efficient for this %s task.",
	},
	"search": {
		"I need to search for information about %s.",
		"Let me look up the relevant %s documentation.",
		"I should find resources related to %s.",
	},
	"read_file": {
		"I need to check the contents of the %s configuration file.",
		"Let me read the %s file to understand the current state.",
		"I should look at the %s file before making changes.",
	},
	"write_file": {
		"I need to create the %s configuration file.",
		"Let me write the updated %s to disk.",
		"I should save the %s output to a file.",
	},
	"http_request": {
		"I need to call the %s API endpoint.",
		"Let me make an HTTP request to fetch the %s data.",
		"I should hit the %s service to get the latest information.",
	},
}

func (g *Generator) answerThought() string {
	topic := g.pick(topics)
	s := fmt.Sprintf(g.pick(answerStarts), topic)
	s += g.pick(answerContinuations)
	// Add 6-12 elaboration sentences for longer thoughts.
	nElab := 6 + g.rng.IntN(7)
	for range nElab {
		s += fmt.Sprintf(g.pick(thoughtElaborations), g.pick(topics), g.pick(topics))
	}
	return s
}

func (g *Generator) toolThought(toolName string) string {
	topic := g.pick(topics)
	templates, ok := toolReasons[toolName]
	if !ok {
		templates = toolReasons["search"]
	}
	s := fmt.Sprintf(g.pick(templates), topic)
	// Add 5-10 elaboration sentences.
	nElab := 5 + g.rng.IntN(6)
	for range nElab {
		s += fmt.Sprintf(g.pick(thoughtElaborations), g.pick(topics), g.pick(topics))
	}
	return s
}
