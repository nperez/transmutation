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

// Command enrich takes existing haiku tool-call samples and adds complex
// argument structures to make training data more diverse. It reads JSONL
// training pairs from stdin, enriches a percentage of tool-call samples
// with additional arguments (nested objects, arrays, mixed types), and
// regenerates XML targets.
package main

import (
	"bufio"
	"encoding/json"
	"encoding/xml"
	"flag"
	"fmt"
	"math/rand/v2"
	"os"
	"strings"

	"nickandperla.net/transmutation/pkg/xmlconv"
)

type TrainingPair struct {
	Input  string `json:"input"`
	Target string `json:"target"`
}

// Extra argument fields to inject into tool calls, organized by tool type.
var extraArgsByTool = map[string][]map[string]any{
	"execute_sql": {
		{"database": "production", "timeout_ms": 5000, "read_only": true},
		{"database": "analytics", "limit": 1000, "offset": 0, "explain": true},
		{"connection": map[string]any{"host": "db.internal", "port": 5432, "ssl": true}},
		{"params": []any{"active", 30, true}, "prepared": true},
		{"schema": "public", "transaction": map[string]any{"isolation": "serializable", "retry": 3}},
	},
	"execute_python": {
		{"env": map[string]any{"PYTHONPATH": "/app/lib", "DEBUG": "1"}, "timeout": 30},
		{"args": []any{"--verbose", "--config", "prod.yaml"}, "working_dir": "/home/user"},
		{"requirements": []any{"pandas>=2.0", "numpy", "scikit-learn"}, "venv": true},
		{"stdin_data": "input,value\n1,hello\n2,world", "capture_stderr": true},
	},
	"execute_javascript": {
		{"runtime": "node", "args": []any{"--max-old-space-size=4096"}, "timeout": 60},
		{"env": map[string]any{"NODE_ENV": "production", "PORT": 3000}, "esm": true},
		{"packages": []any{"lodash", "axios", "cheerio"}, "typescript": false},
	},
	"execute_shell": {
		{"shell": "/bin/bash", "env": map[string]any{"PATH": "/usr/local/bin:/usr/bin", "HOME": "/root"}},
		{"timeout": 120, "working_dir": "/var/log", "capture_output": true},
		{"user": "deploy", "sudo": false, "args": []any{"-e", "-o", "pipefail"}},
	},
	"execute_go": {
		{"build_tags": []any{"integration", "linux"}, "race": true, "timeout": "30s"},
		{"env": map[string]any{"GOOS": "linux", "GOARCH": "amd64", "CGO_ENABLED": "0"}},
		{"args": []any{"-v", "-count=1", "./..."}, "working_dir": "/app"},
	},
	"search": {
		{"filters": map[string]any{"date_range": map[string]any{"start": "2024-01-01", "end": "2024-12-31"}, "type": "article"}, "max_results": 20},
		{"index": "documents", "boost": map[string]any{"title": 2.0, "body": 1.0}, "fuzzy": true},
		{"sources": []any{"web", "docs", "wiki"}, "language": "en", "safe_search": true},
	},
	"read_file": {
		{"encoding": "utf-8", "max_bytes": 1048576, "follow_links": true},
		{"line_range": map[string]any{"start": 1, "end": 100}, "strip_comments": false},
		{"format": "json", "validate": true},
	},
	"write_file": {
		{"mode": "0644", "create_dirs": true, "backup": true},
		{"encoding": "utf-8", "append": false, "atomic": true},
		{"owner": "www-data", "permissions": map[string]any{"read": true, "write": true, "execute": false}},
	},
	"http_request": {
		{"headers": map[string]any{"Authorization": "Bearer tok_xxx", "Content-Type": "application/json", "Accept": "application/json"}, "timeout": 30},
		{"follow_redirects": true, "max_retries": 3, "verify_ssl": true},
		{"query_params": map[string]any{"page": 1, "per_page": 50, "sort": "created_at"}, "cache": false},
	},
}

func main() {
	pct := flag.Float64("pct", 25, "percentage of tool-call samples to enrich (0-100)")
	seed := flag.Uint64("seed", 42, "random seed")
	flag.Parse()

	rng := rand.New(rand.NewPCG(*seed, *seed^0xbeef))
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 0, 1024*1024), 10*1024*1024)
	enc := json.NewEncoder(os.Stdout)

	total := 0
	tools := 0
	enriched := 0

	for scanner.Scan() {
		total++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		var pair TrainingPair
		if err := json.Unmarshal([]byte(line), &pair); err != nil {
			enc.Encode(pair)
			continue
		}

		// Parse the input JSON.
		var obj map[string]any
		if err := json.Unmarshal([]byte(pair.Input), &obj); err != nil {
			enc.Encode(pair)
			continue
		}

		// Check if it's a tool call.
		toolRaw, hasTool := obj["tool"]
		if !hasTool || toolRaw == nil {
			enc.Encode(pair)
			continue
		}
		tools++

		// Roll for enrichment.
		if rng.Float64()*100 >= *pct {
			enc.Encode(pair)
			continue
		}

		// Extract tool info.
		toolMap, ok := toolRaw.(map[string]any)
		if !ok {
			enc.Encode(pair)
			continue
		}
		toolName, _ := toolMap["tool_name"].(string)
		argsRaw, hasArgs := toolMap["arguments"]
		if !hasArgs {
			enc.Encode(pair)
			continue
		}
		args, ok := argsRaw.(map[string]any)
		if !ok {
			enc.Encode(pair)
			continue
		}

		// Pick extra args for this tool type.
		extras, hasExtras := extraArgsByTool[toolName]
		if !hasExtras || len(extras) == 0 {
			enc.Encode(pair)
			continue
		}

		// Merge random extra args into existing args.
		extra := extras[rng.IntN(len(extras))]
		for k, v := range extra {
			args[k] = v
		}
		toolMap["arguments"] = args

		// Regenerate pretty-printed JSON and XML.
		pretty, err := json.MarshalIndent(obj, "", "  ")
		if err != nil {
			enc.Encode(pair) // fallback
			continue
		}
		xmlOut, err := xmlconv.Convert(pretty)
		if err != nil {
			enc.Encode(pair) // fallback
			continue
		}

		// Verify XML.
		dec := xml.NewDecoder(strings.NewReader("<root>" + xmlOut + "</root>"))
		valid := true
		for {
			_, err := dec.Token()
			if err != nil {
				if err.Error() != "EOF" {
					valid = false
				}
				break
			}
		}
		if !valid {
			enc.Encode(pair) // fallback
			continue
		}

		pair.Input = string(pretty)
		pair.Target = xmlOut
		enc.Encode(pair)
		enriched++
	}

	if err := scanner.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	fmt.Fprintf(os.Stderr, "Enriched %d of %d tool samples (%d total, %.1f%%)\n",
		enriched, tools, total, float64(enriched)/float64(max(tools, 1))*100)
}
