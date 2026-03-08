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

type Shell struct{}

func (Shell) Name() string { return "shell" }

func (Shell) Generate(rng *rand.Rand) string {
	switch rng.IntN(5) {
	case 0:
		return genShellPipe(rng)
	case 1:
		return genShellConditional(rng)
	case 2:
		return genShellLoop(rng)
	case 3:
		return genShellFunction(rng)
	default:
		return genShellOneliner(rng)
	}
}

var (
	shCmds  = []string{"grep", "awk", "sed", "cut", "sort", "uniq", "wc", "head", "tail", "xargs", "find", "cat"}
	shFiles = []string{"/var/log/syslog", "/etc/hosts", "/tmp/output.txt", "$HOME/.config/app.conf", "/proc/meminfo", "data.csv", "input.json"}
	shVars  = []string{"$USER", "$HOME", "$PWD", "$PATH", "${HOSTNAME}", "${APP_ENV}", "${DB_HOST}", "${API_KEY}"}
	shFlags = []string{"-r", "-n", "-v", "-i", "-l", "-e", "-f", "-d", "-w"}
)

func genShellPipe(rng *rand.Rand) string {
	stages := make([]string, 2+rng.IntN(4))
	stages[0] = pick(rng, shCmds) + " " + pick(rng, shFlags) + " " + pick(rng, shFiles)
	for i := 1; i < len(stages); i++ {
		cmd := pick(rng, shCmds)
		switch cmd {
		case "grep":
			stages[i] = fmt.Sprintf("grep %s '%s'", pick(rng, shFlags), pick(rng, []string{"error", "warning", "failed", "success", "^[0-9]", "pattern.*match"}))
		case "awk":
			stages[i] = fmt.Sprintf("awk '{print $%d}'", 1+rng.IntN(5))
		case "sed":
			stages[i] = fmt.Sprintf("sed 's/%s/%s/g'", pick(rng, []string{"old", "foo", "error", "\\t"}), pick(rng, []string{"new", "bar", "warn", " "}))
		case "sort":
			stages[i] = "sort " + pick(rng, []string{"-n", "-r", "-u", "-k2", "-t,"})
		case "head":
			stages[i] = fmt.Sprintf("head -n %d", 5+rng.IntN(95))
		default:
			stages[i] = cmd + " " + pick(rng, shFlags)
		}
	}
	result := strings.Join(stages, " | ")
	if rng.Float64() < 0.3 {
		result += " > " + pick(rng, shFiles)
	}
	return result
}

func genShellConditional(rng *rand.Rand) string {
	cond := pick(rng, []string{
		fmt.Sprintf("-f %s", pick(rng, shFiles)),
		fmt.Sprintf("-d %s", pick(rng, shVars)),
		fmt.Sprintf("-z %s", pick(rng, shVars)),
		fmt.Sprintf("-n %s", pick(rng, shVars)),
		"$? -eq 0",
		"\"${APP_ENV}\" = \"production\"",
	})
	return fmt.Sprintf("if [ %s ]; then\n  echo \"condition met\"\n  %s %s\nelse\n  echo \"condition not met\" >&2\n  exit 1\nfi",
		cond, pick(rng, shCmds), pick(rng, shFiles))
}

func genShellLoop(rng *rand.Rand) string {
	if rng.Float64() < 0.5 {
		return fmt.Sprintf("for f in %s/*.log; do\n  echo \"Processing $f\"\n  %s \"$f\" >> /tmp/output.txt\ndone",
			pick(rng, []string{"/var/log", "$HOME/logs", "/tmp"}), pick(rng, shCmds))
	}
	return fmt.Sprintf("while IFS= read -r line; do\n  echo \"$line\" | %s %s\ndone < %s",
		pick(rng, shCmds), pick(rng, shFlags), pick(rng, shFiles))
}

func genShellFunction(rng *rand.Rand) string {
	name := pick(rng, []string{"cleanup", "deploy", "backup", "check_status", "setup_env", "run_tests"})
	return fmt.Sprintf("%s() {\n  local %s=%s\n  echo \"Running %s...\"\n  %s %s %s\n  return $?\n}",
		name, pick(rng, []string{"input", "output", "target", "config"}), pick(rng, shVars),
		name, pick(rng, shCmds), pick(rng, shFlags), pick(rng, shFiles))
}

func genShellOneliner(rng *rand.Rand) string {
	return pick(rng, []string{
		fmt.Sprintf("find %s -name '*.log' -mtime +%d -exec rm {} \\;", pick(rng, shVars), 7+rng.IntN(23)),
		fmt.Sprintf("tar -czf backup_%s.tar.gz %s", "${HOSTNAME}", pick(rng, shFiles)),
		fmt.Sprintf("curl -s -H 'Authorization: Bearer %s' https://api.example.com/v1/data | jq '.results[]'", pick(rng, shVars)),
		fmt.Sprintf("docker exec -it $(docker ps -q -f name=%s) /bin/bash", pick(rng, []string{"web", "api", "db", "redis"})),
		fmt.Sprintf("ssh %s@%s 'sudo systemctl restart %s'", pick(rng, []string{"deploy", "admin", "root"}), pick(rng, []string{"prod-01", "staging", "db-master"}), pick(rng, []string{"nginx", "app", "postgresql"})),
	})
}
