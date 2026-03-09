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
	shFlags = []string{"-r", "-n", "-v", "-i", "-l", "-e", "-f", "-d", "-w", "-c", "-q"}
)

func shVar(rng *rand.Rand) string {
	names := []string{"USER", "HOME", "PWD", "PATH", "HOSTNAME", "APP_ENV", "DB_HOST", "API_KEY",
		"PORT", "LOG_LEVEL", "CONFIG_DIR", "DATA_DIR", "CACHE_TTL", "MAX_RETRIES"}
	if rng.Float64() < 0.5 {
		return "$" + pick(rng, names)
	}
	return "${" + pick(rng, names) + "}"
}

func shFile(rng *rand.Rand) string {
	return RandPath(rng)
}

func genShellPipe(rng *rand.Rand) string {
	nStages := 2 + rng.IntN(4)
	stages := make([]string, nStages)
	stages[0] = pick(rng, shCmds) + " " + pick(rng, shFlags) + " " + shFile(rng)
	for i := 1; i < nStages; i++ {
		cmd := pick(rng, shCmds)
		switch cmd {
		case "grep":
			patterns := []string{"error", "warning", "failed", "success", "timeout",
				VarName(rng), "^[0-9]", "pattern.*match", "not found"}
			stages[i] = fmt.Sprintf("grep %s '%s'", pick(rng, shFlags), pick(rng, patterns))
		case "awk":
			stages[i] = fmt.Sprintf("awk '{print $%d}'", 1+rng.IntN(8))
		case "sed":
			from := pick(rng, []string{VarName(rng), "old", "error", "\\t", "TODO"})
			to := pick(rng, []string{VarName(rng), "new", "warn", " ", "DONE"})
			stages[i] = fmt.Sprintf("sed 's/%s/%s/g'", from, to)
		case "sort":
			stages[i] = "sort " + pick(rng, []string{"-n", "-r", "-u", "-k2", "-t,", "-nr"})
		case "head":
			stages[i] = fmt.Sprintf("head -n %d", 1+rng.IntN(100))
		case "tail":
			stages[i] = fmt.Sprintf("tail -n %d", 1+rng.IntN(100))
		default:
			stages[i] = cmd + " " + pick(rng, shFlags)
		}
	}
	result := strings.Join(stages, " | ")
	if rng.Float64() < 0.3 {
		result += " > " + shFile(rng)
	}
	return result
}

func genShellConditional(rng *rand.Rand) string {
	conds := []string{
		fmt.Sprintf("-f %s", shFile(rng)),
		fmt.Sprintf("-d %s", shVar(rng)),
		fmt.Sprintf("-z %s", shVar(rng)),
		fmt.Sprintf("-n %s", shVar(rng)),
		"$? -eq 0",
		fmt.Sprintf("\"%s\" = \"%s\"", shVar(rng), VarName(rng)),
		fmt.Sprintf("%s -gt %s", shVar(rng), RandInt(rng)),
	}
	return fmt.Sprintf("if [ %s ]; then\n  echo \"condition met\"\n  %s %s\nelse\n  echo \"condition not met\" >&2\n  exit 1\nfi",
		pick(rng, conds), pick(rng, shCmds), shFile(rng))
}

func genShellLoop(rng *rand.Rand) string {
	if rng.Float64() < 0.5 {
		exts := []string{"*.log", "*.txt", "*.csv", "*.json", "*.yaml"}
		return fmt.Sprintf("for f in %s/%s; do\n  echo \"Processing $f\"\n  %s \"$f\" >> %s\ndone",
			shVar(rng), pick(rng, exts), pick(rng, shCmds), shFile(rng))
	}
	return fmt.Sprintf("while IFS= read -r line; do\n  echo \"$line\" | %s %s\ndone < %s",
		pick(rng, shCmds), pick(rng, shFlags), shFile(rng))
}

func genShellFunction(rng *rand.Rand) string {
	name := SnakeFuncName(rng)
	localVar := VarName(rng)
	return fmt.Sprintf("%s() {\n  local %s=%s\n  echo \"Running %s...\"\n  %s %s %s\n  return $?\n}",
		name, localVar, shVar(rng),
		name, pick(rng, shCmds), pick(rng, shFlags), shFile(rng))
}

func genShellOneliner(rng *rand.Rand) string {
	oneliners := []string{
		fmt.Sprintf("find %s -name '*.log' -mtime +%d -exec rm {} \\;", shVar(rng), 1+rng.IntN(30)),
		fmt.Sprintf("tar -czf backup_%s.tar.gz %s", RandInt(rng), shFile(rng)),
		fmt.Sprintf("curl -s -H 'Authorization: Bearer %s' %s | jq '.results[]'", shVar(rng), RandURL(rng)),
		fmt.Sprintf("docker exec -it $(docker ps -q -f name=%s) /bin/bash", VarName(rng)),
		fmt.Sprintf("ssh %s@%s 'sudo systemctl restart %s'",
			pick(rng, []string{"deploy", "admin", "root"}),
			VarName(rng)+"-"+RandInt(rng),
			VarName(rng)),
		fmt.Sprintf("rsync -avz --delete %s/ %s@%s:%s/", shFile(rng),
			pick(rng, []string{"deploy", "backup"}), VarName(rng), shFile(rng)),
	}
	return pick(rng, oneliners)
}
