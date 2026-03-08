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

type Python struct{}

func (Python) Name() string { return "python" }

func (Python) Generate(rng *rand.Rand) string {
	switch rng.IntN(6) {
	case 0:
		return genPyFunction(rng)
	case 1:
		return genPyListComp(rng)
	case 2:
		return genPyClass(rng)
	case 3:
		return genPyImports(rng)
	case 4:
		return genPyFString(rng)
	default:
		return genPyAsyncFunc(rng)
	}
}

var (
	pyNames    = []string{"process", "handle", "validate", "transform", "parse", "fetch", "compute", "analyze", "filter", "convert"}
	pyArgs     = []string{"data", "config", "user", "items", "request", "response", "options", "result", "context", "params"}
	pyTypes    = []string{"str", "int", "float", "bool", "list", "dict", "Optional[str]", "List[int]", "Dict[str, Any]", "Tuple[int, ...]"}
	pyModules  = []string{"os", "sys", "json", "pathlib", "typing", "dataclasses", "asyncio", "logging", "collections", "functools", "itertools", "re", "datetime"}
	pyExcepts  = []string{"ValueError", "TypeError", "KeyError", "RuntimeError", "ConnectionError", "TimeoutError", "FileNotFoundError"}
)

func genPyFunction(rng *rand.Rand) string {
	name := pick(rng, pyNames)
	args := pickN(rng, pyArgs, 1+rng.IntN(3))
	typedArgs := make([]string, len(args))
	for i, a := range args {
		typedArgs[i] = a + ": " + pick(rng, pyTypes)
	}
	ret := pick(rng, pyTypes)
	body := "    result = " + pick(rng, pyArgs) + "\n"
	if rng.Float64() < 0.5 {
		body += fmt.Sprintf("    if not %s:\n        raise %s(\"invalid %s\")\n",
			pick(rng, pyArgs), pick(rng, pyExcepts), pick(rng, pyArgs))
	}
	body += "    return result"
	return fmt.Sprintf("def %s(%s) -> %s:\n%s", name, strings.Join(typedArgs, ", "), ret, body)
}

func genPyListComp(rng *rand.Rand) string {
	item := pick(rng, pyArgs)
	coll := pick(rng, pyArgs)
	transform := pick(rng, []string{
		item + ".strip()", item + ".lower()", "str(" + item + ")",
		item + " * 2", item + "[0]", "len(" + item + ")",
	})
	filter := ""
	if rng.Float64() < 0.6 {
		filter = fmt.Sprintf(" if %s", pick(rng, []string{
			item, item + " is not None", "len(" + item + ") > 0",
			item + " != ''", "isinstance(" + item + ", str)",
		}))
	}
	return fmt.Sprintf("[%s for %s in %s%s]", transform, item, coll, filter)
}

func genPyClass(rng *rand.Rand) string {
	name := "Data" + pick(rng, []string{"Handler", "Processor", "Manager", "Service", "Client", "Factory"})
	fields := pickN(rng, pyArgs, 2+rng.IntN(3))
	lines := []string{fmt.Sprintf("class %s:", name)}
	initArgs := make([]string, len(fields))
	for i, f := range fields {
		initArgs[i] = f + ": " + pick(rng, pyTypes)
	}
	lines = append(lines, fmt.Sprintf("    def __init__(self, %s):", strings.Join(initArgs, ", ")))
	for _, f := range fields {
		lines = append(lines, fmt.Sprintf("        self.%s = %s", f, f))
	}
	if rng.Float64() < 0.5 {
		lines = append(lines, "", "    def __repr__(self) -> str:")
		lines = append(lines, fmt.Sprintf("        return f\"%s(%s)\"", name, pick(rng, fields)+"={self."+pick(rng, fields)+"}"))
	}
	return strings.Join(lines, "\n")
}

func genPyImports(rng *rand.Rand) string {
	mods := pickN(rng, pyModules, 2+rng.IntN(4))
	lines := make([]string, len(mods))
	for i, m := range mods {
		if rng.Float64() < 0.3 {
			lines[i] = fmt.Sprintf("from %s import %s", m, pick(rng, pyNames))
		} else {
			lines[i] = "import " + m
		}
	}
	return strings.Join(lines, "\n")
}

func genPyFString(rng *rand.Rand) string {
	parts := make([]string, 2+rng.IntN(3))
	for i := range parts {
		if rng.Float64() < 0.5 {
			parts[i] = fmt.Sprintf("{%s}", pick(rng, pyArgs))
		} else {
			parts[i] = pick(rng, []string{"Error:", "Result:", "User", "Status:", "Value =", "Count:", "Path:"})
		}
	}
	return "f\"" + strings.Join(parts, " ") + "\""
}

func genPyAsyncFunc(rng *rand.Rand) string {
	name := pick(rng, pyNames)
	arg := pick(rng, pyArgs)
	return fmt.Sprintf("async def %s(%s: %s) -> %s:\n    async with aiohttp.ClientSession() as session:\n        response = await session.get(url)\n        return await response.json()",
		name, arg, pick(rng, pyTypes), pick(rng, pyTypes))
}
