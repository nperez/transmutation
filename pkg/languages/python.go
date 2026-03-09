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
	pyTypes   = []string{"str", "int", "float", "bool", "list", "dict", "Optional[str]", "List[int]", "Dict[str, Any]", "Tuple[int, ...]", "bytes", "Set[str]", "Any"}
	pyModules = []string{"os", "sys", "json", "pathlib", "typing", "dataclasses", "asyncio", "logging", "collections", "functools", "itertools", "re", "datetime", "hashlib", "base64", "uuid"}
	pyExcepts = []string{"ValueError", "TypeError", "KeyError", "RuntimeError", "ConnectionError", "TimeoutError", "FileNotFoundError", "PermissionError", "IndexError", "AttributeError"}
)

func genPyFunction(rng *rand.Rand) string {
	name := SnakeFuncName(rng)
	nArgs := 1 + rng.IntN(4)
	args := make([]string, nArgs)
	for i := range args {
		args[i] = VarName(rng) + ": " + pick(rng, pyTypes)
	}
	ret := pick(rng, pyTypes)
	body := "    result = " + VarName(rng) + "\n"
	if rng.Float64() < 0.5 {
		body += fmt.Sprintf("    if not %s:\n        raise %s(\"invalid %s\")\n",
			VarName(rng), pick(rng, pyExcepts), VarName(rng))
	}
	body += "    return result"
	return fmt.Sprintf("def %s(%s) -> %s:\n%s", name, strings.Join(args, ", "), ret, body)
}

func genPyListComp(rng *rand.Rand) string {
	item := VarName(rng)
	coll := VarName(rng)
	transforms := []string{
		item + ".strip()", item + ".lower()", "str(" + item + ")",
		item + " * 2", item + "[0]", "len(" + item + ")",
		item + ".upper()", "int(" + item + ")", item + " + " + RandInt(rng),
	}
	transform := pick(rng, transforms)
	filter := ""
	if rng.Float64() < 0.6 {
		filters := []string{
			item, item + " is not None", "len(" + item + ") > 0",
			item + " != ''", "isinstance(" + item + ", str)",
			item + " > " + RandInt(rng), item + " not in seen",
		}
		filter = " if " + pick(rng, filters)
	}
	return fmt.Sprintf("[%s for %s in %s%s]", transform, item, coll, filter)
}

func genPyClass(rng *rand.Rand) string {
	name := TypeName(rng)
	nFields := 2 + rng.IntN(4)
	fields := make([]string, nFields)
	for i := range fields {
		fields[i] = VarName(rng)
	}
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
		method := SnakeFuncName(rng)
		lines = append(lines, "", fmt.Sprintf("    def %s(self) -> %s:", method, pick(rng, pyTypes)))
		lines = append(lines, fmt.Sprintf("        return self.%s", pick(rng, fields)))
	}
	return strings.Join(lines, "\n")
}

func genPyImports(rng *rand.Rand) string {
	nMods := 2 + rng.IntN(5)
	mods := pickN(rng, pyModules, nMods)
	lines := make([]string, len(mods))
	for i, m := range mods {
		if rng.Float64() < 0.3 {
			lines[i] = fmt.Sprintf("from %s import %s", m, SnakeFuncName(rng))
		} else {
			lines[i] = "import " + m
		}
	}
	return strings.Join(lines, "\n")
}

func genPyFString(rng *rand.Rand) string {
	nParts := 2 + rng.IntN(4)
	parts := make([]string, nParts)
	for i := range parts {
		if rng.Float64() < 0.5 {
			parts[i] = fmt.Sprintf("{%s}", VarName(rng))
		} else {
			labels := []string{"Error:", "Result:", "User", "Status:", "Value =", "Count:", "Path:", "ID:", "Total:"}
			parts[i] = pick(rng, labels)
		}
	}
	return "f\"" + strings.Join(parts, " ") + "\""
}

func genPyAsyncFunc(rng *rand.Rand) string {
	name := SnakeFuncName(rng)
	arg := VarName(rng)
	url := RandURL(rng)
	return fmt.Sprintf("async def %s(%s: %s) -> %s:\n    async with aiohttp.ClientSession() as session:\n        response = await session.get(\"%s\")\n        return await response.json()",
		name, arg, pick(rng, pyTypes), pick(rng, pyTypes), url)
}
