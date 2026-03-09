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

type JavaScript struct{}

func (JavaScript) Name() string { return "javascript" }

func (JavaScript) Generate(rng *rand.Rand) string {
	switch rng.IntN(5) {
	case 0:
		return genJSArrow(rng)
	case 1:
		return genJSAsync(rng)
	case 2:
		return genJSDestructure(rng)
	case 3:
		return genJSTemplateLiteral(rng)
	default:
		return genJSClass(rng)
	}
}

func genJSArrow(rng *rand.Rand) string {
	name := FuncName(rng)
	nArgs := 1 + rng.IntN(3)
	args := make([]string, nArgs)
	for i := range args {
		args[i] = VarName(rng)
	}
	prop := VarName(rng)
	body := VarName(rng) + ".map(item => item." + prop + ")"
	if rng.Float64() < 0.5 {
		body = VarName(rng) + ".filter(x => x !== null).map(x => x." + prop + ")"
	}
	return fmt.Sprintf("const %s = (%s) => {\n  return %s;\n};", name, strings.Join(args, ", "), body)
}

func genJSAsync(rng *rand.Rand) string {
	name := FuncName(rng)
	arg := VarName(rng)
	url := fmt.Sprintf("`/api/v%d/%s/${%s}`", 1+rng.IntN(3), VarName(rng), VarName(rng))
	return fmt.Sprintf("async function %s(%s) {\n  try {\n    const response = await fetch(%s);\n    if (!response.ok) throw new Error(`HTTP ${response.status}`);\n    return await response.json();\n  } catch (error) {\n    console.error('Failed to %s:', error);\n    throw error;\n  }\n}",
		name, arg, url, name)
}

func genJSDestructure(rng *rand.Rand) string {
	nFields := 2 + rng.IntN(4)
	fields := make([]string, nFields)
	for i := range fields {
		fields[i] = VarName(rng)
	}
	source := VarName(rng)
	lines := []string{
		fmt.Sprintf("const { %s } = %s;", strings.Join(fields, ", "), source),
	}
	if rng.Float64() < 0.5 {
		arr := []string{VarName(rng), VarName(rng)}
		lines = append(lines, fmt.Sprintf("const [%s, ...rest] = %s;", strings.Join(arr, ", "), VarName(rng)))
	}
	return strings.Join(lines, "\n")
}

func genJSTemplateLiteral(rng *rand.Rand) string {
	nParts := 2 + rng.IntN(4)
	parts := make([]string, nParts)
	for i := range parts {
		if rng.Float64() < 0.5 {
			parts[i] = "${" + VarName(rng) + "}"
		} else {
			labels := []string{"Error:", "Status:", "User", "Result:", "ID:", "Path:", "Count:", "Value:"}
			parts[i] = pick(rng, labels)
		}
	}
	return fmt.Sprintf("const %s = `%s`;", VarName(rng), strings.Join(parts, " "))
}

func genJSClass(rng *rand.Rand) string {
	name := TypeName(rng)
	nFields := 2 + rng.IntN(4)
	fields := make([]string, nFields)
	for i := range fields {
		fields[i] = VarName(rng)
	}
	assignments := ""
	for _, f := range fields {
		assignments += fmt.Sprintf("\n    this.%s = %s;", f, f)
	}
	method := FuncName(rng)
	return fmt.Sprintf("class %s {\n  constructor(%s) {%s\n  }\n\n  async %s() {\n    return this.%s;\n  }\n}",
		name, strings.Join(fields, ", "), assignments, method, pick(rng, fields))
}
