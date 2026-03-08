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

var (
	jsNames = []string{"getData", "handleRequest", "processItem", "validateInput", "formatOutput", "fetchUser", "parseResponse", "transformData", "filterResults", "computeTotal"}
	jsVars  = []string{"data", "config", "user", "items", "response", "result", "options", "payload", "context", "state"}
)

func genJSArrow(rng *rand.Rand) string {
	name := pick(rng, jsNames)
	args := pickN(rng, jsVars, 1+rng.IntN(3))
	body := pick(rng, jsVars) + ".map(item => item." + pick(rng, jsVars) + ")"
	if rng.Float64() < 0.5 {
		body = pick(rng, jsVars) + ".filter(x => x !== null).map(x => x." + pick(rng, jsVars) + ")"
	}
	return fmt.Sprintf("const %s = (%s) => {\n  return %s;\n};", name, strings.Join(args, ", "), body)
}

func genJSAsync(rng *rand.Rand) string {
	name := pick(rng, jsNames)
	arg := pick(rng, jsVars)
	url := pick(rng, []string{"`/api/v1/users/${id}`", "`/api/v2/data/${type}`", "`${baseUrl}/items/${id}`", "'/api/health'"})
	return fmt.Sprintf("async function %s(%s) {\n  try {\n    const response = await fetch(%s);\n    if (!response.ok) throw new Error(`HTTP ${response.status}`);\n    return await response.json();\n  } catch (error) {\n    console.error('Failed to %s:', error);\n    throw error;\n  }\n}",
		name, arg, url, name)
}

func genJSDestructure(rng *rand.Rand) string {
	fields := pickN(rng, jsVars, 2+rng.IntN(4))
	source := pick(rng, jsVars)
	lines := []string{
		fmt.Sprintf("const { %s } = %s;", strings.Join(fields, ", "), source),
	}
	if rng.Float64() < 0.5 {
		arr := pickN(rng, jsVars, 2)
		lines = append(lines, fmt.Sprintf("const [%s, ...rest] = %s;", strings.Join(arr, ", "), pick(rng, jsVars)))
	}
	return strings.Join(lines, "\n")
}

func genJSTemplateLiteral(rng *rand.Rand) string {
	parts := make([]string, 2+rng.IntN(3))
	for i := range parts {
		if rng.Float64() < 0.5 {
			parts[i] = "${" + pick(rng, jsVars) + "}"
		} else {
			parts[i] = pick(rng, []string{"Error:", "Status:", "User", "Result:", "ID:", "Path:"})
		}
	}
	return "const message = `" + strings.Join(parts, " ") + "`;"
}

func genJSClass(rng *rand.Rand) string {
	name := pick(rng, []string{"DataService", "ApiClient", "EventHandler", "StateManager", "CacheProvider", "AuthMiddleware"})
	fields := pickN(rng, jsVars, 2+rng.IntN(3))
	ctorArgs := strings.Join(fields, ", ")
	assignments := ""
	for _, f := range fields {
		assignments += fmt.Sprintf("\n    this.%s = %s;", f, f)
	}
	method := pick(rng, jsNames)
	return fmt.Sprintf("class %s {\n  constructor(%s) {%s\n  }\n\n  async %s() {\n    return this.%s;\n  }\n}",
		name, ctorArgs, assignments, method, pick(rng, fields))
}
