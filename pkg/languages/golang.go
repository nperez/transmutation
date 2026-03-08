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

type Go struct{}

func (Go) Name() string { return "go" }

func (Go) Generate(rng *rand.Rand) string {
	switch rng.IntN(5) {
	case 0:
		return genGoStruct(rng)
	case 1:
		return genGoFunc(rng)
	case 2:
		return genGoGoroutine(rng)
	case 3:
		return genGoInterface(rng)
	default:
		return genGoErrorHandling(rng)
	}
}

var (
	goTypes   = []string{"string", "int", "int64", "float64", "bool", "[]byte", "error", "context.Context", "time.Time", "time.Duration", "[]string", "map[string]any"}
	goNames   = []string{"Process", "Handle", "Validate", "Transform", "Parse", "Fetch", "Compute", "Execute", "Initialize", "Cleanup"}
	goVars    = []string{"data", "config", "user", "items", "result", "err", "ctx", "opts", "buf", "client"}
	goStructs = []string{"Server", "Client", "Handler", "Service", "Repository", "Config", "Request", "Response", "Worker", "Manager"}
	goFields  = []string{"ID", "Name", "Type", "Status", "CreatedAt", "UpdatedAt", "Config", "Data", "Logger", "Client", "Timeout", "MaxRetries"}
)

func genGoStruct(rng *rand.Rand) string {
	name := pick(rng, goStructs)
	fields := pickN(rng, goFields, 3+rng.IntN(4))
	lines := []string{fmt.Sprintf("type %s struct {", name)}
	for _, f := range fields {
		tag := fmt.Sprintf("`json:\"%s\"`", strings.ToLower(f[:1])+f[1:])
		lines = append(lines, fmt.Sprintf("\t%s %s %s", f, pick(rng, goTypes), tag))
	}
	lines = append(lines, "}")
	return strings.Join(lines, "\n")
}

func genGoFunc(rng *rand.Rand) string {
	name := pick(rng, goNames)
	receiver := pick(rng, goStructs)
	arg := pick(rng, goVars)
	argType := pick(rng, goTypes)
	retType := pick(rng, goTypes)
	return fmt.Sprintf(`func (s *%s) %s(%s %s) (%s, error) {
	if %s == nil {
		return %s, fmt.Errorf("%s: %s is required")
	}
	result, err := s.%s(%s)
	if err != nil {
		return %s, fmt.Errorf("%s: %%w", err)
	}
	return result, nil
}`, receiver, name, arg, argType, retType,
		arg, zeroValue(retType), strings.ToLower(name), arg,
		pick(rng, []string{"process", "handle", "validate", "execute"}), arg,
		zeroValue(retType), strings.ToLower(name))
}

func genGoGoroutine(rng *rand.Rand) string {
	chanType := pick(rng, []string{"string", "int", "error", "struct{}", "[]byte"})
	return fmt.Sprintf(`ch := make(chan %s, %d)
var wg sync.WaitGroup

for _, item := range %s {
	wg.Add(1)
	go func(v %s) {
		defer wg.Done()
		result, err := %s(v)
		if err != nil {
			log.Printf("error processing %%v: %%v", v, err)
			return
		}
		ch <- result
	}(item)
}

go func() {
	wg.Wait()
	close(ch)
}()

for result := range ch {
	fmt.Println(result)
}`, chanType, 1+rng.IntN(100),
		pick(rng, goVars),
		pick(rng, goTypes),
		strings.ToLower(pick(rng, goNames)))
}

func genGoInterface(rng *rand.Rand) string {
	name := pick(rng, goStructs) + "er"
	methods := pickN(rng, goNames, 2+rng.IntN(3))
	lines := []string{fmt.Sprintf("type %s interface {", name)}
	for _, m := range methods {
		lines = append(lines, fmt.Sprintf("\t%s(ctx context.Context, %s %s) (%s, error)",
			m, pick(rng, goVars), pick(rng, goTypes), pick(rng, goTypes)))
	}
	lines = append(lines, "}")
	return strings.Join(lines, "\n")
}

func genGoErrorHandling(rng *rand.Rand) string {
	fn := strings.ToLower(pick(rng, goNames))
	return fmt.Sprintf(`%s, err := %s(%s)
if err != nil {
	var %sErr *%sError
	if errors.As(err, &%sErr) {
		log.Printf("%s-specific error: %%v", %sErr)
		return nil, %sErr
	}
	return nil, fmt.Errorf("%s failed: %%w", err)
}`,
		pick(rng, goVars), fn, pick(rng, goVars),
		fn, pick(rng, goStructs),
		fn,
		fn, fn,
		fn,
		fn)
}

func zeroValue(t string) string {
	switch t {
	case "string":
		return `""`
	case "int", "int64", "float64":
		return "0"
	case "bool":
		return "false"
	default:
		return "nil"
	}
}
