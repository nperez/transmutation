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

var goTypes = []string{
	"string", "int", "int64", "float64", "bool", "[]byte", "error",
	"context.Context", "time.Time", "time.Duration", "[]string",
	"map[string]any", "io.Reader", "io.Writer", "http.Handler",
}

func goTag(field string) string {
	lower := strings.ToLower(field[:1]) + field[1:]
	return fmt.Sprintf("`json:\"%s\"`", lower)
}

func genGoStruct(rng *rand.Rand) string {
	name := TypeName(rng)
	nFields := 3 + rng.IntN(5)
	lines := []string{fmt.Sprintf("type %s struct {", name)}
	for range nFields {
		field := FuncName(rng) // PascalCase field name
		lines = append(lines, fmt.Sprintf("\t%s %s %s", field, pick(rng, goTypes), goTag(field)))
	}
	lines = append(lines, "}")
	return strings.Join(lines, "\n")
}

func genGoFunc(rng *rand.Rand) string {
	name := FuncName(rng)
	receiver := TypeName(rng)
	arg := VarName(rng)
	argType := pick(rng, goTypes)
	retType := pick(rng, goTypes)
	innerCall := SnakeFuncName(rng)
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
		innerCall, arg,
		zeroValue(retType), strings.ToLower(name))
}

func genGoGoroutine(rng *rand.Rand) string {
	chanType := pick(rng, []string{"string", "int", "error", "struct{}", "[]byte"})
	itemsVar := VarName(rng)
	workerFunc := SnakeFuncName(rng)
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
		itemsVar,
		pick(rng, goTypes),
		workerFunc)
}

func genGoInterface(rng *rand.Rand) string {
	name := TypeName(rng)
	nMethods := 2 + rng.IntN(3)
	lines := []string{fmt.Sprintf("type %s interface {", name)}
	for range nMethods {
		m := FuncName(rng)
		lines = append(lines, fmt.Sprintf("\t%s(ctx context.Context, %s %s) (%s, error)",
			m, VarName(rng), pick(rng, goTypes), pick(rng, goTypes)))
	}
	lines = append(lines, "}")
	return strings.Join(lines, "\n")
}

func genGoErrorHandling(rng *rand.Rand) string {
	fn := SnakeFuncName(rng)
	errType := TypeName(rng)
	resultVar := VarName(rng)
	argVar := VarName(rng)
	return fmt.Sprintf(`%s, err := %s(%s)
if err != nil {
	var %sErr *%s
	if errors.As(err, &%sErr) {
		log.Printf("%s-specific error: %%v", %sErr)
		return nil, %sErr
	}
	return nil, fmt.Errorf("%s failed: %%w", err)
}`,
		resultVar, fn, argVar,
		VarName(rng), errType,
		VarName(rng),
		fn, VarName(rng),
		VarName(rng),
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
