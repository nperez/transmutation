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
	"math/rand/v2"

	"nickandperla.net/transmutation/pkg/jsongen"
)

// AsJsongenGenerators adapts all language generators to the jsongen.LanguageGenerator interface.
func AsJsongenGenerators() []jsongen.LanguageGenerator {
	all := All()
	adapted := make([]jsongen.LanguageGenerator, len(all))
	for i, g := range all {
		adapted[i] = &jsongenAdapter{g}
	}
	return adapted
}

type jsongenAdapter struct {
	g Generator
}

func (a *jsongenAdapter) Name() string                   { return a.g.Name() }
func (a *jsongenAdapter) Generate(rng *rand.Rand) string { return a.g.Generate(rng) }
