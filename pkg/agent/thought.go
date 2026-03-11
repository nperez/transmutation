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

package agent

import "nickandperla.net/transmutation/pkg/randtext"

func (g *Generator) answerThought() string {
	// 3-6 sentences of reasoning about a random topic.
	return randtext.Thought(g.rng, 3, 6)
}

func (g *Generator) toolThought(toolName string) string {
	// Start with a tool-specific reason, then elaborate.
	s := randtext.ToolReason(g.rng, toolName)
	nElab := 2 + g.rng.IntN(3)
	for range nElab {
		s += " " + randtext.Sentence(g.rng)
	}
	return s
}
