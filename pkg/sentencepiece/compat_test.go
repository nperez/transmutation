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

package sentencepiece

import (
	"testing"
)

func TestEncodingConsistency(t *testing.T) {
	p := loadTestModel(t)

	// Verify that XML tag tokens each encode to a single USER_DEFINED piece.
	tags := []string{"<object>", "</object>", "<entry>", "</entry>",
		"<key>", "</key>", "<value>", "</value>", "<array>", "</array>",
		"<![CDATA[", "]]>"}

	for _, tag := range tags {
		ids := p.Encode(tag, false, false)
		// With dummy prefix, we expect [separator_token, tag_token].
		// Find the tag token (should be one of the USER_DEFINED IDs).
		found := false
		for _, id := range ids {
			if id >= 0 && id < len(p.pieces) {
				if p.pieces[id].GetType() == ModelProto_SentencePiece_USER_DEFINED {
					if p.pieces[id].GetPiece() == tag {
						found = true
					}
				}
			}
		}
		if !found {
			t.Errorf("Encode(%q) = %v, expected a single USER_DEFINED piece for the tag", tag, ids)
		}
	}
}

func TestEncodeXMLSnippet(t *testing.T) {
	p := loadTestModel(t)

	// A typical model output.
	xml := `<object><entry><key>name</key><value>Alice</value></entry></object>`

	ids := p.Encode(xml, false, false)
	decoded := p.Decode(ids)

	// Trim dummy prefix space.
	if len(decoded) > 0 && decoded[0] == ' ' {
		decoded = decoded[1:]
	}

	if decoded != xml {
		t.Errorf("round-trip failed:\n  input:   %q\n  decoded: %q\n  ids: %v", xml, decoded, ids)
	}
}
