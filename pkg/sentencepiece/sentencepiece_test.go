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
	"os"
	"strings"
	"testing"
)

func loadTestModel(t *testing.T) *Processor {
	t.Helper()
	path := "../../models/tokenizer.model"
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skip("tokenizer model not found at", path)
	}
	p, err := Load(path)
	if err != nil {
		t.Fatal("failed to load model:", err)
	}
	return p
}

func TestSpecialIDs(t *testing.T) {
	p := loadTestModel(t)
	if p.BOS() < 0 {
		t.Error("BOS not found")
	}
	if p.EOS() < 0 {
		t.Error("EOS not found")
	}
	if p.PAD() < 0 {
		t.Error("PAD not found")
	}
	if p.UNK() < 0 {
		t.Error("UNK not found")
	}
	t.Logf("BOS=%d EOS=%d PAD=%d UNK=%d Vocab=%d", p.BOS(), p.EOS(), p.PAD(), p.UNK(), p.VocabSize())
}

func TestDecodeXMLTags(t *testing.T) {
	p := loadTestModel(t)

	// These are the XML tags that must be USER_DEFINED pieces.
	tags := []string{
		"<object>", "</object>",
		"<entry>", "</entry>",
		"<key>", "</key>",
		"<value>", "</value>",
		"<array>", "</array>",
		"<![CDATA[", "]]>",
	}

	for _, tag := range tags {
		// Encode the tag.
		ids := p.Encode(tag, false, false)
		if len(ids) == 0 {
			t.Errorf("Encode(%q) returned empty", tag)
			continue
		}

		// Decode back and check round-trip.
		decoded := p.Decode(ids)
		// The dummy prefix adds a leading space; trim it for comparison.
		decoded = strings.TrimLeft(decoded, " ")
		if decoded != tag {
			t.Errorf("round-trip failed for %q: encoded to %v, decoded to %q", tag, ids, decoded)
		}
	}
}

func TestEncodeDecodeSimple(t *testing.T) {
	p := loadTestModel(t)

	text := `{"name": "Alice", "age": 30}`
	ids := p.Encode(text, false, false)
	if len(ids) == 0 {
		t.Fatal("Encode returned empty for simple JSON")
	}

	decoded := p.Decode(ids)
	// Trim the dummy prefix space.
	decoded = strings.TrimLeft(decoded, " ")
	if decoded != text {
		t.Errorf("round-trip mismatch:\n  input:   %q\n  decoded: %q", text, decoded)
	}
}

func TestEncodeDecodeXMLDocument(t *testing.T) {
	p := loadTestModel(t)

	xmlDoc := `<object>
<entry>
<key>name</key>
<value>Alice</value>
</entry>
</object>`

	ids := p.Encode(xmlDoc, false, false)
	decoded := strings.TrimLeft(p.Decode(ids), " ")

	if decoded != xmlDoc {
		t.Errorf("XML round-trip mismatch:\n  input:   %q\n  decoded: %q", xmlDoc, decoded)
	}

	// Verify the XML tags are present in decoded output (the original bug).
	for _, tag := range []string{"<object>", "<entry>", "<key>", "<value>", "</entry>", "</object>"} {
		if !strings.Contains(decoded, tag) {
			t.Errorf("decoded output missing tag %q", tag)
		}
	}
}

func TestDecodeUserDefinedByID(t *testing.T) {
	p := loadTestModel(t)

	// XML tags are USER_DEFINED pieces at IDs 4-11.
	expected := map[int]string{
		4: "<object>", 5: "</object>",
		6: "<entry>", 7: "</entry>",
		8: "<key>", 9: "</key>",
		10: "<value>", 11: "</value>",
	}

	for id, want := range expected {
		decoded := p.Decode([]int{id})
		if decoded != want {
			t.Errorf("Decode([%d]) = %q, want %q", id, decoded, want)
		}
	}
}
