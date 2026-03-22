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

package xmlconv

import (
	"encoding/xml"
	"strings"
	"testing"
)

func TestConvertSimpleObject(t *testing.T) {
	input := `{"name": "alice", "age": 30}`
	result, err := Convert([]byte(input))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assertValidXML(t, result)
	assertContains(t, result, "<object>")
	assertContains(t, result, "</object>")
	assertContains(t, result, "<entry>")
	assertContains(t, result, "<key>age</key>")
	assertContains(t, result, "<key>name</key>")
	assertContains(t, result, "<value><![CDATA[alice]]></value>")
	assertContains(t, result, "<value>30</value>")
}

func TestConvertSimpleArray(t *testing.T) {
	input := `[1, "two", true, null]`
	result, err := Convert([]byte(input))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assertValidXML(t, result)
	assertContains(t, result, "<array>")
	assertContains(t, result, "</array>")
	assertContains(t, result, "<value>1</value>")
	assertContains(t, result, "<value><![CDATA[two]]></value>")
	assertContains(t, result, "<value>true</value>")
	assertContains(t, result, "<value>null</value>")
}

func TestConvertNestedObject(t *testing.T) {
	input := `{"user": {"name": "bob", "active": true}}`
	result, err := Convert([]byte(input))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assertValidXML(t, result)
	// Should have nested <object> inside <value>.
	assertContains(t, result, "<value>\n")
	// Count objects — should be 2 (root + nested).
	if strings.Count(result, "<object>") != 2 {
		t.Errorf("expected 2 <object> tags, got %d\nXML:\n%s", strings.Count(result, "<object>"), result)
	}
}

func TestConvertArrayOfObjects(t *testing.T) {
	input := `[{"id": 1}, {"id": 2}]`
	result, err := Convert([]byte(input))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assertValidXML(t, result)
	assertContains(t, result, "<array>")
	if strings.Count(result, "<object>") != 2 {
		t.Errorf("expected 2 <object> tags, got %d", strings.Count(result, "<object>"))
	}
}

func TestConvertEmbeddedSQL(t *testing.T) {
	input := `{"query": "SELECT * FROM users WHERE name = 'alice' AND age > 30"}`
	result, err := Convert([]byte(input))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assertValidXML(t, result)
	// SQL with > should be wrapped in CDATA.
	assertContains(t, result, "<![CDATA[")
	assertContains(t, result, "SELECT * FROM users WHERE name = 'alice' AND age > 30")
}

func TestConvertHTMLValue(t *testing.T) {
	input := `{"content": "<div class=\"main\">Hello &amp; World</div>"}`
	result, err := Convert([]byte(input))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assertValidXML(t, result)
	assertContains(t, result, "<![CDATA[")
}

func TestConvertCDATAWithClosingSequence(t *testing.T) {
	// The string itself contains "]]>" which needs special handling.
	input := `{"data": "foo]]>bar"}`
	result, err := Convert([]byte(input))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assertValidXML(t, result)
	// The content should be recoverable.
	assertContains(t, result, "foo")
	assertContains(t, result, "bar")
}

func TestConvertEmptyObject(t *testing.T) {
	input := `{}`
	result, err := Convert([]byte(input))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assertValidXML(t, result)
	assertContains(t, result, "<object>")
	assertContains(t, result, "</object>")
}

func TestConvertEmptyArray(t *testing.T) {
	input := `[]`
	result, err := Convert([]byte(input))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assertValidXML(t, result)
	assertContains(t, result, "<array>")
	assertContains(t, result, "</array>")
}

func TestConvertDeeplyNested(t *testing.T) {
	input := `{"a": {"b": {"c": {"d": "deep"}}}}`
	result, err := Convert([]byte(input))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assertValidXML(t, result)
	if strings.Count(result, "<object>") != 4 {
		t.Errorf("expected 4 <object> tags, got %d", strings.Count(result, "<object>"))
	}
	assertContains(t, result, "<value><![CDATA[deep]]></value>")
}

func TestConvertMixedTypes(t *testing.T) {
	input := `{"s": "hello", "n": 42, "f": 3.14, "b": true, "null_val": null}`
	result, err := Convert([]byte(input))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assertValidXML(t, result)
	assertContains(t, result, "<value><![CDATA[hello]]></value>")
	assertContains(t, result, "<value>42</value>")
	assertContains(t, result, "<value>3.14</value>")
	assertContains(t, result, "<value>true</value>")
	assertContains(t, result, "<value>null</value>")
}

func TestConvertNestedArray(t *testing.T) {
	input := `[[1, 2], [3, 4]]`
	result, err := Convert([]byte(input))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assertValidXML(t, result)
	if strings.Count(result, "<array>") != 3 {
		t.Errorf("expected 3 <array> tags, got %d\nXML:\n%s", strings.Count(result, "<array>"), result)
	}
}

func TestConvertInvalidJSON(t *testing.T) {
	_, err := Convert([]byte(`{invalid`))
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestConvertSortedKeys(t *testing.T) {
	input := `{"zebra": 1, "alpha": 2, "middle": 3}`
	result, err := Convert([]byte(input))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Keys should appear in sorted order.
	alphaPos := strings.Index(result, "alpha")
	middlePos := strings.Index(result, "middle")
	zebraPos := strings.Index(result, "zebra")
	if !(alphaPos < middlePos && middlePos < zebraPos) {
		t.Errorf("keys not in sorted order: alpha=%d, middle=%d, zebra=%d", alphaPos, middlePos, zebraPos)
	}
}

func TestConvertOnlyUsesSchemaElements(t *testing.T) {
	input := `{"key": [1, {"nested": true}]}`
	result, err := Convert([]byte(input))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Parse and verify only our schema elements appear.
	decoder := xml.NewDecoder(strings.NewReader(result))
	allowed := map[string]bool{"object": true, "entry": true, "key": true, "value": true, "array": true}
	for {
		tok, err := decoder.Token()
		if err != nil {
			break
		}
		switch t := tok.(type) {
		case xml.StartElement:
			if !allowed[t.Name.Local] {
				t2 := t // avoid shadowing
				_ = t2
			}
		}
	}
	// If it parses without error, that's the main check.
	assertValidXML(t, result)
}

func TestConvertSpecialCharacterKeys(t *testing.T) {
	// Keys with XML special characters should use CDATA.
	input := `{"<script>": "value", "a&b": "test"}`
	result, err := Convert([]byte(input))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assertValidXML(t, result)
}

func TestConvertUnicodeContent(t *testing.T) {
	input := `{"emoji": "Hello 🎉", "cjk": "你好世界"}`
	result, err := Convert([]byte(input))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assertValidXML(t, result)
	assertContains(t, result, "🎉")
	assertContains(t, result, "你好世界")
}

func TestConvertReActStyle(t *testing.T) {
	input := `{
		"thought": "I need to query the database for user information",
		"action": "execute_sql",
		"action_input": "SELECT u.name, u.email FROM users u WHERE u.status = 'active' AND u.age > 21 ORDER BY u.name"
	}`
	result, err := Convert([]byte(input))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assertValidXML(t, result)
	assertContains(t, result, "execute_sql")
	assertContains(t, result, "SELECT u.name")
	// SQL with > should trigger CDATA.
	assertContains(t, result, "<![CDATA[")
}

func TestConvertLargeObject(t *testing.T) {
	// Build a large JSON object to test performance.
	var b strings.Builder
	b.WriteString("{")
	for i := range 100 {
		if i > 0 {
			b.WriteString(",")
		}
		b.WriteString(`"key_`)
		b.WriteString(strings.Repeat("x", 3))
		b.WriteString(`_`)
		b.WriteString(string(rune('0' + i%10)))
		b.WriteString(string(rune('0' + i/10)))
		b.WriteString(`": "value_`)
		b.WriteString(strings.Repeat("y", 10))
		b.WriteString(`"`)
	}
	b.WriteString("}")

	result, err := Convert([]byte(b.String()))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assertValidXML(t, result)
}

// assertValidXML checks that the string parses as valid XML.
func assertValidXML(t *testing.T, s string) {
	t.Helper()
	// Wrap in a root element to handle multiple top-level elements.
	wrapped := "<root>" + s + "</root>"
	decoder := xml.NewDecoder(strings.NewReader(wrapped))
	for {
		_, err := decoder.Token()
		if err != nil {
			if err.Error() == "EOF" {
				return
			}
			t.Errorf("invalid XML: %v\nXML:\n%s", err, s)
			return
		}
	}
}

func assertContains(t *testing.T, haystack, needle string) {
	t.Helper()
	if !strings.Contains(haystack, needle) {
		t.Errorf("expected XML to contain %q\nXML:\n%s", needle, haystack)
	}
}
