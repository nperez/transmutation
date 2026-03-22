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

// Package xmlconv converts parsed JSON values into the fixed transmutation XML schema.
// The schema uses six element names: <object>, <entry>, <key>, <value>, <array>,
// plus CDATA sections for values containing XML-special characters.
// No attributes, no declarations, no namespaces.
package xmlconv

import (
	"encoding/json"
	"fmt"
	"strings"
)

// Convert takes a JSON byte slice and returns the corresponding XML string.
func Convert(jsonBytes []byte) (string, error) {
	var raw any
	if err := json.Unmarshal(jsonBytes, &raw); err != nil {
		return "", fmt.Errorf("xmlconv: invalid JSON: %w", err)
	}
	var b strings.Builder
	writeValue(&b, raw, 0)
	return b.String(), nil
}

// ConvertAny takes an already-parsed JSON value and returns XML.
func ConvertAny(v any) string {
	var b strings.Builder
	writeValue(&b, v, 0)
	return b.String()
}

func writeValue(b *strings.Builder, v any, indent int) {
	prefix := strings.Repeat("  ", indent)
	switch val := v.(type) {
	case map[string]any:
		writeObject(b, val, indent)
	case []any:
		writeArray(b, val, indent)
	case string:
		b.WriteString(prefix)
		b.WriteString("<value>")
		writeValueContent(b, val)
		b.WriteString("</value>\n")
	case float64:
		b.WriteString(prefix)
		b.WriteString("<value>")
		b.WriteString(formatNumber(val))
		b.WriteString("</value>\n")
	case bool:
		b.WriteString(prefix)
		b.WriteString("<value>")
		if val {
			b.WriteString("true")
		} else {
			b.WriteString("false")
		}
		b.WriteString("</value>\n")
	case nil:
		b.WriteString(prefix)
		b.WriteString("<value>null</value>\n")
	}
}

func writeObject(b *strings.Builder, obj map[string]any, indent int) {
	prefix := strings.Repeat("  ", indent)
	b.WriteString(prefix)
	b.WriteString("<object>\n")

	// Sort keys for deterministic output.
	keys := sortedKeys(obj)
	for _, key := range keys {
		val := obj[key]
		entryPrefix := strings.Repeat("  ", indent+1)
		b.WriteString(entryPrefix)
		b.WriteString("<entry>\n")

		keyPrefix := strings.Repeat("  ", indent+2)
		b.WriteString(keyPrefix)
		b.WriteString("<key>")
		writeTextContent(b, key)
		b.WriteString("</key>\n")

		// For nested containers, the <value> wraps the container.
		switch nested := val.(type) {
		case map[string]any:
			b.WriteString(keyPrefix)
			b.WriteString("<value>\n")
			writeObject(b, nested, indent+3)
			b.WriteString(keyPrefix)
			b.WriteString("</value>\n")
		case []any:
			b.WriteString(keyPrefix)
			b.WriteString("<value>\n")
			writeArray(b, nested, indent+3)
			b.WriteString(keyPrefix)
			b.WriteString("</value>\n")
		default:
			writeValue(b, val, indent+2)
		}

		b.WriteString(entryPrefix)
		b.WriteString("</entry>\n")
	}

	b.WriteString(prefix)
	b.WriteString("</object>\n")
}

func writeArray(b *strings.Builder, arr []any, indent int) {
	prefix := strings.Repeat("  ", indent)
	b.WriteString(prefix)
	b.WriteString("<array>\n")

	for _, item := range arr {
		switch nested := item.(type) {
		case map[string]any:
			itemPrefix := strings.Repeat("  ", indent+1)
			b.WriteString(itemPrefix)
			b.WriteString("<value>\n")
			writeObject(b, nested, indent+2)
			b.WriteString(itemPrefix)
			b.WriteString("</value>\n")
		case []any:
			itemPrefix := strings.Repeat("  ", indent+1)
			b.WriteString(itemPrefix)
			b.WriteString("<value>\n")
			writeArray(b, nested, indent+2)
			b.WriteString(itemPrefix)
			b.WriteString("</value>\n")
		default:
			writeValue(b, item, indent+1)
		}
	}

	b.WriteString(prefix)
	b.WriteString("</array>\n")
}

// writeTextContent writes a string, using CDATA only when it contains special characters.
// Used for keys (which are safe JSON field names).
func writeTextContent(b *strings.Builder, s string) {
	if needsCDATA(s) {
		writeCDATA(b, s)
	} else {
		b.WriteString(s)
	}
}

// writeValueContent writes a string value, always wrapped in CDATA.
// Always using CDATA eliminates a lookahead problem in autoregressive decoding:
// the model must decide CDATA before seeing content tokens, and can't go back
// to add it if special characters appear later. Unconditional CDATA makes the
// decision trivial — always emit <![CDATA[ after <value>.
func writeValueContent(b *strings.Builder, s string) {
	writeCDATA(b, s)
}

// needsCDATA returns true if the string contains characters that need escaping in XML.
func needsCDATA(s string) bool {
	for _, c := range s {
		if c == '<' || c == '>' || c == '&' || c == '"' || c == '\'' {
			return true
		}
	}
	return strings.Contains(s, "]]>")
}

// writeCDATA writes a string wrapped in CDATA. Handles the edge case where the
// string itself contains "]]>" by splitting into multiple CDATA sections.
func writeCDATA(b *strings.Builder, s string) {
	if !strings.Contains(s, "]]>") {
		b.WriteString("<![CDATA[")
		b.WriteString(s)
		b.WriteString("]]>")
		return
	}
	// Split on "]]>" and rejoin with CDATA section boundaries.
	parts := strings.Split(s, "]]>")
	for i, part := range parts {
		b.WriteString("<![CDATA[")
		b.WriteString(part)
		if i < len(parts)-1 {
			b.WriteString("]]]]><![CDATA[>")
		} else {
			b.WriteString("]]>")
		}
	}
}

func formatNumber(f float64) string {
	if f == float64(int64(f)) && f >= -1e15 && f <= 1e15 {
		return fmt.Sprintf("%d", int64(f))
	}
	return fmt.Sprintf("%g", f)
}

func sortedKeys(m map[string]any) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	// Simple insertion sort — maps are small.
	for i := 1; i < len(keys); i++ {
		for j := i; j > 0 && keys[j] < keys[j-1]; j-- {
			keys[j], keys[j-1] = keys[j-1], keys[j]
		}
	}
	return keys
}
