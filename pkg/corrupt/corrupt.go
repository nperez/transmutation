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

// Package corrupt applies various corruptions to valid JSON strings to
// simulate the kinds of broken JSON that LLM agents produce.
package corrupt

import "math/rand/v2"

// Corruption is a function that takes valid JSON and returns corrupted JSON.
type Corruption func(json string, rng *rand.Rand) string

// Config controls which corruptions are applied and with what probability.
type Config struct {
	QuoteStripProb   float64
	CommaDropProb    float64
	ColonDropProb    float64
	CommentProb      float64
	WrapperProb      float64
	TrailingProb     float64
	WhitespaceProb   float64
	BracketProb      float64
}

// LightConfig returns a configuration with light corruption.
func LightConfig() Config {
	return Config{
		QuoteStripProb:   0.3,
		CommaDropProb:    0.2,
		ColonDropProb:    0.1,
		CommentProb:      0.2,
		WrapperProb:      0.3,
		TrailingProb:     0.2,
		WhitespaceProb:   0.2,
		BracketProb:      0.10,
	}
}

// MediumConfig returns a configuration with medium corruption.
func MediumConfig() Config {
	return Config{
		QuoteStripProb:   0.5,
		CommaDropProb:    0.4,
		ColonDropProb:    0.2,
		CommentProb:      0.4,
		WrapperProb:      0.5,
		TrailingProb:     0.3,
		WhitespaceProb:   0.4,
		BracketProb:      0.15,
	}
}

// HeavyConfig returns a configuration with heavy corruption.
func HeavyConfig() Config {
	return Config{
		QuoteStripProb:   0.7,
		CommaDropProb:    0.6,
		ColonDropProb:    0.4,
		CommentProb:      0.6,
		WrapperProb:      0.7,
		TrailingProb:     0.5,
		WhitespaceProb:   0.6,
		BracketProb:      0.25,
	}
}

// SubtleConfig returns a configuration with very light corruption —
// the kind of subtle errors LLMs actually produce in production.
// Missing quote, trailing comma, extra brace.
func SubtleConfig() Config {
	return Config{
		QuoteStripProb: 0.08,
		CommaDropProb:  0.05,
		ColonDropProb:  0.0,
		CommentProb:    0.0,
		WrapperProb:    0.0,
		TrailingProb:   0.15,
		WhitespaceProb: 0.0,
		BracketProb:    0.05,
	}
}

// NoneConfig returns a configuration with no corruption (clean passthrough).
func NoneConfig() Config {
	return Config{}
}

// Apply applies a random subset of corruptions based on the config probabilities.
// The order of application is designed to avoid compounding conflicts.
func Apply(json string, cfg Config, rng *rand.Rand) string {
	// Apply in a specific order to minimize interactions:
	// 1. Structural corruptions (quotes, commas, colons, brackets)
	// 2. Additions (comments, trailing commas)
	// 3. Cosmetic (whitespace)
	// 4. Wrapper (preamble/postamble) — always last

	if rng.Float64() < cfg.QuoteStripProb {
		json = StripQuotes(json, rng)
	}
	if rng.Float64() < cfg.CommaDropProb {
		json = DropCommas(json, rng)
	}
	if rng.Float64() < cfg.ColonDropProb {
		json = DropColons(json, rng)
	}
	if rng.Float64() < cfg.BracketProb {
		json = MangleBrackets(json, rng)
	}
	if rng.Float64() < cfg.TrailingProb {
		json = AddTrailingCommas(json, rng)
	}
	if rng.Float64() < cfg.CommentProb {
		json = InsertComments(json, rng)
	}
	if rng.Float64() < cfg.WhitespaceProb {
		json = MangleWhitespace(json, rng)
	}
	if rng.Float64() < cfg.WrapperProb {
		json = AddWrapper(json, rng)
	}

	return json
}
