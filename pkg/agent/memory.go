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

import (
	"fmt"

	"nickandperla.net/transmutation/pkg/languages"
)

var memoryTemplates = []string{
	"User prefers %s for %s. This was established in a previous session when discussing %s.",
	"Previous request was about %s. The user was trying to resolve an issue with %s.",
	"The %s service is running on port %s. It was migrated from %s last week.",
	"Use %s for the %s module. The team standardized on this after evaluating %s.",
	"Remember: %s requires authentication. The token expires every %s hours.",
	"The %s table has %s rows. It is partitioned by %s for performance.",
	"Default timeout is %s seconds. Increase to %s for long-running %s operations.",
	"User's preferred language is %s. They also have experience with %s.",
	"The %s endpoint uses %s method. Rate limited to %s requests per minute.",
	"Last deployment was to %s environment. The %s service had issues during rollout.",
	"The %s feature flag is enabled. It was turned on for %s testing.",
	"User has admin access to %s. They manage the %s and %s modules.",
	"The %s config was updated recently. Changed %s from the default value.",
	"Retry limit is set to %s. Exponential backoff starts at %s milliseconds.",
	"The %s job runs every %s minutes. It processes data from the %s pipeline.",
	"The %s database uses %s replication. Failover is configured for %s.",
	"API version %s is deprecated. Users should migrate to %s by end of quarter.",
	"The %s cache has a TTL of %s seconds. It stores %s data for fast lookups.",
	"Build pipeline for %s takes approximately %s minutes. Includes %s and integration tests.",
	"The %s monitoring alert fires when %s exceeds %s threshold.",
}

var memoryFills = []string{
	"Python", "JavaScript", "Go", "SQL", "REST", "GraphQL",
	"production", "staging", "development", "testing",
	"PostgreSQL", "Redis", "MongoDB", "Elasticsearch",
	"the main", "the backup", "the legacy", "the new",
}

func (g *Generator) generateMemory() []string {
	n := 6 + g.rng.IntN(5) // 6-10 entries

	entries := make([]string, n)
	for i := range entries {
		template := g.pick(memoryTemplates)

		nArgs := 0
		for _, c := range template {
			if c == '%' {
				nArgs++
			}
		}
		// Count actual %s occurrences.
		nArgs = 0
		for j := 0; j < len(template)-1; j++ {
			if template[j] == '%' && template[j+1] == 's' {
				nArgs++
			}
		}

		args := make([]any, nArgs)
		for j := range args {
			switch g.rng.IntN(3) {
			case 0:
				args[j] = g.pick(memoryFills)
			case 1:
				args[j] = languages.RandInt(g.rng)
			default:
				args[j] = g.pick(topics)
			}
		}
		entries[i] = fmt.Sprintf(template, args...)
	}
	return entries
}
