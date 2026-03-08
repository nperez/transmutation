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

type SQL struct{}

func (SQL) Name() string { return "sql" }

func (SQL) Generate(rng *rand.Rand) string {
	switch rng.IntN(5) {
	case 0:
		return genSelect(rng)
	case 1:
		return genInsert(rng)
	case 2:
		return genUpdate(rng)
	case 3:
		return genDelete(rng)
	default:
		return genSelectWithJoin(rng)
	}
}

var (
	sqlTables  = []string{"users", "orders", "products", "sessions", "logs", "accounts", "payments", "events", "tasks", "messages"}
	sqlColumns = []string{"id", "name", "email", "status", "created_at", "updated_at", "amount", "type", "description", "user_id", "count", "price", "title", "body", "is_active"}
	sqlOps     = []string{"=", "!=", ">", "<", ">=", "<=", "LIKE", "IN", "IS NOT NULL", "IS NULL"}
	sqlValues  = []string{"'active'", "'pending'", "'admin'", "42", "100", "0", "'%test%'", "NOW()", "TRUE", "FALSE", "NULL"}
	sqlAggs    = []string{"COUNT(*)", "SUM(amount)", "AVG(price)", "MAX(created_at)", "MIN(id)"}
	sqlAliases = []string{"u", "o", "p", "s", "t", "a", "e"}
)

func genSelect(rng *rand.Rand) string {
	table := pick(rng, sqlTables)
	cols := pickN(rng, sqlColumns, 2+rng.IntN(4))
	q := fmt.Sprintf("SELECT %s FROM %s", strings.Join(cols, ", "), table)
	if rng.Float64() < 0.7 {
		q += " WHERE " + genWhere(rng)
	}
	if rng.Float64() < 0.3 {
		q += " ORDER BY " + pick(rng, sqlColumns)
		if rng.Float64() < 0.5 {
			q += " DESC"
		}
	}
	if rng.Float64() < 0.3 {
		q += fmt.Sprintf(" LIMIT %d", 10+rng.IntN(90))
	}
	return q
}

func genSelectWithJoin(rng *rand.Rand) string {
	t1 := pick(rng, sqlTables)
	t2 := pick(rng, sqlTables)
	a1 := pick(rng, sqlAliases)
	a2 := pick(rng, sqlAliases)
	if a1 == a2 {
		a2 = a2 + "2"
	}
	cols := fmt.Sprintf("%s.%s, %s.%s", a1, pick(rng, sqlColumns), a2, pick(rng, sqlColumns))
	joinType := "JOIN"
	if rng.Float64() < 0.4 {
		joinType = "LEFT JOIN"
	}
	q := fmt.Sprintf("SELECT %s FROM %s %s %s %s %s ON %s.id = %s.%s",
		cols, t1, a1, joinType, t2, a2, a1, a2, pick(rng, sqlColumns))
	if rng.Float64() < 0.5 {
		q += " WHERE " + a1 + "." + pick(rng, sqlColumns) + " " + pick(rng, sqlOps[:6]) + " " + pick(rng, sqlValues)
	}
	if rng.Float64() < 0.3 {
		q += " GROUP BY " + a1 + "." + pick(rng, sqlColumns)
		if rng.Float64() < 0.5 {
			q += " HAVING " + pick(rng, sqlAggs) + " > " + fmt.Sprintf("%d", rng.IntN(100))
		}
	}
	return q
}

func genInsert(rng *rand.Rand) string {
	table := pick(rng, sqlTables)
	cols := pickN(rng, sqlColumns, 2+rng.IntN(4))
	vals := make([]string, len(cols))
	for i := range vals {
		vals[i] = pick(rng, sqlValues)
	}
	return fmt.Sprintf("INSERT INTO %s (%s) VALUES (%s)",
		table, strings.Join(cols, ", "), strings.Join(vals, ", "))
}

func genUpdate(rng *rand.Rand) string {
	table := pick(rng, sqlTables)
	sets := make([]string, 1+rng.IntN(3))
	for i := range sets {
		sets[i] = pick(rng, sqlColumns) + " = " + pick(rng, sqlValues)
	}
	q := fmt.Sprintf("UPDATE %s SET %s", table, strings.Join(sets, ", "))
	if rng.Float64() < 0.8 {
		q += " WHERE " + genWhere(rng)
	}
	return q
}

func genDelete(rng *rand.Rand) string {
	table := pick(rng, sqlTables)
	return fmt.Sprintf("DELETE FROM %s WHERE %s", table, genWhere(rng))
}

func genWhere(rng *rand.Rand) string {
	clauses := make([]string, 1+rng.IntN(3))
	for i := range clauses {
		col := pick(rng, sqlColumns)
		op := pick(rng, sqlOps)
		switch op {
		case "IS NOT NULL", "IS NULL":
			clauses[i] = col + " " + op
		case "IN":
			vals := pickN(rng, sqlValues[:7], 2+rng.IntN(3))
			clauses[i] = col + " IN (" + strings.Join(vals, ", ") + ")"
		default:
			clauses[i] = col + " " + op + " " + pick(rng, sqlValues)
		}
	}
	joiner := " AND "
	if rng.Float64() < 0.3 {
		joiner = " OR "
	}
	return strings.Join(clauses, joiner)
}
