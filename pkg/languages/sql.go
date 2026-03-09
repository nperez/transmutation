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
	sqlOps  = []string{"=", "!=", ">", "<", ">=", "<=", "LIKE", "IN", "IS NOT NULL", "IS NULL"}
	sqlAggs = []string{"COUNT(*)", "SUM(amount)", "AVG(price)", "MAX(created_at)", "MIN(id)"}
)

func genSelect(rng *rand.Rand) string {
	table := TableName(rng)
	nCols := 2 + rng.IntN(4)
	cols := make([]string, nCols)
	for i := range cols {
		cols[i] = ColumnName(rng)
	}
	q := fmt.Sprintf("SELECT %s FROM %s", strings.Join(cols, ", "), table)
	if rng.Float64() < 0.7 {
		q += " WHERE " + genWhere(rng)
	}
	if rng.Float64() < 0.3 {
		q += " ORDER BY " + ColumnName(rng)
		if rng.Float64() < 0.5 {
			q += " DESC"
		}
	}
	if rng.Float64() < 0.3 {
		q += fmt.Sprintf(" LIMIT %d", 1+rng.IntN(200))
	}
	return q
}

func genSelectWithJoin(rng *rand.Rand) string {
	t1 := TableName(rng)
	t2 := TableName(rng)
	a1 := string(rune('a' + rng.IntN(6)))
	a2 := string(rune('a' + 6 + rng.IntN(6)))

	cols := fmt.Sprintf("%s.%s, %s.%s, %s.%s",
		a1, ColumnName(rng), a2, ColumnName(rng), a1, ColumnName(rng))

	joinType := "JOIN"
	if rng.Float64() < 0.3 {
		joinType = "LEFT JOIN"
	} else if rng.Float64() < 0.15 {
		joinType = "RIGHT JOIN"
	}

	q := fmt.Sprintf("SELECT %s FROM %s %s %s %s %s ON %s.id = %s.%s",
		cols, t1, a1, joinType, t2, a2, a1, a2, ColumnName(rng))
	if rng.Float64() < 0.5 {
		q += " WHERE " + a1 + "." + ColumnName(rng) + " " + pick(rng, sqlOps[:6]) + " " + RandSQLValue(rng)
	}
	if rng.Float64() < 0.3 {
		q += " GROUP BY " + a1 + "." + ColumnName(rng)
		if rng.Float64() < 0.5 {
			q += " HAVING " + pick(rng, sqlAggs) + " > " + RandInt(rng)
		}
	}
	return q
}

func genInsert(rng *rand.Rand) string {
	table := TableName(rng)
	nCols := 2 + rng.IntN(5)
	cols := make([]string, nCols)
	vals := make([]string, nCols)
	for i := range cols {
		cols[i] = ColumnName(rng)
		vals[i] = RandSQLValue(rng)
	}
	return fmt.Sprintf("INSERT INTO %s (%s) VALUES (%s)",
		table, strings.Join(cols, ", "), strings.Join(vals, ", "))
}

func genUpdate(rng *rand.Rand) string {
	table := TableName(rng)
	nSets := 1 + rng.IntN(4)
	sets := make([]string, nSets)
	for i := range sets {
		sets[i] = ColumnName(rng) + " = " + RandSQLValue(rng)
	}
	q := fmt.Sprintf("UPDATE %s SET %s", table, strings.Join(sets, ", "))
	if rng.Float64() < 0.8 {
		q += " WHERE " + genWhere(rng)
	}
	return q
}

func genDelete(rng *rand.Rand) string {
	table := TableName(rng)
	return fmt.Sprintf("DELETE FROM %s WHERE %s", table, genWhere(rng))
}

func genWhere(rng *rand.Rand) string {
	nClauses := 1 + rng.IntN(3)
	clauses := make([]string, nClauses)
	for i := range clauses {
		col := ColumnName(rng)
		op := pick(rng, sqlOps)
		switch op {
		case "IS NOT NULL", "IS NULL":
			clauses[i] = col + " " + op
		case "IN":
			nVals := 2 + rng.IntN(4)
			vals := make([]string, nVals)
			for j := range vals {
				vals[j] = RandSQLValue(rng)
			}
			clauses[i] = col + " IN (" + strings.Join(vals, ", ") + ")"
		default:
			clauses[i] = col + " " + op + " " + RandSQLValue(rng)
		}
	}
	joiner := " AND "
	if rng.Float64() < 0.3 {
		joiner = " OR "
	}
	return strings.Join(clauses, joiner)
}
