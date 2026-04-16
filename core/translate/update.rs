use crate::sync::Arc;
use rustc_hash::FxHashMap as HashMap;

use crate::schema::{columns_affected_by_update, ROWID_SENTINEL};
use crate::translate::emitter::Resolver;
use crate::translate::expr::{bind_and_rewrite_expr, BindingBehavior};
use crate::translate::expression_index::expression_index_column_usage;
use crate::translate::plan::{ColumnMask, Operation, Scan};
use crate::translate::planner::{parse_limit, ROWID_STRS};
use crate::{
    bail_parse_error,
    schema::{Schema, Table},
    util::normalize_ident,
    vdbe::builder::{ProgramBuilder, ProgramBuilderOpts},
    CaptureDataChangesExt, Connection,
};
use turso_parser::ast::{self, Expr};

use super::emitter::emit_program;
use super::expr::process_returning_clause;
use super::optimizer::optimize_plan;
use super::plan::{
    ColumnUsedMask, DmlSafety, IterationDirection, JoinedTable, Plan, ResultSetColumn,
    TableReferences, UpdatePlan, WhereTerm,
};
use super::planner::{parse_from, parse_where};
use super::subquery::{
    plan_subqueries_from_returning, plan_subqueries_from_update_sets,
    plan_subqueries_from_where_clause,
};
/*
* Update is simple. By default we scan the table, and for each row, we check the WHERE
* clause. If it evaluates to true, we build the new record with the updated value and insert.
*
* EXAMPLE:
*
sqlite> explain update t set a = 100 where b = 5;
addr  opcode         p1    p2    p3    p4             p5  comment
----  -------------  ----  ----  ----  -------------  --  -------------
0     Init           0     16    0                    0   Start at 16
1     Null           0     1     2                    0   r[1..2]=NULL
2     Noop           1     0     1                    0
3     OpenWrite      0     2     0     3              0   root=2 iDb=0; t
4     Rewind         0     15    0                    0
5       Column         0     1     6                    0   r[6]= cursor 0 column 1
6       Ne             7     14    6     BINARY-8       81  if r[6]!=r[7] goto 14
7       Rowid          0     2     0                    0   r[2]= rowid of 0
8       IsNull         2     15    0                    0   if r[2]==NULL goto 15
9       Integer        100   3     0                    0   r[3]=100
10      Column         0     1     4                    0   r[4]= cursor 0 column 1
11      Column         0     2     5                    0   r[5]= cursor 0 column 2
12      MakeRecord     3     3     1                    0   r[1]=mkrec(r[3..5])
13      Insert         0     1     2     t              7   intkey=r[2] data=r[1]
14    Next           0     5     0                    1
15    Halt           0     0     0                    0
16    Transaction    0     1     1     0              1   usesStmtJournal=0
17    Integer        5     7     0                    0   r[7]=5
18    Goto           0     1     0                    0
*/
pub fn translate_update(
    body: ast::Update,
    resolver: &Resolver,
    program: &mut ProgramBuilder,
    connection: &Arc<crate::Connection>,
) -> crate::Result<()> {
    let mut plan = prepare_update_plan(program, resolver, body, connection, false)?;

    if let Plan::Update(ref mut update_plan) = plan {
        debug_assert!(
            update_plan.ephemeral_plan.is_none(),
            "UPDATE ephemeral plans are introduced during optimization"
        );
        plan_subqueries_from_where_clause(
            program,
            &mut update_plan.non_from_clause_subqueries,
            &mut update_plan.table_references,
            &mut update_plan.where_clause,
            resolver,
            connection,
        )?;
    }

    optimize_plan(program, &mut plan, resolver)?;

    if let Plan::Update(ref update_plan) = plan {
        super::stmt_journal::set_update_stmt_journal_flags(
            program,
            update_plan,
            resolver,
            connection,
        )?;
    }

    let opts = ProgramBuilderOpts {
        num_cursors: 1,
        approx_num_insns: 20,
        approx_num_labels: 4,
    };
    program.extend(&opts);
    emit_program(connection, resolver, program, plan, |_| {})?;
    Ok(())
}

/// Normalize a planned UPDATE RHS into the per-column expressions consumed by SET.
///
/// SQLite keeps row-value RHS expressions intact until subquery planning has
/// expanded their true width. UPDATE does the same here: once the RHS has been
/// planned, multi-column subqueries become a single `SubqueryResult { RowValue }`
/// and can be projected into scalar field reads without cloning the subquery.
fn split_update_set_values(expr: &Expr, target_count: usize) -> crate::Result<Vec<Box<Expr>>> {
    match expr {
        Expr::Parenthesized(vals) => {
            if vals.len() != target_count {
                bail_parse_error!("{} columns assigned {} values", target_count, vals.len());
            }
            Ok(vals.clone())
        }
        Expr::SubqueryResult {
            subquery_id,
            lhs,
            not_in,
            query_type:
                ast::SubqueryType::RowValue {
                    result_reg_start,
                    num_regs,
                },
        } => {
            if *num_regs != target_count {
                bail_parse_error!("{} columns assigned {} values", target_count, num_regs);
            }
            Ok((0..*num_regs)
                .map(|offset| {
                    Box::new(Expr::SubqueryResult {
                        subquery_id: *subquery_id,
                        lhs: lhs.clone(),
                        not_in: *not_in,
                        query_type: ast::SubqueryType::RowValue {
                            result_reg_start: result_reg_start + offset,
                            num_regs: 1,
                        },
                    })
                })
                .collect())
        }
        Expr::Subquery(_) => Err(crate::LimboError::InternalError(
            "UPDATE set clause subquery should be planned before normalization".to_string(),
        )),
        expr => {
            if target_count != 1 {
                bail_parse_error!("{} columns assigned 1 values", target_count);
            }
            Ok(vec![expr.clone().into()])
        }
    }
}

pub fn translate_update_for_schema_change(
    body: ast::Update,
    resolver: &Resolver,
    program: &mut ProgramBuilder,
    connection: &Arc<crate::Connection>,
    ddl_query: &str,
    after: impl FnOnce(&mut ProgramBuilder),
) -> crate::Result<()> {
    let mut plan = prepare_update_plan(program, resolver, body, connection, true)?;

    if let Plan::Update(update_plan) = &mut plan {
        if program.capture_data_changes_info().has_updates() {
            update_plan.cdc_update_alter_statement = Some(ddl_query.to_string());
        }

        debug_assert!(
            update_plan.ephemeral_plan.is_none(),
            "UPDATE ephemeral plans are introduced during optimization"
        );
        plan_subqueries_from_where_clause(
            program,
            &mut update_plan.non_from_clause_subqueries,
            &mut update_plan.table_references,
            &mut update_plan.where_clause,
            resolver,
            connection,
        )?;
    }

    optimize_plan(program, &mut plan, resolver)?;
    let opts = ProgramBuilderOpts {
        num_cursors: 1,
        approx_num_insns: 20,
        approx_num_labels: 4,
    };
    program.extend(&opts);
    emit_program(connection, resolver, program, plan, after)?;
    Ok(())
}

fn validate_update(
    schema: &Schema,
    body: &ast::Update,
    table_name: &str,
    is_internal_schema_change: bool,
    conn: &Arc<Connection>,
) -> crate::Result<()> {
    // Check if this is a system table that should be protected from direct writes
    if !is_internal_schema_change
        && !conn.is_nested_stmt()
        && !conn.is_mvcc_bootstrap_connection()
        && !crate::schema::can_write_to_table(table_name)
    {
        crate::bail_parse_error!("table {} may not be modified", table_name);
    }
    if !body.order_by.is_empty() {
        bail_parse_error!("ORDER BY is not supported in UPDATE");
    }
    // Check if this is a materialized view
    if schema.is_materialized_view(table_name) {
        bail_parse_error!("cannot modify materialized view {}", table_name);
    }

    // Check if this table has any incompatible dependent views
    let incompatible_views = schema.has_incompatible_dependent_views(table_name);
    if !incompatible_views.is_empty() {
        use crate::incremental::compiler::DBSP_CIRCUIT_VERSION;
        bail_parse_error!(
            "Cannot UPDATE table '{}' because it has incompatible dependent materialized view(s): {}. \n\
             These views were created with a different DBSP version than the current version ({}). \n\
             Please DROP and recreate the view(s) before modifying this table.",
            table_name,
            incompatible_views.join(", "),
            DBSP_CIRCUIT_VERSION
        );
    }
    Ok(())
}

pub fn prepare_update_plan(
    program: &mut ProgramBuilder,
    resolver: &Resolver,
    mut body: ast::Update,
    connection: &Arc<crate::Connection>,
    is_internal_schema_change: bool,
) -> crate::Result<Plan> {
    let database_id = resolver.resolve_existing_table_database_id_qualified(&body.tbl_name)?;
    let schema = resolver.schema();
    let table_name = &body.tbl_name.name;
    let table = match resolver.with_schema(database_id, |s| s.get_table(table_name.as_str())) {
        Some(table) => table,
        None => bail_parse_error!("Parse error: no such table: {}", table_name),
    };
    if program.trigger.is_some() && table.virtual_table().is_some() {
        bail_parse_error!(
            "unsafe use of virtual table \"{}\"",
            body.tbl_name.name.as_str()
        );
    }
    let schema_cookie = resolver.with_schema(database_id, |s| s.schema_version);
    program.begin_write_on_database(database_id, schema_cookie);
    validate_update(
        schema,
        &body,
        table_name.as_str(),
        is_internal_schema_change,
        connection,
    )?;

    // Extract WITH, OR conflict clause, and INDEXED BY before borrowing body mutably
    let with = body.with.take();
    let or_conflict = body.or_conflict.take();
    let indexed = body.indexed.take();

    let table_name = table.get_name();

    let joined_tables = vec![JoinedTable {
        table: match table.as_ref() {
            Table::Virtual(vtab) => Table::Virtual(vtab.clone()),
            Table::BTree(btree_table) => Table::BTree(btree_table.clone()),
            _ => unreachable!(),
        },
        identifier: body.tbl_name.alias.as_ref().map_or_else(
            || table_name.to_string(),
            |alias| alias.as_str().to_string(),
        ),
        internal_id: program.table_reference_counter.next(),
        op: build_scan_op(&table),
        join_info: None,
        col_used_mask: ColumnUsedMask::default(),
        column_use_counts: Vec::new(),
        expression_index_usages: Vec::new(),
        database_id,
        indexed,
    }];
    let mut table_references = TableReferences::new(joined_tables, vec![]);
    let has_from_clause = body.from.is_some();
    let mut where_clause = vec![];
    let mut vtab_predicates = vec![];
    parse_from(
        body.from.take(),
        resolver,
        program,
        with,
        true,
        &mut where_clause,
        &mut vtab_predicates,
        &mut table_references,
        connection,
    )?;

    let target_identifier = body.tbl_name.alias.as_ref().map_or_else(
        || normalize_ident(body.tbl_name.name.as_str()),
        |alias| normalize_ident(alias.as_str()),
    );
    let target_table_name = normalize_ident(body.tbl_name.name.as_str());
    let illegal_target_reference = if body.tbl_name.alias.is_some() {
        table_references
            .joined_tables()
            .iter()
            .skip(1)
            .any(|joined| normalize_ident(joined.identifier.as_str()) == target_identifier)
    } else {
        table_references
            .joined_tables()
            .iter()
            .skip(1)
            .any(|joined| {
                normalize_ident(joined.table.get_name()) == target_table_name
                    && normalize_ident(joined.identifier.as_str()) == target_table_name
            })
    };
    if illegal_target_reference {
        bail_parse_error!(
            "target object/alias may not appear in FROM clause: {}",
            body.tbl_name
                .alias
                .as_ref()
                .map_or(body.tbl_name.name.as_str(), |alias| alias.as_str())
        );
    }

    let column_lookup: HashMap<String, usize> = table
        .columns()
        .iter()
        .enumerate()
        .filter_map(|(i, col)| col.name.as_ref().map(|name| (name.to_lowercase(), i)))
        .collect();

    let mut non_from_clause_subqueries = vec![];

    for set in &mut body.sets {
        bind_and_rewrite_expr(
            &mut set.expr,
            Some(&mut table_references),
            None,
            resolver,
            BindingBehavior::ResultColumnsNotAllowed,
        )?;
    }

    plan_subqueries_from_update_sets(
        program,
        &mut non_from_clause_subqueries,
        &mut table_references,
        &mut body.sets,
        resolver,
        connection,
    )?;

    let mut set_clauses: Vec<(usize, Box<Expr>)> = Vec::with_capacity(body.sets.len());

    // Process each SET assignment and map column names to expressions
    // e.g the statement `SET x = 1, y = 2, z = 3` has 3 set assigments
    for set in &mut body.sets {
        let values = split_update_set_values(set.expr.as_ref(), set.col_names.len())?;

        for (col_name, expr) in set.col_names.iter().zip(values.iter()) {
            let ident = normalize_ident(col_name.as_str());

            let col_index = match column_lookup.get(&ident) {
                Some(idx) => {
                    // cannot update generated columns directly
                    table.columns()[*idx].ensure_not_generated("UPDATE", col_name.as_str())?;
                    *idx
                }
                None => {
                    // Check if this is the 'rowid' keyword
                    if ROWID_STRS.iter().any(|s| s.eq_ignore_ascii_case(&ident)) {
                        // Find the rowid alias column if it exists
                        if let Some((idx, _col)) = table
                            .columns()
                            .iter()
                            .enumerate()
                            .find(|(_i, c)| c.is_rowid_alias())
                        {
                            // Use the rowid alias column index
                            match set_clauses.iter_mut().find(|(i, _)| i == &idx) {
                                Some((_, existing_expr)) => existing_expr.clone_from(expr),
                                None => set_clauses.push((idx, expr.clone())),
                            }
                            idx
                        } else {
                            // No rowid alias, use sentinel value for actual rowid
                            match set_clauses.iter_mut().find(|(i, _)| *i == ROWID_SENTINEL) {
                                Some((_, existing_expr)) => existing_expr.clone_from(expr),
                                None => set_clauses.push((ROWID_SENTINEL, expr.clone())),
                            }
                            ROWID_SENTINEL
                        }
                    } else {
                        crate::bail_parse_error!("no such column: {}.{}", table_name, col_name);
                    }
                }
            };
            match set_clauses.iter_mut().find(|(idx, _)| *idx == col_index) {
                Some((_, existing_expr)) => {
                    // When multiple SET col[n] = val for the same column are desugared,
                    // compose them: replace the column reference in the new expression
                    // with the existing expression, so
                    //   col = array_set_element(col, 0, 'X')  then  col = array_set_element(col, 2, 'Z')
                    // becomes col = array_set_element(array_set_element(col, 0, 'X'), 2, 'Z')
                    if let Expr::FunctionCall {
                        name,
                        args: new_args,
                        ..
                    } = expr.as_ref()
                    {
                        if name.as_str().eq_ignore_ascii_case("array_set_element")
                            && new_args.len() == 3
                        {
                            let mut composed_args = new_args.clone();
                            composed_args[0].clone_from(existing_expr);
                            *existing_expr = Box::new(Expr::FunctionCall {
                                name: name.clone(),
                                distinctness: None,
                                args: composed_args,
                                order_by: vec![],
                                filter_over: turso_parser::ast::FunctionTail {
                                    filter_clause: None,
                                    over_clause: None,
                                },
                            });
                        } else {
                            existing_expr.clone_from(expr);
                        }
                    } else {
                        existing_expr.clone_from(expr);
                    }
                }
                None => set_clauses.push((col_index, expr.clone())),
            }
        }
    }

    // Plan subqueries in RETURNING expressions before processing
    // (so SubqueryResult nodes are cloned into result_columns)
    let mut returning_target = table_references.joined_tables()[0].clone();
    // SQLite resolves RETURNING columns for an aliased UPDATE target through the
    // base table name, not the UPDATE alias. Keep the target table in scope, but
    // under its schema name so `RETURNING t.col` works while `RETURNING alias.col`
    // still fails.
    returning_target.identifier = table_name.to_string();
    let mut returning_table_references = TableReferences::new(
        vec![returning_target],
        table_references.outer_query_refs().to_vec(),
    );
    plan_subqueries_from_returning(
        program,
        &mut non_from_clause_subqueries,
        &mut returning_table_references,
        &mut body.returning,
        resolver,
        connection,
    )?;

    let result_columns = process_returning_clause(
        &mut body.returning,
        &mut returning_table_references,
        resolver,
    )?;

    let order_by = vec![];

    // Sqlite determines we should create an ephemeral table if we do not have a FROM clause
    // Difficult to say what items from the plan can be checked for this so currently just checking if a RowId Alias is referenced
    // https://github.com/sqlite/sqlite/blob/master/src/update.c#L395
    // https://github.com/sqlite/sqlite/blob/master/src/update.c#L670
    let columns = table.columns();
    add_vtab_predicates_to_where_clause(
        &mut vtab_predicates,
        &mut table_references,
        &result_columns,
        &mut where_clause,
        resolver,
    )?;
    // Parse the WHERE clause
    parse_where(
        body.where_clause.as_deref(),
        &mut table_references,
        Some(&result_columns),
        &mut where_clause,
        resolver,
    )?;

    // Parse the LIMIT/OFFSET clause
    let (limit, offset) = body
        .limit
        .map_or(Ok((None, None)), |l| parse_limit(l, resolver))?;

    // Determine which indexes need updating
    let indexes: Vec<_> = resolver.with_schema(database_id, |s| {
        s.get_indices(table_name).cloned().collect()
    });
    let target_table_ref = table_references
        .joined_tables()
        .first()
        .expect("UPDATE must have a target table reference");
    let target_table_internal_id = target_table_ref.internal_id;
    let target_table_ref = target_table_ref.clone();
    let rowid_alias_used = set_clauses
        .iter()
        .any(|(idx, _)| *idx == ROWID_SENTINEL || columns[*idx].is_rowid_alias());
    let updated_cols = (!rowid_alias_used).then(|| set_clauses.iter().map(|(i, _)| *i).collect());
    let affected_cols = updated_cols
        .as_ref()
        .map(|updated_cols: &ColumnMask| columns_affected_by_update(columns, updated_cols));
    let mut indexes_to_update = Vec::new();

    for idx in indexes {
        let mut must_update = rowid_alias_used;
        let mut expression_cols_used = ColumnUsedMask::default();

        for col in idx.columns.iter() {
            if let Some(expr) = col.expr.as_ref() {
                let cols_used =
                    expression_index_column_usage(expr.as_ref(), &target_table_ref, resolver)?;
                // if it turns out that we need to update the index, we'll need to use all columns
                // that are used by indexed expressions
                expression_cols_used |= &cols_used;

                if !must_update
                    && affected_cols.as_ref().is_some_and(|affected_cols| {
                        cols_used.iter().any(|cidx| affected_cols.get(cidx))
                    })
                {
                    // an index must be updated if any of the affected columns is used in an indexed expression
                    must_update = true;
                }
            } else if !must_update
                && affected_cols
                    .as_ref()
                    .is_some_and(|affected_cols| affected_cols.get(col.pos_in_table))
            {
                // an index must be updated if any of its columns is affected by the update
                must_update = true;
            }
        }

        if !must_update {
            if let Some(where_expr) = &idx.where_clause {
                let cols_used = expression_index_column_usage(
                    where_expr.as_ref(),
                    &target_table_ref,
                    resolver,
                )?;
                // a partial index must be updated if any column of its WHERE clause is affected
                must_update = affected_cols.as_ref().is_some_and(|affected_cols| {
                    cols_used.iter().any(|cidx| affected_cols.get(cidx))
                });
            }
        }

        if must_update {
            for col_idx in expression_cols_used.iter() {
                table_references.mark_column_used(target_table_internal_id, col_idx);
            }
            indexes_to_update.push(idx);
        }
    }

    Ok(Plan::Update(UpdatePlan {
        table_references,
        has_from_clause,
        or_conflict,
        set_clauses,
        materialized_set_clauses: None,
        where_clause,
        returning: if result_columns.is_empty() {
            None
        } else {
            Some(result_columns)
        },
        order_by,
        limit,
        offset,
        contains_constant_false_condition: false,
        indexes_to_update,
        ephemeral_plan: None,
        cdc_update_alter_statement: None,
        non_from_clause_subqueries,
        safety: DmlSafety::default(),
    }))
}

fn build_scan_op(table: &Table) -> Operation {
    match table {
        Table::BTree(_) => Operation::Scan(Scan::BTreeTable {
            iter_dir: IterationDirection::Forwards,
            index: None,
        }),
        Table::Virtual(_) => Operation::default_scan_for(table),
        _ => unreachable!(),
    }
}

fn add_vtab_predicates_to_where_clause(
    vtab_predicates: &mut Vec<Expr>,
    table_references: &mut TableReferences,
    result_columns: &[ResultSetColumn],
    out_where_clause: &mut Vec<WhereTerm>,
    resolver: &Resolver,
) -> crate::Result<()> {
    for expr in vtab_predicates.iter_mut() {
        bind_and_rewrite_expr(
            expr,
            Some(table_references),
            Some(result_columns),
            resolver,
            BindingBehavior::TryCanonicalColumnsFirst,
        )?;
    }
    for expr in vtab_predicates.drain(..) {
        let from_outer_join = vtab_predicate_table_id(&expr).and_then(|table_id| {
            table_references
                .find_joined_table_by_internal_id(table_id)
                .and_then(|t| {
                    t.join_info
                        .as_ref()
                        .and_then(|ji| ji.is_outer().then_some(table_id))
                })
        });
        out_where_clause.push(WhereTerm {
            expr,
            from_outer_join,
            consumed: false,
        });
    }
    Ok(())
}

fn vtab_predicate_table_id(expr: &Expr) -> Option<ast::TableInternalId> {
    match expr {
        Expr::Binary(lhs, _, _) | Expr::IsNull(lhs) => match lhs.as_ref() {
            Expr::Column { table, .. } => Some(*table),
            _ => None,
        },
        _ => None,
    }
}
