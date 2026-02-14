#!/usr/bin/env Rscript
# CausalQueries bridge: reads JSON from stdin, runs CQ, writes JSON to stdout.
# Called by pt/cq_bridge.py via subprocess.
#
# Input JSON schema:
#   model_statement: string (DAGitty format, e.g. "X -> M -> Y")
#   data: list of objects (rows), each key=variable_name, value=0/1/null
#   queries: list of query strings for population-level estimands
#   case_level_queries: list of {case_id, row_index, query}
#   restrictions: list of restriction strings (optional)
#   confounds: list of [var1, var2] pairs (optional)
#
# Output JSON schema:
#   population_estimands: list of {query, using, mean, sd, cred_low, cred_high}
#   case_level_estimands: list of {case_id, query, mean, sd, cred_low, cred_high}
#   diagnostics: object with Stan diagnostics

suppressPackageStartupMessages({
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("jsonlite package required: install.packages('jsonlite')")
  }
  if (!requireNamespace("CausalQueries", quietly = TRUE)) {
    stop("CausalQueries package required: install.packages('CausalQueries')")
  }
  library(jsonlite)
  library(CausalQueries)
})

tryCatch({
  # Read JSON from stdin
  input_text <- paste(readLines("stdin", warn = FALSE), collapse = "\n")
  input <- fromJSON(input_text, simplifyVector = FALSE)

  # Build data frame from input, converting NULL to NA
  df_list <- lapply(input$data, function(row) {
    lapply(row, function(val) {
      if (is.null(val)) NA_integer_ else as.integer(val)
    })
  })
  df <- do.call(rbind.data.frame, df_list)

  # Create the causal model
  model <- make_model(input$model_statement)

  # Apply restrictions if any
  if (!is.null(input$restrictions) && length(input$restrictions) > 0) {
    for (restriction in input$restrictions) {
      model <- set_restrictions(model, statement = restriction)
    }
  }

  # Apply confounds if any
  if (!is.null(input$confounds) && length(input$confounds) > 0) {
    for (confound_pair in input$confounds) {
      model <- set_confound(model, confound = list(confound_pair[[1]], confound_pair[[2]]))
    }
  }

  # Update model with data
  model <- update_model(model, df, refresh = 0)

  # Population-level queries
  pop_estimands <- list()
  if (!is.null(input$queries) && length(input$queries) > 0) {
    for (q in input$queries) {
      result <- query_model(model, queries = q, using = "posteriors")
      pop_estimands[[length(pop_estimands) + 1]] <- list(
        query = q,
        given = "",
        using = "posteriors",
        mean = result$mean,
        sd = result$sd,
        cred_low = result$cred.low,
        cred_high = result$cred.high
      )
    }
  }

  # Case-level queries
  case_estimands <- list()
  if (!is.null(input$case_level_queries) && length(input$case_level_queries) > 0) {
    for (clq in input$case_level_queries) {
      row_idx <- clq$row_index
      given_str <- paste(
        sapply(names(df), function(col) {
          val <- df[row_idx, col]
          if (!is.na(val)) paste0(col, "==", val) else NULL
        }),
        collapse = " & "
      )
      # Remove empty entries from NA values
      given_str <- gsub(" & $", "", gsub("^ & ", "", gsub(" &  & ", " & ", given_str)))

      if (nchar(given_str) > 0) {
        result <- tryCatch({
          query_model(model, queries = clq$query, given = given_str, using = "posteriors")
        }, error = function(e) {
          list(mean = NA, sd = NA, cred.low = NA, cred.high = NA)
        })
        case_estimands[[length(case_estimands) + 1]] <- list(
          case_id = clq$case_id,
          query = clq$query,
          mean = result$mean,
          sd = result$sd,
          cred_low = result$cred.low,
          cred_high = result$cred.high
        )
      }
    }
  }

  # Output
  output <- list(
    population_estimands = pop_estimands,
    case_level_estimands = case_estimands,
    diagnostics = list(
      n_cases = nrow(df),
      n_variables = ncol(df)
    )
  )

  cat(toJSON(output, auto_unbox = TRUE, null = "null", na = "null"))

}, error = function(e) {
  cat(toJSON(list(error = conditionMessage(e)), auto_unbox = TRUE))
  quit(status = 1)
})
