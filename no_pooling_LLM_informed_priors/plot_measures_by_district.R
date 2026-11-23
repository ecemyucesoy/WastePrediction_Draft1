# Per-district plots of each measure for the held-out points (8th and 9th; 7th too for LLO-3).
# For every scheme and point it writes:
#   - one paneled figure with all measures:      plots/measures_by_district/<scheme>_<point>_all_measures.jpg
#   - one figure per measure (single panel):     plots/measures_by_district/<scheme>_<point>_<measure>.jpg
# Reads the saved CV tables only (no refits, no table is overwritten).
# Panels A-E: one dot per district, x = observed SAM (log scale), so the large districts (outlier candidates) are on the right.
# Panel F: districts sorted by squared error, cumulative share of the total squared error (the basis of RMSE).

suppressPackageStartupMessages({ library(dplyr); library(ggplot2); library(ggrepel); library(cowplot) })

BASE <- "C:/Users/ecemy/OneDrive - INSEAD/Desktop/Waste prediction/Code/Draft 1/no_pooling_LLM_informed_priors/"
OUT  <- file.path(BASE, "plots", "measures_by_district"); dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

SHOW <- c("Naive", "hierarchical", "chatgpt_pooled", "gemini_pooled", "claude_pooled")   # pooled models only
COL  <- c(Naive = "grey60", hierarchical = "grey20", chatgpt_pooled = "#1baf7a", gemini_pooled = "#eb6834", claude_pooled = "#2a78d6")
LAB  <- c(Naive = "Naive", hierarchical = "Hierarchical", chatgpt_pooled = "ChatGPT (pooled)", gemini_pooled = "Gemini (pooled)", claude_pooled = "Claude (pooled)")
SCHEMES <- list(LLO2 = c("8th", "9th"), LLO3 = c("7th", "8th", "9th"))
EXCLUDE_DISTRICTS <- c("Rab Dhuure")   # same districts as left out of the metrics in the main script
NLAB <- 4  # number of largest districts (by observed SAM) to label

theme_m <- theme_minimal(base_size = 11) + theme(legend.position = "none", plot.title = element_text(face = "bold", size = 11))

fmt <- function(x) ifelse(abs(x) >= 100, formatC(x, format = "f", digits = 0, big.mark = ","), formatC(x, format = "f", digits = 2))

# lines: data.frame(Model, y, name) = reported summary value(s) of the measure, drawn as horizontal lines in the model's colour.
# The first name is dashed and labelled with its value on the left edge; a second name (if any) is dotted.
scatter_panel <- function(df, ycol, title, ylab, logy = TRUE, hline = NULL, big, lines = NULL) {
  df$y <- df[[ycol]]
  if (logy) df <- df[is.finite(df$y) & df$y > 0, ]
  p <- ggplot(df, aes(Observed, y, colour = Model)) +
    { if (!is.null(hline)) geom_hline(yintercept = hline, colour = "grey60", linetype = 2) } +
    { if (!is.null(lines)) {
        lines <- lines[lines$Model %in% df$Model & is.finite(lines$y) & (!logy | lines$y > 0), ]
        lines$name <- factor(lines$name, unique(lines$name))
        list(geom_hline(data = lines, aes(yintercept = y, colour = Model, linetype = name), linewidth = 0.7, alpha = 0.9),
             scale_linetype_manual(values = c("dashed", "dotted")[seq_along(levels(lines$name))], name = NULL, guide = "none"),
             # value box in the top-left corner: one row per model, values of the dashed (and dotted) line(s)
             geom_label(data = { v <- lines %>% arrange(Model, name) %>% group_by(Model) %>% summarise(txt = paste(fmt(y), collapse = "  /  "), .groups = "drop")
                                 v$k <- as.numeric(factor(v$Model, SHOW[SHOW %in% v$Model])); v$txt <- paste0(LAB[as.character(v$Model)], ":  ", v$txt); v },
                        aes(x = min(df$Observed), y = Inf, label = txt, colour = Model, vjust = 1.25 * k - 0.05), hjust = 0, size = 2.7, fontface = "bold",
                        fill = scales::alpha("white", 0.85), label.size = 0, label.padding = unit(0.08, "lines"), show.legend = FALSE))
      } } +
    geom_point(alpha = 0.75, size = 1.6) +
    geom_text_repel(data = df[df$Model == "hierarchical" & df$District %in% big, ], aes(label = District), colour = "black", size = 2.6,
                    min.segment.length = 0, box.padding = 0.4, max.overlaps = Inf, seed = 1) +
    scale_colour_manual(values = COL, labels = LAB, name = NULL) +
    scale_x_log10(labels = scales::comma) +
    labs(title = title, x = "Observed SAM (log scale)", y = ylab) + theme_m
  if (logy) p <- p + scale_y_log10(labels = scales::comma) else p <- p + scale_y_continuous(labels = scales::comma)
  p
}

make_panels <- function(scheme, point) {
  d <- read.csv(file.path(BASE, "outputs", paste0(scheme, "_CV_all_models_with_CRPS.csv")), stringsAsFactors = FALSE)
  d <- d[d$Point == point & d$Model %in% SHOW & !d$District %in% EXCLUDE_DISTRICTS, ]
  if (!nrow(d)) return(NULL)
  d$Model <- factor(d$Model, SHOW); d$Error <- d$Predicted - d$Observed
  nv <- d[d$Model == "Naive", c("District", "AbsError")]; names(nv)[2] <- "NaiveAbs"
  d <- left_join(d, nv, by = "District"); d$RelAbs <- ifelse(d$NaiveAbs > 0, d$AbsError / d$NaiveAbs, NA)
  big <- d %>% filter(Model == "Naive") %>% slice_max(Observed, n = NLAB) %>% pull(District)

  cum <- d %>% group_by(Model) %>% arrange(desc(SqError), .by_group = TRUE) %>% mutate(Rank = row_number(), Share = cumsum(SqError) / sum(SqError)) %>% ungroup()
  top <- cum %>% filter(Rank == 1, Model != "Naive") %>% select(Model, District)
  pF <- ggplot(cum, aes(Rank, Share, colour = Model)) + geom_line(linewidth = 0.9) +
    scale_colour_manual(values = COL, labels = LAB, name = NULL) + scale_y_continuous(labels = scales::percent, limits = c(0, 1)) +
    labs(title = "F. Cumulative share of squared error (RMSE)", x = "Districts, largest squared error first", y = "Share of total squared error") + theme_m

  # reported values of the measures, from the same tables the other results come from
  rel <- read.csv(file.path(BASE, "outputs", paste0(scheme, "_CV_relative_scores.csv")), check.names = FALSE)
  rel <- rel[rel$Point == point & rel$Model %in% SHOW, ]
  L <- function(col, nm) data.frame(Model = factor(rel$Model, SHOW), y = rel[[col]], name = nm)
  mm <- d %>% group_by(Model) %>% summarise(MeanSigned = mean(Error), MAPEm = mean(MAPE), CRPSm = mean(CRPS), .groups = "drop")
  Lm <- function(col, nm) data.frame(Model = mm$Model, y = mm[[col]], name = nm)

  list(
    A_abs_error     = scatter_panel(d, "AbsError", "A. Absolute error: dashed = RMSE, dotted = MAD", "Absolute error", TRUE, NULL, big, rbind(L("RMSE", "RMSE"), L("MAD", "MAD"))),
    B_signed_error  = scatter_panel(d, "Error", "B. Signed error: dashed = mean (bias ratio pred/obs in CSV)", "Predicted - observed", FALSE, 0, big, Lm("MeanSigned", "mean signed error")),
    C_abs_pct_error = scatter_panel(d, "MAPE", "C. Absolute % error: dashed = MAPE, dotted = wMAPE", "Absolute % error", TRUE, NULL, big, rbind(Lm("MAPEm", "MAPE"), L("wMAPE_pop", "wMAPE"))),
    D_CRPS          = scatter_panel(d %>% filter(!is.na(CRPS)), "CRPS", "D. CRPS: dashed = mean CRPS", "CRPS (counts)", TRUE, NULL, big, Lm("CRPSm", "mean CRPS")[!is.na(mm$CRPSm), ]),
    E_rel_to_naive  = scatter_panel(d %>% filter(Model != "Naive"), "RelAbs", "E. Abs. error / Naive abs. error: dashed = geoP", "Ratio to Naive (log scale)", TRUE, 1, big, L("RelRMSE_geomean_vsPersistence", "geoP")),
    F_cum_sq_share  = pF)
}

for (scheme in names(SCHEMES)) for (point in SCHEMES[[scheme]]) {
  pl <- make_panels(scheme, point); if (is.null(pl)) next
  leg <- get_legend(pl[[1]] + theme(legend.position = "bottom", legend.text = element_text(size = 10)))
  ttl <- ggdraw() + draw_label(paste0(scheme, ", held-out ", point, " point: every measure by district (pooled models, Naive and hierarchical)"), fontface = "bold", x = 0.01, hjust = 0, size = 13)
  sub <- ggdraw() + draw_label(paste0("Labels: the ", NLAB, " districts with the largest observed SAM. Dots far from the rest on the right of panels A-E are the outlier candidates."), x = 0.01, hjust = 0, size = 9.5, colour = "grey30")
  allp <- plot_grid(ttl, sub, plot_grid(plotlist = unname(pl), ncol = 3), leg, ncol = 1, rel_heights = c(0.05, 0.04, 1, 0.06))
  ggsave(file.path(OUT, paste0(scheme, "_", point, "_all_measures.jpg")), allp, width = 17, height = 10, dpi = 150, bg = "white")
  for (nm in names(pl)) {
    one <- plot_grid(pl[[nm]] + theme(legend.position = "bottom") + labs(subtitle = paste0(scheme, ", held-out ", point, " point")), ncol = 1)
    ggsave(file.path(OUT, paste0(scheme, "_", point, "_", nm, ".jpg")), one, width = 8, height = 5.5, dpi = 150, bg = "white")
  }
  cat("done", scheme, point, "\n")
}
