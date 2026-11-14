# Results by point (LLO-2): wRMSE, wMAPE (%) and Relative RMSE w/Naive
# Values and labels are copied from the presentation table; edit `dat` to update the figures.

library(ggplot2)
library(dplyr)
library(tidyr)

models <- c("Naïve (B)", "Hist. Mean (B)", "Pooled priors",
            "Prior x 0.5", "Prior x 0.25", "Prior x 0.1")

dat <- data.frame(
  point = rep(c("8th", "9th"), each = 6),
  model = rep(models, 2),
  wRMSE = c(1719.288, 1581.793, 1603.531, 1477.281, 1419.557, 1381.604,
            6103.640, 7240.093, 7752.617, 7596.234, 7501.292, 7484.671),
  wMAPE = c(24.767, 38.022, 20.722, 20.049, 19.113, 17.973,
            54.811, 62.643, 57.903, 57.701, 57.583, 57.412),
  geoP  = c(1.000, 1.611, 0.890, 0.821, 0.834, 0.727,
            1.000, 1.007, 0.902, 0.939, 0.945, 0.967)
)

panel_labels <- c(wRMSE = "wRMSE",
                  wMAPE = "wMAPE (%)",
                  geoP  = "Relative RMSE w/Naïve")

bar_cols <- setNames(c("grey75", "grey88", "#1d4e7a", "#3f7eaa", "#7aa6c9", "#b0cde3"), models)

make_plot <- function(pt, title) {
  d <- dat |>
    filter(point == pt) |>
    mutate(model = factor(model, levels = models), y = rev(seq_along(models)))

  long <- d |>
    pivot_longer(c(wRMSE, wMAPE, geoP), names_to = "metric", values_to = "value") |>
    mutate(metric = factor(metric, levels = names(panel_labels), labels = panel_labels),
           from   = ifelse(metric == panel_labels[["geoP"]], 1, 0),
           xmin   = pmin(from, value),
           xmax   = pmax(from, value),
           label  = case_when(metric == panel_labels[["wRMSE"]] ~ formatC(value, format = "f", digits = 0, big.mark = ","),
                              metric == panel_labels[["wMAPE"]] ~ sprintf("%.1f", value),
                              TRUE                              ~ sprintf("%.2f", value)))

  lim <- long |>
    group_by(metric) |>
    summarise(top = ifelse(first(metric) == panel_labels[["geoP"]], 1.85, 1.32 * max(value)),
              lo  = ifelse(first(metric) == panel_labels[["geoP"]], 0.6, 0),
              .groups = "drop")

  ggplot(long) +
    geom_blank(data = lim, aes(x = top, y = 1)) +
    geom_blank(data = lim, aes(x = lo,  y = 1)) +
    geom_rect(aes(xmin = xmin, xmax = xmax, ymin = y - 0.36, ymax = y + 0.36, fill = model)) +
    geom_text(data = left_join(long, lim, by = "metric"),
              aes(x = top, y = y, label = label), hjust = 1, size = 3.6, colour = "grey15") +
    facet_wrap(~ metric, nrow = 1, scales = "free_x") +
    scale_y_continuous(breaks = d$y, labels = as.character(d$model), expand = expansion(add = 0.5)) +
    scale_x_continuous(expand = expansion(mult = c(0, 0.01))) +
    scale_fill_manual(values = bar_cols, guide = "none") +
    labs(title = title) +
    theme_classic(base_size = 12) +
    theme(axis.title = element_blank(),
          axis.line.y = element_blank(), axis.ticks.y = element_blank(),
          axis.text.y = element_text(colour = "grey10"),
          strip.background = element_blank(),
          strip.text = element_text(face = "bold", hjust = 0, size = 11),
          panel.spacing.x = unit(1.6, "lines"),
          plot.title = element_text(face = "bold", size = 14),
          plot.title.position = "plot")
}

OUT_DIR <- "C:/Users/ecemy/OneDrive - INSEAD/Desktop/Waste prediction/Code/Draft 1/no_pooling_LLM_informed_priors/plots/presentation"
if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR, recursive = TRUE)
ggsave(file.path(OUT_DIR, "results_8th_point.png"), make_plot("8th", "8th point"),
       width = 11, height = 3.6, dpi = 300, type = "cairo")
ggsave(file.path(OUT_DIR, "results_9th_point.png"), make_plot("9th", "9th point"),
       width = 11, height = 3.6, dpi = 300, type = "cairo")
