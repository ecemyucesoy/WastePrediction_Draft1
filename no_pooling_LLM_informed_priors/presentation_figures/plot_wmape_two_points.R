# wMAPE (%) at the 8th and 9th held-out points: benchmarks vs. LLM-informed (pooled priors)
# Values are copied from the presentation table; edit `dat` to update the figure.

library(ggplot2)

models <- c("Naive\n(B)", "Hist. Average\n(B)", "LLM-informed")

dat <- data.frame(
  point = rep(c("8th point", "9th point"), each = 3),
  model = factor(rep(models, 2), levels = models),
  wmape = c(24.767, 38.022, 20.722,
            54.811, 62.643, 57.903)
)

p <- ggplot(dat, aes(model, wmape, fill = model)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = sprintf("%.1f", wmape)), vjust = -0.5, size = 4) +
  facet_wrap(~ point, nrow = 1, strip.position = "bottom") +
  scale_fill_manual(values = setNames(c("grey75", "grey88", "#1d4e7a"), models), guide = "none") +
  scale_y_continuous(limits = c(0, 70), expand = expansion(mult = c(0, 0.02))) +
  labs(y = "wMAPE (%)", x = NULL) +
  theme_classic(base_size = 13) +
  theme(strip.background = element_blank(),
        strip.placement = "outside",
        strip.text = element_text(face = "bold", size = 13, margin = margin(t = 8)),
        panel.spacing.x = unit(3, "lines"),
        axis.text.x = element_text(colour = "grey10", size = 10.5),
        axis.title.y = element_text(margin = margin(r = 8)),
        axis.line.x = element_line(colour = "grey30"),
        axis.ticks.x = element_blank())

OUT_DIR <- "C:/Users/ecemy/OneDrive - INSEAD/Desktop/Waste prediction/Code/Draft 1/no_pooling_LLM_informed_priors/plots/presentation"
if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR, recursive = TRUE)
ggsave(file.path(OUT_DIR, "wmape_8th_9th.png"), p, width = 9.5, height = 4.8, dpi = 300, type = "cairo")
