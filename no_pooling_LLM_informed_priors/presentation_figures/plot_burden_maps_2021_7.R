# Single-timepoint burden maps (2021-7): K = 1.6 benchmark vs. ChatGPT pooled model
# Same bins as the faceted burden maps; burden = SAM x (1 + K) for the benchmark.

suppressPackageStartupMessages({ library(sf); library(ggplot2); library(dplyr) })

base <- "C:/Users/ecemy/OneDrive - INSEAD/Desktop/Waste prediction/Code/Draft 1/no_pooling_LLM_informed_priors/"
tp   <- "2021-7"

src <- readLines(paste0(base, "no_pooling_LLM_informed_priors.R"))
i0  <- grep("^district_name_mapping <- c\\(", src); i1 <- i0 + which(grepl("^\\)", src[i0:length(src)]))[1] - 1
eval(parse(text = src[i0:i1]))
name_map <- tibble(District = names(district_name_mapping), Shapefile_District_Name = unname(district_name_mapping))

w <- read.csv(paste0(base, "outputs/burden_comparison_wide.csv"), check.names = FALSE) |> filter(Timepoint == tp)
cat("districts:", nrow(w), "\n")
cat("Total burden", tp, ": K=1.6 =", round(sum(w[["k1.6"]])), "; ChatGPT pooled =", round(sum(w$llm_chatgpt_pooled)),
    "; ratio =", round(sum(w$llm_chatgpt_pooled) / sum(w[["k1.6"]]), 2), "\n")

breaks <- c(0, seq(5000, 50000, by = 5000), Inf)
labels <- c("0-5k", "5k-10k", "10k-15k", "15k-20k", "20k-25k", "25k-30k", "30k-35k", "35k-40k", "40k-45k", "45k-50k", ">50k")
cols   <- rev(grDevices::hcl.colors(length(labels), "YlGn"))   # same green ramp as the other burden maps (>50k = darkest)

shp <- st_read(paste0(base, "../../SOM_adm_2015_July/SOM_adm2.shp"), quiet = TRUE) |> rename(Shapefile_District_Name = NAME_2)

make_map <- function(value_col, file, show_legend) {
  d <- w |> select(District, value = all_of(value_col)) |> left_join(name_map, by = "District")
  m <- shp |> left_join(d, by = "Shapefile_District_Name") |> filter(!is.na(value)) |>
    mutate(bin = factor(cut(value, breaks = breaks, labels = labels, right = FALSE, include.lowest = TRUE), levels = labels))
  p <- ggplot(m) +
    geom_sf(aes(fill = bin), colour = "black", linewidth = 0.25, show.legend = TRUE) +
    scale_fill_manual(name = "Burden Range", values = setNames(cols, labels), limits = labels, breaks = labels, labels = labels, drop = FALSE) +
    theme_minimal() +
    theme(axis.text = element_blank(), axis.ticks = element_blank(), axis.title = element_blank(), panel.grid = element_blank())
  if (!show_legend) {   # keep the same layout, but hide the legend
    p <- p + theme(legend.title = element_text(colour = "white"), legend.text = element_text(colour = "white"),
                   legend.key = element_rect(fill = "white", colour = "white")) +
      guides(fill = guide_legend(override.aes = list(fill = "white", colour = "white")))
  }
  dir.create(paste0(base, "plots/presentation"), showWarnings = FALSE, recursive = TRUE)
  ggsave(paste0(base, "plots/presentation/", file), p, width = 7, height = 8, dpi = 300, type = "cairo")
}
make_map("k1.6",               "burden_map_2021-7_K1.6.png",          FALSE)
make_map("llm_chatgpt_pooled", "burden_map_2021-7_chatgpt_pooled.png", TRUE)
