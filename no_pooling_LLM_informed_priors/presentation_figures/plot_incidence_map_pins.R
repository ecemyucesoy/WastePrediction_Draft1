suppressPackageStartupMessages({ library(sf); library(ggplot2); library(dplyr); library(ggrepel) })
base <- "C:/Users/ecemy/OneDrive - INSEAD/Desktop/Waste prediction/Code/Draft 1/no_pooling_LLM_informed_priors/"
src <- readLines(paste0(base, "no_pooling_LLM_informed_priors.R"))
i0 <- grep("^district_name_mapping <- c\\(", src); i1 <- i0 + which(grepl("^\\)", src[i0:length(src)]))[1] - 1
eval(parse(text = src[i0:i1]))
name_map <- tibble(District = names(district_name_mapping), Shapefile_District_Name = unname(district_name_mapping))

out <- paste0(base, "outputs/")
hier_out <- "C:/Users/ecemy/OneDrive - INSEAD/Desktop/Waste prediction/Code/Draft 1/hierarchical_weakly_informative_priors/"
files <- c(paste0(hier_out, "outputs/parameter_and_burden_summary_hierarchical.csv"),
           paste0(out, "parameter_and_burden_summary_", c("claude", "gemini", "chatgpt"), "_pooled.csv"))
dfs <- lapply(files, read.csv)
cap <- max(unlist(lapply(dfs, function(d) { v <- d$gamma_count; v[-which.max(v)] })), na.rm = TRUE)
cat("shared cap for count bins:", round(cap), "\n")
d <- dfs[[4]][, c("District", "gamma_count", "gamma")]

n_bins <- 8; step <- cap / n_bins
breaks <- c(seq(0, cap, by = step), Inf)
labels <- c(sapply(seq_len(n_bins), function(i) paste0(round((i - 1) * step / 1000, 1), "k\u2013", round(i * step / 1000, 1), "k")), paste0(">", round(cap / 1000, 1), "k"))
cols <- c(rev(grDevices::hcl.colors(n_bins, "YlGn")), "#FFD700")

shp <- st_read(paste0(base, "../../SOM_adm_2015_July/SOM_adm2.shp"), quiet = TRUE) |> rename(Shapefile_District_Name = NAME_2)
m <- shp |> left_join(name_map |> left_join(d, by = "District"), by = "Shapefile_District_Name") |> filter(!is.na(gamma_count)) |>
  mutate(bin = factor(cut(gamma_count, breaks = breaks, labels = labels, right = FALSE, include.lowest = TRUE), levels = labels))

hi <- c("Banadir", "Ceel Barde", "Baydhaba", "Afgooye", "Doolow")
lo <- c("Xarardheere", "Burco", "Ceel Dheere", "Hobyo", "Ceel Buur")
pins <- m |> filter(District %in% c(hi, lo)) |> mutate(group = ifelse(District %in% hi, "Highest", "Lowest"))
pts <- suppressWarnings(st_point_on_surface(pins))
pts <- cbind(st_drop_geometry(pts), st_coordinates(pts))
cat("pins found:", paste(pts$District, collapse = ", "), "\n")
red <- "#d62728"; green <- "#0f8a5f"

p <- ggplot(m) +
  geom_sf(aes(fill = bin), colour = "black", linewidth = 0.25, show.legend = TRUE) +
  scale_fill_manual(name = "New cases\nper period", values = setNames(cols, labels), limits = labels, breaks = labels, labels = labels, drop = FALSE, na.value = "grey80") +
  geom_point(data = pts, aes(X, Y, colour = group), shape = 19, size = 3.2, show.legend = FALSE) +
  geom_point(data = pts, aes(X, Y), shape = 1, size = 3.2, colour = "white", stroke = 0.8) +
  geom_text_repel(data = pts, aes(X, Y, label = District, colour = group), size = 3.6, fontface = "bold", box.padding = 0.7, point.padding = 0.3,
                  min.segment.length = 0, segment.size = 0.4, max.overlaps = Inf, seed = 11, bg.color = "white", bg.r = 0.12, show.legend = FALSE) +
  scale_colour_manual(values = c(Highest = red, Lowest = green), guide = "none") +
  theme_minimal() + theme(axis.text = element_blank(), axis.ticks = element_blank(), axis.title = element_blank(), panel.grid = element_blank())
dir.create(paste0(base, "plots/presentation"), showWarnings = FALSE, recursive = TRUE)
ggsave(paste0(base, "plots/presentation/incidence_map_chatgpt_pooled_pins.png"), p, width = 10, height = 8, dpi = 300, type = "cairo")
cat("saved\n")
